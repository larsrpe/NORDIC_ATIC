import torch
import cvxpy as cp
import scipy
import numpy as np
import pickle
from typing import List,Tuple
import cv2 as cv

from image_utils import image_to_pdf_args
from PIL import Image




class GMMInterpolator:

    def __init__(self,t_start: float,dt: float,ps: List[torch.Tensor], ms: List[torch.Tensor]) -> None:
        self.t_start = t_start
        self.dt = dt
        self.num_poinst = len(ps)
        self.t_end = t_start + dt*self.num_poinst
        self.ps = ps
        self.ms = ms
        self.PIs: List[torch.Tensor] = []

    def extend(self, x: int):
        self.ms*=x
        self.PIs*=x
        self.ps*=x
        self.num_poinst = len(self.ps)
    def speedup(self,x:int):
        self.dt/=x
        self.t_end = self.t_start + self.dt*self.num_poinst
    
    def interpolate(self): 
        for i in range(self.num_poinst-1):
            p0 = self.ps[i]
            p1 = self.ps[i+1]
            m0 = self.ms[i]
            m1 = self.ms[i+1]
            self.PIs.append(self.iterpolate_between(p0,p1,m0,m1))
        
        print("interpolation done")


    def iterpolate_between(self,p0: torch.Tensor,p1: torch.Tensor,m0: torch.Tensor,m1: torch.Tensor) -> torch.Tensor:
        N0 = m0.shape[0]
        N1 = m1.shape[0]
        c = torch.cdist(m0,m1).numpy().flatten()**2

        diag = scipy.sparse.diags([1], [0], shape=(N0, N0)).tocsc()
        ones = scipy.sparse.csc_matrix(np.ones((1,N1)))
        A1 = scipy.sparse.kron(diag,ones,format='csc')

        diag = scipy.sparse.diags([1], [0], shape=(N1, N1)).tocsc()
        ones = scipy.sparse.csc_matrix(np.ones((1,N0)))
        A2 = scipy.sparse.kron(ones,diag,format='csc')


        A_cons = scipy.sparse.vstack((A1,A2))
        B_cons = np.concatenate((p0.flatten(),p1.flatten()))
       

       
        solved = False
        i = 0

        A_cons.resize(A_cons.shape[0]-1,A_cons.shape[1]) #remove one row for better numrical proparties
        B_cons = B_cons[:-1]

        while not solved:
            print(i)
            i+=1
            sol = scipy.optimize.linprog(c,A_eq=A_cons, b_eq=B_cons, method='highs')
            solved =sol.success
            if not solved:
                A_cons.resize(A_cons.shape[0]-4,A_cons.shape[1]) #remove one 4row for better numrical proparties
                B_cons = B_cons[:-4]
            

        print(sol.message)

        
        pi_vec = sol.x
        idx = (pi_vec == 0)
        pi = pi_vec.reshape(N0,N1)

        print("res:", np.linalg.norm(pi.sum(axis=1)-p0.numpy().flatten()) + np.linalg.norm(pi.sum(axis=0)-p1.numpy().flatten()))
        print("sparse",len(pi_vec[idx]),"of",len(pi_vec))
        return torch.from_numpy(pi)



    def get_weights(self,t: float) -> torch.Tensor:

        t = t-self.t_start
        time_step = int(t/self.dt)
        if time_step < 0:
            time_step = 0
            t=0
        elif time_step >= self.num_poinst-2:
            time_step = self.num_poinst-3
            t = time_step*self.dt

        PI = self.PIs[time_step].flatten()
        idx = PI>0
        PI = PI[idx]
        return PI

    
    def get_means(self,t: float) -> torch.Tensor:
        t = t-self.t_start
        
        time_step = int(t/self.dt)
        if time_step < 0:
            time_step = 0
            t=0
        elif time_step >= self.num_poinst-2:
            time_step = self.num_poinst-3
            t = time_step*self.dt

        PI = self.PIs[time_step].flatten()
        idx = PI>0
        m0 = self.ms[time_step]
        m1 = self.ms[time_step+1]
        h = (t-time_step*self.dt)/self.dt
    
        meansx = ((1-h)*(m0[:,0]).reshape(1,-1) + h*(m1[:,0]).reshape(-1,1)).flatten()[idx]
        meansy = ((1-h)*(m0[:,1]).reshape(1,-1) + h*(m1[:,1]).reshape(-1,1)).flatten()[idx]

        means = torch.concatenate((meansx[:,None],meansy[:,None]),axis=1)

        return means.double()
    
    def get_means_dot(self,t: float) -> torch.Tensor:
        t = t-self.t_start
        
        time_step = int(t/self.dt)
        if time_step < 0:
            time_step = 0
            t=0
        elif time_step >= self.num_poinst-2:
            time_step = self.num_poinst-3
            t = time_step*self.dt

        PI = self.PIs[time_step].flatten()
        idx = PI>0
        m0 = self.ms[time_step]
        m1 = self.ms[time_step+1]
        h_dot = 1/self.dt
    
        means_dot_x = (-h_dot*(m0[:,0]).reshape(1,-1) + h_dot*(m1[:,0]).reshape(-1,1)).flatten()[idx]
        means_dot_y = (-h_dot*(m0[:,1]).reshape(1,-1) + h_dot*(m1[:,1]).reshape(-1,1)).flatten()[idx]

        means_dot = torch.concatenate((means_dot_x[:,None],means_dot_y[:,None]),axis=1)

        return means_dot.double()
    
    def save_coeff(self,filename: str):
        str = f'./data/{filename}.pt'
        torch.save(self.PIs,str)
    

    def load_coeff(self,filename: str) -> "GMMInterpolator":
        str = f'./data/{filename}.pt'
        pis=torch.load(str)
        self.PIs = pis
        
    
    @classmethod
    def walking_man(cls,t_start:float,L:int,resolution: Tuple[int,int]= (128,128)) -> "GMMInterpolator":
        pil_frames = []
        video_T = 2#sec
        cap = cv.VideoCapture('./videos/man_walking.mp4')
        started = False
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                started = True
                pil_frames.append(Image.fromarray(np.uint8(frame)))
            if not ret and started:
                break
        
        pil_frames = pil_frames[::2]
        dt = video_T/(len(pil_frames)-1)
        ps = []
        ms = [] 
        ps = []
        ms = [] 
        for frame in pil_frames:
            means,p = image_to_pdf_args(frame,L,resolution)
            p = p.reshape(-1).double()
            m = means.reshape(-1,2).double()
            ps.append(p)
            ms.append(m)

        return cls(t_start,dt,ps,ms)


        
    

if __name__ == "__main__":
    N = 64*64
    ms,ps = [],[]
    #m = 10*torch.rand(20,2)

    for i in range(2):
        p = torch.rand(N)
        ps.append(p/(p.sum()))
        ms.append(10*torch.rand(N,2))


    GMMInterpolator(0,0.2,ps,ms)
