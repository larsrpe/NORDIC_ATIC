from typing import List,Tuple

import os
import torch
import scipy
import numpy as np
import cv2 as cv
from PIL import Image

from src.image_utils import image_to_pdf_args

class GMMInterpolator:
    def __init__(self,t_start: float,dt: float,ps: List[torch.Tensor], ms: List[torch.Tensor]) -> None:
        self.t_start = t_start
        self.dt = dt
        self.num_poinst = len(ps)
        self.t_end = t_start + dt*self.num_poinst
        self.ps = ps
        self.ms = ms
        self.PIs: List[torch.Tensor] = []
        self.last_timestep = -1
        self.last_idx = None

    def extend(self, x: int):
        "extend the time axis by repeating"
        self.ms*=x
        self.PIs*=x
        self.ps*=x
        self.num_poinst = len(self.ps)
    def speedup(self,x:int):
        "speedup or truncate the time axes"
        self.dt/=x
        self.t_end = self.t_start + self.dt*self.num_poinst
    def interpolate(self): 
        "do the interpolation"
        for i in range(self.num_poinst-1):
            p0 = self.ps[i]
            p1 = self.ps[i+1]
            m0 = self.ms[i]
            m1 = self.ms[i+1]
            self.PIs.append(self._iterpolate_between(p0,p1,m0,m1))
        
        print("interpolation done")

    def _iterpolate_between(self,p0: torch.Tensor,p1: torch.Tensor,m0: torch.Tensor,m1: torch.Tensor) -> torch.Tensor:
        "interpolate between two gmms"
        # setup optimization problem
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

        #solve the optimization problem
        #this optimization problem is always feasable (use PI = p0p1')
        #however the constraint matrix A is rank deficient so numerically we might encounter infeasabilities
        #this is solved by removing some of the constraints, since some often are redundant 
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
                A_cons.resize(A_cons.shape[0]-4,A_cons.shape[1]) #remove 4 rows for better numrical proparties
                B_cons = B_cons[:-4]
        print(sol.message)
        pi_vec = sol.x
        idx = (pi_vec == 0)
        pi = pi_vec.reshape(N0,N1)
        print("res:", np.linalg.norm(pi.sum(axis=1)-p0.numpy().flatten()) + np.linalg.norm(pi.sum(axis=0)-p1.numpy().flatten()))
        print("sparse",len(pi_vec[idx]),"of",len(pi_vec))
        return torch.from_numpy(pi)


    def get_params(self,t: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        "returns the params of the interpoalted gmm"
        r = t-self.t_start
        time_step = int(r.item()/self.dt)
        if r < 0:
            return self.ms[0],self.ps[0].to(torch.float32)
        elif time_step > self.num_poinst-2:
            return self.ms[-1],self.ps[-1].to(torch.float32)
        PI = self.PIs[time_step]
        if time_step != self.last_timestep:
            self.last_timestep = time_step
            self.last_idx = PI.flatten()>0

        idx = self.last_idx
        m0 = self.ms[time_step]
        m1 = self.ms[time_step+1]
        time_since_last_step = (r-time_step*self.dt)
        s = time_since_last_step/self.dt
        m0_ext = m0.repeat_interleave(PI.size(0),0)[idx]
        m1_ext = m1.repeat(PI.size(1),1)[idx]
        means = (1-s)*m0_ext + s * m1_ext
        weights = PI.flatten()[idx]
        return means,weights.to(torch.float32)   
    
    def save_coeff(self,path: str):
        str = f'{path}.pt'
        if not os.path.exists(os.path.dirname(str)):
            os.mkdir(os.path.dirname(str))
        torch.save(self.PIs,str)
    def load_coeff(self,path: str) -> "GMMInterpolator":
        str = f'{path}.pt'
        pis=torch.load(str)
        self.PIs = pis
        
    
    @classmethod
    def walking_man(cls,t_start:float,L:int,resolution: Tuple[int,int]= (64,64)) -> "GMMInterpolator":
        "Make a interpolation based on gmm between every frame of the walking man video"
        pil_frames = []
        video_T = 2#sec
        cap = cv.VideoCapture('videos/man_walking.mp4')
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
            p = p.reshape(-1)
            m = means.reshape(-1,2)
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
