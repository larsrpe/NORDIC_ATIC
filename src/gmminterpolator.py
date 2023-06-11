import torch
import cvxpy as cp
import scipy
import numpy as np
import pickle
from typing import List




class GMMInterpolator:

    def __init__(self,t_start: float,dt: float,ps: List[torch.Tensor], ms: List[torch.Tensor],sigma=1) -> None:
        self.t_start = t_start
        self.dt = dt
        self.num_poinst = len(ps)
        self.t_end = t_start + dt*self.num_poinst
        self.ps = ps
        self.ms = ms
        self.var = sigma**2
    
        self.PIs: List[torch.Tensor] = []

        for i in range(self.num_poinst-1):
            p0 = ps[i]
            p1 = ps[i+1]
            m0 = ms[i]
            m1 = ms[i+1]
            self.PIs.append(self.iterpolate_between1(p0,p1,m0,m1))
        
        print("interpolation done")


    def iterpolate_between(self,p0: torch.Tensor,p1: torch.Tensor,m0: torch.Tensor,m1: torch.Tensor) -> torch.Tensor:
        N0 = m0.shape[0]
        N1 = m1.shape[0]
        PI = cp.Variable((N0,N1))
        C = torch.cdist(m0,m1).numpy()**2

        cost = 0
        constr = []

        eta = 1e2

        constr.append(cp.sum(PI, axis=1) == p0.numpy())
        constr.append(cp.sum(PI[:,:-1], axis=0) == p1.numpy()[:-1])
        constr +=[PI>=0]
        cost += eta*(cp.sum(PI[:,-1])-p1.numpy()[-1])**2
        cost += cp.sum(cp.multiply(PI,C))
        #cost += cp.norm1(PI)
        
        

        problem = cp.Problem(cp.Minimize(cost),constr)
        problem.solve(verbose=True,solver='SCS',eps_abs=1e-4, eps_rel=1e-4)
        print(problem.status)

        pi = PI.value
        print("res=", np.linalg.norm(pi.sum(axis=1)-p0.numpy()) + np.linalg.norm(pi.sum(axis=0)-p1.numpy()))
        print("sum=", pi.sum())
        print("sparsity:", (pi[pi>0]).size)
        return torch.from_numpy(PI.value)
    
    def iterpolate_between1(self,p0: torch.Tensor,p1: torch.Tensor,m0: torch.Tensor,m1: torch.Tensor) -> torch.Tensor:
        N0 = m0.shape[0]
        N1 = m1.shape[0]
        print(N0,N1)
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
        while not solved:
            print(i)
            i+=1
            A_cons.resize(A_cons.shape[0]-1,A_cons.shape[1]) #remove one row for better numrical proparties
            B_cons = B_cons[:-1]
            sol = scipy.optimize.linprog(c,A_eq=A_cons, b_eq=B_cons, method='highs')
            solved =sol.success
            

        print(sol.message)

        
        pi_vec = sol.x
        idx = (pi_vec == 0)
        pi = pi_vec.reshape(N0,N1)

        print("res:", np.linalg.norm(pi.sum(axis=1)-p0.numpy().flatten()) + np.linalg.norm(pi.sum(axis=0)-p1.numpy().flatten()))
        print("sparse",len(pi_vec[idx]),"of",len(pi_vec))
        return torch.from_numpy(pi)



    def get_weights(self,t: float,x,y) -> torch.Tensor:

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

    
    def get_means(self,t: float,x,y) -> torch.Tensor:
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
    
    def save(self,filename: str):
        picklefile = open(f'./data/+{filename}', 'wb')
        pickle.dump(self, picklefile)
        picklefile.close()
    
    @classmethod
    def load_from_file(cls,filename: str) -> "GMMInterpolator":
        picklefile = open(f'./data/{filename}', 'wb')
        x = pickle.load(picklefile)
        picklefile.close()
        return x

        
    

if __name__ == "__main__":
    N = 64*64
    ms,ps = [],[]
    #m = 10*torch.rand(20,2)

    for i in range(2):
        p = torch.rand(N)
        ps.append(p/(p.sum()))
        ms.append(10*torch.rand(N,2))


    GMMInterpolator(0,0.2,ps,ms)
