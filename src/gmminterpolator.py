import torch
import cvxpy as cp
import scipy
import numpy as np
import ecos

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
            self.PIs.append(self.iterpolate_between(p0,p1,m0,m1))
        
        print("interpolation done")


    def iterpolate_between(self,p0: torch.Tensor,p1: torch.Tensor,m0: torch.Tensor,m1: torch.Tensor) -> torch.Tensor:
        N0 = m0.shape[0]
        N1 = m1.shape[0]
        PI = cp.Variable((N0,N1))
        C = torch.cdist(m0,m1).numpy()**2

        cost = 0
        constr = []

        constr += [cp.sum(PI, axis=0) == p0.numpy()]
        constr += [cp.sum(PI, axis=1) == p1.numpy()]
        constr +=[PI>=0]
        cost += cp.sum(cp.multiply(PI,C))

        problem = cp.Problem(cp.Minimize(cost),constr)
        problem.solve(verbose=True,solver='SCS')
        print(problem.status)
        return torch.from_numpy(PI.value)
    
    def iterpolate_between1(self,p0: torch.Tensor,p1: torch.Tensor,m0: torch.Tensor,m1: torch.Tensor) -> torch.Tensor:
        N0 = m0.shape[0]
        N1 = m1.shape[0]
        print(N0,N1)
        c = torch.cdist(m0,m1).numpy().flatten()

        diag = scipy.sparse.diags([1], [0], shape=(N0, N0)).tocsc()
        ones = scipy.sparse.csc_matrix(np.ones((1,N1)))
        A1 = scipy.sparse.kron(diag,ones,format='csc')

        diag = scipy.sparse.diags([1], [0], shape=(N1, N1)).tocsc()
        ones = scipy.sparse.csc_matrix(np.ones((1,N0)))
        A2 = scipy.sparse.kron(ones,diag,format='csc')


        A = scipy.sparse.vstack((A1,A2)) 
        B = np.concatenate((p0.flatten(),p1.flatten()))

        #print(A.toarray())

        #pi =  np.kron(p0.reshape(-1,1),p1.reshape(1,-1))
        #return torch.from_numpy(pi)
        #pi_vec = pi.flatten()

        #print(A.shape,B.shape,N0*N1)
    
        opt = {'primal_feasibility_tolerance':1e-2,
                   'dual_feasibility_tolerance':1e-2,
                   'ipm_optimality_tolerance': 1e-2,
                   'presolve': True
                   }
        
        #opt = {'tol':1e-6,
        #       'presolve': True}

        #print(p1.sum(),p0.sum())

        #sol1 = scipy.optimize.linprog(c,A_eq=A, b_eq=B, method='highs',options=opt)#tol = 1e-2)
        #ecos.solve(c,G,h,dims,A,B)
        #print(sol1.success)
        #print(np.linalg.norm(sol1.con))
        #print(sol1.message)

        #sol2 =scipy.sparse.linalg.lsmr(A, B)



        #print(sol2[3])
    
       
        pi_vec = sol1.x 
        pi = pi_vec.reshape(N0,N1)
        print(pi.sum())
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
        print(len(PI))
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
        return means
        
    

if __name__ == "__main__":
    N = 200
    ms,ps = [],[]
    #m = 10*torch.rand(20,2)

    for i in range(4):
        p = torch.rand(N)
        ps.append(p/(p.sum()))
        ms.append(10*torch.rand(N,2))


    GMMInterpolator(0,0.2,ps,ms)
