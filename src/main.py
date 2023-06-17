import torch
import math
import numpy as np

from sim import sim
from densitycontrollers import WalkingManController,GausianController
from gmminterpolator import GMMInterpolator
from viz import viz_sim,viz_vectorfield,viz_sim_with_forces,viz_pdf,viz_tracking





if __name__ == "__main__":
    
    N=300
    L=10
    T = 3
    t_start = 1
    h = L/20
    D = 5

    
    controller = WalkingManController(L,h,D,1,t_start,resolution=(64,64),use_ff=True)
    #controller = GausianController(L,h,D,t_start,t_start+0.5,use_ff=True)
    

    
    
    
    def pdf(t,x):
        controller.update_refrence(t)
        return controller.fd.eval_batch(x)
        

    X0 = torch.tensor([4,L])*torch.rand(N, 2) + torch.tensor([3,0])
    #X0 = torch.rand(N,2)*torch.tensor([1.5*L/4,L/4])  +torch.tensor([L/2*(1-1.5/4),L/8])
    t_eval = np.linspace(0,T,int(60*T)+1)
    viz_pdf(pdf,t_eval,L,50,"wm after rebase")
    t,y = sim(controller,X0,t_eval)
    viz_sim(t,y,L,"wm after rebase")
    #viz_tracking(t,y,L,pdf,100,"tracking gaussian after rebase")
    #viz_sim_with_forces(t,vectorfield,y,L,'man_walking_with_ff_test')





    


