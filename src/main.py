import torch
import math
import numpy as np


from densities import GridGMM,GaussianPDF,TimevaryingParams
from densitycontrollers import GMMController,ImshowController,VideoController
from fields import LarsField, VelocityField
from sim import sim
from viz import viz_sim,viz_vectorfield,viz_sim_with_forces,viz_pdf





if __name__ == "__main__":
    
    N=300
    L=10
    T = 3
    t_start =1
    h = L/20
    D = 5

    #fd = GridGMM.from_image('roy_safari.png',L,sigma_pixels=2)
    #fd.plot(t=0,points=300,filename='roy_test')


  
    #means = torch.tensor([[5,3.5],
    #                      [5,6.5]])
    #
    #w_start = torch.tensor([1,0])
    #w_end = torch.tensor([0,1])

    #controller = GMMController(means,w_start,w_end,t_start,t_end,h,D,L)

    #controller = ImshowController(t_start,t_end,h,D,L)
    controller = VideoController(t_start,h,D,L)

    def vectorfield(t,x):
        return controller.get_feedforward(t,x) #*controller.f_d.eval(t,x)

    
    controller.f_d.plot(t=0,points=300,filename='wm_plot')
    
    
    X0 = torch.tensor([4,L])*torch.rand(N, 2) + torch.tensor([3,0])
    t_eval = np.linspace(0,T,int(20*T)+1)
    viz_vectorfield(vectorfield,t_eval,L,65,"man_walking_ff_test")
    viz_pdf(controller.f_d.eval,t_eval,L,100,"man_walking_test")
    t,y = sim(controller,X0,t_eval)
    viz_sim(t,y,L,'man_walking_with_ff_test')
    viz_sim_with_forces(t,vectorfield,y,L,'man_walking_with_ff_test')
    

    
    


