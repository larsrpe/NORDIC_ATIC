import torch
import numpy as np
import os

from src.sim import sim
from src.densitycontrollers import GausianController,WalkingManController
from src.gmminterpolator import GMMInterpolator
from src.densities import GMM
from src.viz import viz_sim,viz_pdf

def demo_gaussian():
    N=50 #number of agents
    L=10 #size of domain
    T = 3 #end time
    t_start = 1 #time when desired density becomes a function of time
    h = L/20 #smoothing param of density estimation
    D = 5 #feedback gain
    X0 = torch.rand(N,2)*torch.tensor([1.5*L/4,L/4])+torch.tensor([L/2*(1-1.5/4),L/8]) #initial position
    t_eval = np.linspace(0,T,int(60*T)+1) #timesteps where we get the simulation results
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'mps':
        device = 'cpu' #for some reason mps is much slower than cpu on M1 
    
    print("using device: ",device)
    
    for use_ff in [False,True]:
        controller = GausianController(L,h,D,t_start,t_start+0.5,use_ff=use_ff).to(device)
        t,y = sim(controller,X0,t_eval)
        str = 'with' if use_ff else 'without'
        viz_sim(t,y,L,f"demo_gaussian_{str}_feed_forward")


def demo_walking_man():
    N=300 #number of agents
    L=10 #size of domain
    T = 3 #end time
    t_start = 1 #time when desired density becomes a function of time
    h = L/20 #smoothing param of density estimation
    D = 5 #feedback gain
    X0 = torch.tensor([4,L])*torch.rand(N, 2) + torch.tensor([3,0])#initial position
    t_eval = np.linspace(0,T,int(60*T)+1) #timesteps where we get the simulation results
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'mps':
        device = 'cpu' #for some reason mps is much slower than cpu on M1 
    
    print("using device: ",device)
    #run sims
    for use_ff in [False,True]:
        controller = WalkingManController(L,h,D,t_start,t_start+0.5,resolution=(64,64),use_ff=use_ff).to(device)
        t,y = sim(controller,X0,t_eval)
        str = 'with' if use_ff else 'without'
        viz_sim(t,y,L,f"demo_walking_man_{str}_feed_forward")

    #viz the desired density
    gmm_inter = GMMInterpolator.walking_man(0.5,L,(64,64))
    gmm_inter.load_coeff('data/walking_man/resolution64')
    def pdf(t,x):
        x = x.float()
        gmm = GMM(*gmm_inter.get_params(torch.tensor(t)),sigma = L/64)
        return gmm.eval_batch(x)
    viz_pdf(pdf,t_eval,L,100,'demo_walking_man_interpolation')

    

if __name__ == "__main__":
    if not os.path.exists('sims'):
        os.mkdir('sims')
    demo_gaussian()
    demo_walking_man()

   


   





    


