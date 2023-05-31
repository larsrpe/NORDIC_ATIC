import torch
import math
import numpy as np


from densities import GridGMM,GaussianPDF,TimevaryingParams
from fields import LarsField, VelocityField
from sim import sim
from viz import viz_sim,viz_vectorfield


def sine_means() -> torch.Tensor:
    u1 = torch.tensor([3,6])
    u2 = torch.tensor([L/3,L-L/3])
    u3 = torch.tensor([L-L/3,L-L/3])
    u4 =  torch.tensor([6,3])
    means = torch.zeros(2,2,2)
    means[0,0,:] = u2
    means[0,1,:] = u3
    means[1,1,:] = u4
    means[1,0,:] = u1
    return means

def sine_weights(t: float) -> torch.Tensor:
    #omega = 1/(T-t_start)*math.pi
    #w1_0=0
    #w2_0=1
    #w3_0=0
    #w4_0=1
    #if t < t_start:
    #    h = 0
    #else: h = t-t_start
    #w2_t = w2_0*(math.sin(omega/2*h))**2
    #w1_t = w1_0*(math.cos(omega/2*h))**2
    #w3_t = w3_0*(math.sin(omega/2*h))**2
    #w4_t = w4_0*(math.cos(omega/2*h))**2
    #return torch.tensor([[w2_t,w3_t],
    #                     [w1_t,w4_t]]).double()
    w1 = torch.Tensor([[0,0],[0,1]]).double()
    w2 = torch.Tensor([[1,0],[0,0]]).double()

    if t < t_start:
        return w1
    h=(t-t_start)/(T-t_start)
    return (w1*(1-h) + h*w2).double()

def sine_weights_dot(t: float):
    #omega = 1/(T-t_start)*2*math.pi
    #w1_0=0
    #w2_0=1
    #w3_0=0
    #w4_0=1
#
    #if t < t_start:
    #    return torch.zeros(2,2).double()
    #else: h = t-t_start
    #w2_t_dot = w2_0*omega*math.sin(omega/2*h)*math.cos((omega/2*h))
    #w1_t_dot = -w1_0*omega*math.sin(omega/2*h)*math.cos((omega/2*h))
    #w3_t_dot = w3_0*omega*math.sin(omega/2*h)*math.cos((omega/2*h))
    #w4_t_dot = -w4_0*omega*math.sin(omega/2*h)*math.cos((omega/2*h))
    #return torch.tensor([[w2_t_dot,w3_t_dot],
    #                     [w1_t_dot,w4_t_dot]]).double()


    if t < t_start:
        return torch.zeros(2,2).double()
    
    w1 = torch.Tensor([[0,0],
                       [0,1]]).double()
    w2 = torch.Tensor([[1,0],
                       [0,0]]).double()
    return (-w1/(T-t_start) + w2/(T-t_start)).double()



if __name__ == "__main__":
    
    N=100
    L=9
    T = 10
    h = L/20
    D = 5
    t_start = 5

    mu_0 = torch.Tensor([3,3])
    mu_T = torch.Tensor([13,13])
    
    params = TimevaryingParams(sine_weights,sine_weights_dot)
    means = sine_means()
    f_d = GridGMM(means,params,L)

    #for t in np.linspace(0,T,50):
    #    f_d.plot(t,100,"test_sine")
    #    w = sine_weights(t)
    #    w_dot = sine_weights_dot(t)
    #    print(w.sum(),w_dot.sum())
    #    print(t)

    


    #f_d = GaussianPDF(1,0,T,mu_0,mu_0)
    #f_d.plot(0,500,"eth_plot")
    #f_d.plot_grid()

    LF = LarsField(f_d,h,D)
    VF = VelocityField(f_d,h,D)

    t_eval = np.linspace(0, T, T*5+1, endpoint=True)

    field = f_d.g
    #print(field(5,(torch.tensor([L-L/3,L/3])+1).double()))
    
    viz_vectorfield(field,t=t_eval,L=L,num_points=50)


    
    #X0 = L*torch.rand(N, 2)
    #
    #t,y = sim(LF,X0,t_eval)

    #print("sim done")
    #viz_sim(t,y,L,'test_sine_LF')


