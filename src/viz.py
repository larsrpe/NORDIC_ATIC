from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from functools import partial
import torch
import numpy as np
from typing import Callable

def viz_sim(t,y,L,file_name = "test"):
    fig = plt.figure() 
   
    # marking the x-axis and y-axis
    axis = plt.axes(xlim =(0,L), 
                    ylim =(0,L)) 
  
    # initializing a line variable
    plot, = axis.plot([], [], "ko", markersize=5)

    
    # data which the line will 
    # contain (x, y)
    def init(): 
        plot.set_data([], [])
        return plot,
    
    def animate(i,Y):
        X = Y[:,i].reshape(-1,2)
        x = X[:,0]
        y = X[:,1]
        plot.set_data(x, y)
        return plot,

    anim = FuncAnimation(fig, partial(animate,Y=y), init_func = init,
                     frames = len(t), interval = 20, blit = True)
  
    writer = FFMpegWriter(fps = 20)
    anim.save(f'sims/{file_name}.mp4',writer = writer)


def viz_vectorfield(Field: Callable[[float,torch.Tensor],torch.Tensor],t,L,num_points: int,file_name = "test"):
    fig = plt.figure() 
   
    # marking the x-axis and y-axis
    axis = plt.axes(xlim =(0,L), 
                    ylim =(0,L)) 
  
    # initializing a line variable

    
    # data which the line will 
    # contain (x, y)
    
    def animate(t):
        xs = np.linspace(0,L,num_points)
        ys = np.linspace(0,L,num_points)
        us = np.zeros(num_points**2)
        vs = np.zeros(num_points**2)
        for i,x in enumerate(xs):
            for j,y in enumerate(ys):
                u,v = tuple(Field(t,torch.tensor([x,y]).double()))
                us[i*num_points+j] = u.item()
                vs[i*num_points+j] = v.item()

        
        xx,yy = np.meshgrid(xs,ys)
        return plt.quiver(xx,yy,us,vs)

    anim = FuncAnimation(fig, animate,
                     frames = len(t), interval = 1/5, blit = False)
  
    writer = FFMpegWriter(fps = 5)
    anim.save(f'sims/vectorfield_{file_name}.mp4',writer = writer)
