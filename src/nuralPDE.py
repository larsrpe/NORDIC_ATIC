from neurodiffeq.conditions import DirichletBVP2D
from neurodiffeq.solvers import Solver2D
from neurodiffeq.monitors import Monitor2D
from neurodiffeq.generators import Generator2D
from neurodiffeq.networks import FCNN 
from neurodiffeq import diff 
import torch

import matplotlib.pyplot as plt

import numpy as np
import math



def rho_dot(x,y) -> float:
    #p1 = (1-t/10)
    #p2 = 1-p1
    u1 = torch.tensor([5,3.5])
    u2 = torch.tensor([5,6.5])
    sigma = 1
    var = sigma**2
    return -1/(2*math.pi*var)*torch.exp(-1/(2*var)*((x-u1[0])**2+(y-u1[1])**2))+ 1/(2*math.pi*var)*torch.exp(-1/(2*var)*((x-u2[0])**2+(y-u2[1])**2))

def rho_f(x,y) -> float:
    p1 = 0.9
    p2 = 0.1
    u1 = torch.tensor([5,3.5])
    u2 = torch.tensor([5,6.5])
    sigma = 1
    var = sigma**2
    return p1/(2*math.pi*var)*torch.exp(-1/(2*var)*((x-u1[0])**2+(y-u1[1])**2))+ p2/(2*math.pi*var)*torch.exp(-1/(2*var)*((x-u2[0])**2+(y-u2[1])**2)) 

def rho_grad_x(x,y):
    p1 = 0.9
    p2 = 0.1
    u1 = torch.tensor([5,3.5])
    u2 = torch.tensor([5,6.5])
    sigma = 1
    var = sigma**2
    return -p1/(2*math.pi*var**2)*torch.exp(-1/(2*var)*((x-u1[0])**2+(y-u1[1])**2))*(x-u1[0]) -p2/(2*math.pi*var**2)*torch.exp(-1/(2*var)*((x-u2[0])**2+(y-u2[1])**2))*(x-u2[0])

def rho_grad_y(x,y):
    p1 = 0.9
    p2 = 0.1
    u1 = torch.tensor([5,3.5])
    u2 = torch.tensor([5,6.5])
    sigma = 1
    var = sigma**2
    return -p1/(2*math.pi*var**2)*torch.exp(-1/(2*var)*((x-u1[0])**2+(y-u1[1])**2))*(y-u1[1]) -p2/(2*math.pi*var**2)*torch.exp(-1/(2*var)*((x-u2[0])**2+(y-u2[1])**2))*(y-u2[1])




cont_eq = lambda u,v,x,y: [
    rho_grad_x(x,y)*u + rho_grad_y(x,y)*v + rho_f(x,y)*(diff(u, x, order=1)* + diff(v, y, order=1)) + rho_dot(x,y)
]

# Define the boundary conditions
# There's only one function to be solved for, so we only have a single condition
conditions = [
    DirichletBVP2D(
        x_min=0, x_min_val=lambda y: 0,
        x_max=10, x_max_val=lambda y: 0,
        y_min=0, y_min_val=lambda x: 0,
        y_max=10, y_max_val=lambda x: 0,
    ),
    DirichletBVP2D(
        x_min=0, x_min_val=lambda y: 0,
        x_max=10, x_max_val=lambda y: 0,
        y_min=0, y_min_val=lambda x: 0,
        y_max=10, y_max_val=lambda x: 0,
    )
]


# Define the neural network to be used
# Again, there's only one function to be solved for, so we only have a single network
nets = [
    FCNN(n_input_units=2, n_output_units=1, hidden_units=[512]),
    FCNN(n_input_units=2, n_output_units=1, hidden_units=[512])
]

# Define the monitor callback
monitor=Monitor2D(check_every=10, xy_min=(0, 0), xy_max=(1, 1))
monitor_callback = monitor.to_callback()

# Instantiate the solver
solver = Solver2D(
    pde_system=cont_eq,
    conditions=conditions,
    xy_min=(0, 0),  # We can omit xy_min when both train_generator and valid_generator are specified
    xy_max=(10, 10),  # We can omit xy_max when both train_generator and valid_generator are specified
    nets=nets,
    train_generator=Generator2D((50, 50), (0, 0), (10, 10), method='equally-spaced-noisy'),
    valid_generator=Generator2D((50, 50), (0, 0), (10, 10), method='equally-spaced'),
)

# Fit the neural network
solver.fit(max_epochs=200, callbacks=[monitor_callback])

# Obtain the solution
solution = solver.get_solution()

xs = np.linspace(0,10,50)
ys = np.linspace(0,10,50)
xx,yy = np.meshgrid(xs,ys)


rho = rho_f(torch.from_numpy(xx),torch.from_numpy(yy)).numpy()
uu,vv = solution(xx,yy,to_numpy=True)


print(uu.shape,vv.shape)

fig = plt.figure(2)
plt.quiver(xx.flatten(),yy.flatten(),uu.flatten()*rho.flatten(),vv.flatten()*rho.flatten())
plt.show()

