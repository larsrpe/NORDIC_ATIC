import torch
from torch import nn
import numpy as np
import cvxpy as cp
import scipy
from matplotlib import pyplot as plt
from typing import Tuple
import math


class ContEQSolver:
    """
    PDE solver that solves the continuety equation:

    DIV(rho*v) = -rho_dot => rho_dx*v1 + rho_dy*v2 + rho*DIV(v) = -rho_dot
    
    for a velocity field V=[V1;V2] on a square domain given the density 'Rho',
    its temporal derivative 'Rho_dot' and its gradient 'Rho_grad' evaluated at the domian at some time t.
        
    L: dimension of the square domain(grid)
    Rho: array of shape (N+1,N+1) where N+1 Rho_ij corresponds to rho evaluated at x = L/N*j and y = L/N*i 
    Rho_dot: array of shape (N+1,N+1) where N+1 Rho_dot_ij corresponds to rho_dot evaluated at x = L/N*j and y = L/N*i
    Rho_grad: array of shape (N+1,N+1,2) where N+1 Rho_ij corresponds to the spatial gradient of eho evaluated at x = L/N*j and y = L/N*i 
    """
    
    def __init__(self,L: float, Rho: np.ndarray,Rho_dot: np.ndarray, Rho_grad: np.ndarray) -> None:
        self.L = L
        self.N = Rho.shape[0]
        self.h = L/N
        self.V1 = None
        self.V2 = None

        self.Rho = Rho
        self.Rho_dot = Rho_dot
        self.Rho_grad = Rho_grad
        self.xs = np.linspace(0,L,N+1)
        self.ys = np.linspace(0,L,N+1)


    def get_spars_AB(self):
        "calculates the system matrixes Ax=B for every grid point using central differences as gradient approximation."

        cols,rows = self.N+1, self.N+1
        xx,yy = np.meshgrid(self.xs[1:-1],self.ys[1:-1])
        
        Dx = scipy.sparse.diags([1, -1], [2, 0], shape=(cols - 2, cols)).tocsc()
        Dy = scipy.sparse.diags([1, -1], [2, 0], shape=(rows-2, rows)).tocsc()
        diag = scipy.sparse.diags([1], [1], shape=(cols-2, cols)).tocsc()
        dx = scipy.sparse.kron(diag,Dx)/(self.h*2)
        dy = scipy.sparse.kron(Dy,diag)/(self.h*2) 
        
        #borderconditions Bg = 0
        border_grid = np.zeros((rows,cols))
        border_grid[:,0]=1
        border_grid[:,-1]=1
        border_grid[0,:]=1
        border_grid[-1,:]=1
        B_numpy = np.diag(border_grid.flatten())
        B_numpy = B_numpy[~np.all(B_numpy == 0, axis=1)]
        B = scipy.sparse.csc_matrix(B_numpy)
        
        #extend to both grids
        Border = scipy.sparse.block_diag((B,B))
        #div = DIV@[g1;g2]
        DIV = scipy.sparse.hstack([dx,dy])
        #rho
        xx,yy = np.meshgrid(self.xs[1:-1],self.ys[1:-1])
        rho = rho_f(xx,yy)
        #rho*DIV
       
        rhoDiag = scipy.sparse.csc_matrix(np.diag(rho.flatten()))
        rhoDIV = rhoDiag@DIV
        
        #grad_rho * V
        


        # get border idx 
        border_grid = np.zeros((rows,cols))
        border_grid[:,0]=1
        border_grid[:,-1]=1
        border_grid[0,:]=1
        border_grid[-1,:]=1

        not_border_mask = border_grid.flatten()==0

        xx,yy = np.meshgrid(self.xs,self.ys)
        grad_x = rho_grad_x(xx,yy)
        grad_x_diag = np.diag(grad_x.flatten())
        grad_y = rho_grad_y(xx,yy)
        grad_y_diag = np.diag(grad_y.flatten())

       
        #remove zero rows
        grad_x_diag=grad_x_diag[not_border_mask,:]
        grad_y_diag=grad_y_diag[not_border_mask,:]
    
        grad = scipy.sparse.hstack([scipy.sparse.csc_matrix(grad_x_diag),scipy.sparse.csc_matrix(grad_y_diag)])
        
        # continuety
        print(grad.shape,rhoDIV.shape,rho.shape)
        A_cont = grad + rhoDIV
        #solve Ax=B
        A = scipy.sparse.vstack([A_cont,Border])
        #create B
        B = np.zeros(A.shape[0])
        xx,yy = np.meshgrid(self.xs[1:-1],self.ys[1:-1])
        Rho_dot = dot(xx,yy)
        B[:Rho_dot.size] = -Rho_dot.flatten()
        return A,B


    def solve(self,t: float):
        rows,cols = self.N+1,self.N+1
        A,B = self.get_spars_AB()
        v_vec = scipy.sparse.linalg.lsqr(A,B)[0]
        self.G1 = v_vec[:rows*cols].reshape(rows,cols)
        self.G2 = v_vec[rows*cols:].reshape(rows,cols)

    def check_sol(self):
        for i in range(1,self.N):
            for j in range(1,self.N):
                x,y = self.xs[i],self.ys[j]
                #g1 = self.G1.detach().numpy()[0,j,i]
                #g2 = self.G2.detach().numpy()[0,j,i]
                v1 = self.G1[j,i]
                v2 = self.G2[j,i]
                rho = rho_f(x,y)
                rho_dot = dot(x,y)
                v1x = (self.G1[j,i+1]-self.G1[j,i-1])/(2*self.h)
                v2y = (self.G2[j+1,i]-self.G2[j-1,i])/(2*self.h)
                rhox = rho_grad_x(x,y)
                rhoy = rho_grad_y(x,y)

                cont = rhox*v1 + rhoy*v2 + rho*(v1x+v2y) + rho_dot
                k =0
                if np.abs(cont) > 0.001:
                    k+=1
                    #print("kuk",cont)
        
        print(k)


    def get_field(self,n: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        xx = np.zeros((self.N+1)**2)
        yy = np.zeros((self.N+1)**2)
        uu = np.zeros((self.N+1)**2)
        vv = np.zeros((self.N+1)**2)
        step = int(self.N/n)
        k = 0
        for i in range(0,self.N+1,step):
            for j in range(0,self.N+1,step):
                x,y = self.xs[i],self.ys[j]
                #g1 = self.G1.detach().numpy()[0,j,i]
                #g2 = self.G2.detach().numpy()[0,j,i]
                g1 = self.G1[j,i]
                g2 = self.G2[j,i]
                norm = math.sqrt(g1**2 + g2**2)
                xx[k] = x
                yy[k] = y
                rho = rho_f(x,y)
                uu[k] = rho*g1 #/(norm) if norm > 5 else g1
                vv[k] = rho*g2 #/(norm)*5 if norm >5 else g2
                k+=1
        

        return xx[:k],yy[:k],uu[:k],vv[:k]
    
def dot(x,y) -> float:
    p1 = 0.9
    p2 = 0.1
    u1 = np.array([5,3.5])
    u2 = np.array([5,5.5])
    sigma = 1
    var = sigma**2
    return -1/(2*math.pi*var)*np.exp(-1/(2*var)*((x-u1[0])**2+(y-u1[1])**2))+ 1/(2*math.pi)*np.exp(-1/(2*var)*((x-u2[0])**2+(y-u2[1])**2))

def rho_f(x,y) -> float:
    p1 = 0.9
    p2 = 0.1
    u1 = np.array([5,3.5])
    u2 = np.array([5,5.5])
    sigma = 1
    var = sigma**2
    return p1/(2*math.pi*var)*np.exp(-1/(2*var)*((x-u1[0])**2+(y-u1[1])**2))+ p2/(2*math.pi)*np.exp(-1/(2*var)*((x-u2[0])**2+(y-u2[1])**2)) 

def rho_grad_x(x,y):
    p1 = 0.9
    p2 = 0.1
    u1 = np.array([5,3.5])
    u2 = np.array([5,5.5])
    sigma = 1
    var = sigma**2
    return -p1/(2*math.pi*var**2)*np.exp(-1/2*((x-u1[0])**2+(y-u1[1])**2))*(x-u1[0]) -p2/(2*math.pi*var**2)*np.exp(-1/2*((x-u2[0])**2+(y-u2[1])**2))*(x-u2[0])

def rho_grad_y(x,y):
    p1 = 0.9
    p2 = 0.1
    u1 = np.array([5,3.5])
    u2 = np.array([5,5.5])
    sigma = 1
    var = sigma**2
    return -p1/(2*math.pi*var**2)*np.exp(-1/2*((x-u1[0])**2+(y-u1[1])**2))*(y-u1[1]) -p2/(2*math.pi*var**2)*np.exp(-1/2*((x-u2[0])**2+(y-u2[1])**2))*(y-u2[1])


if __name__ == "__main__":
    #test_diff()
    T = 10
    L = 10
    N = 250
    solver = ContEQSolver(L,N)
    x= torch.tensor([30,10]).double()
    solver.solve_rho(0.1)
    #solver.solve_pytorch(t=0.1)
    xx,yy,uu,vv = solver.get_field(n=50)
    plt.quiver(xx,yy,uu,vv)
    plt.savefig("./images/fd_pde.jpg")
    solver.check_sol()





                