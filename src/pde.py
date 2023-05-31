import torch
from torch import nn
import numpy as np
import cvxpy as cp
import scipy
from matplotlib import pyplot as plt
from typing import Tuple
import math

from densities import TimevaryingPDF, TimevaryingParams, GridGMM



class ContEQSolver:
    def __init__(self,L: float, rho: TimevaryingPDF,N:int) -> None:
        self.L = L
        self.rho = rho
        self.N = N
        self.h = L/N
        self.G1 = cp.Variable((N+1,N+1))
        self.G2 = cp.Variable((N+1,N+1))
        self.xs = np.linspace(0,L,N+1)
        self.ys = np.linspace(0,L,N+1)

    def solve_sparse(self,t: float):
        cols,rows = self.N+1, self.N+1
        
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
        #solve Ax=B
        #create A
        A = scipy.sparse.vstack([DIV,Border])
        #create B
        B = np.zeros(A.shape[0])
        xx,yy = np.meshgrid(self.xs[1:-1],self.ys[1:-1])
        Rho_dot = dot(xx,yy)
        B[:Rho_dot.size] = -Rho_dot.flatten()

        g_vec = scipy.sparse.linalg.lsqr(A,B)[0]
        self.G1 = g_vec[:rows*cols].reshape(rows,cols)
        self.G2 = g_vec[rows*cols:].reshape(rows,cols)

    def get_spars_AB(self):
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
        xx,yy = np.meshgrid(self.xs,self.ys)


        # get border idx 
        border_grid = np.zeros((rows,cols))
        border_grid[:,0]=1
        border_grid[:,-1]=1
        border_grid[0,:]=1
        border_grid[-1,:]=1

        not_border_mask = border_grid.flatten()==0

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


    def solve_rho(self,t: float):
        rows,cols = self.N+1,self.N+1
        A,B = self.get_spars_AB()
        #v_vec = cp.Variable(2*rows*cols)
        
        #constr = []
        
        #constr+=[A@v_vec==B]
        #cost = cp.norm2(v_vec)

        #problem = cp.Problem(cp.Minimize(cost),constraints=constr)
        #sol = problem.solve(verbose=False,canon_backend=cp.SCIPY_CANON_BACKEND)
        #print(problem.status)
        #print(sol)
        v_vec = scipy.sparse.linalg.lsqr(A,B)[0]
        self.G1 = v_vec[:rows*cols].reshape(rows,cols)
        self.G2 = v_vec[rows*cols:].reshape(rows,cols)




    def solve(self,t: float):
        constr = []
        cost = 0
        for i in range(1,self.N):
            for j in range(1,self.N):
                
                #compute central difference
                g1x = (self.G1[j,i+1]-self.G1[j,i-1])/(2*self.h)
                g1y = (self.G1[j+1,i]-self.G1[j-1,i])/(2*self.h)
                g2x = (self.G2[j,i+1]-self.G1[j,i-1])/(2*self.h)
                g2y = (self.G2[j+1,i]-self.G1[j-1,i])/(2*self.h)

                x,y = self.xs[i],self.ys[j]
                rho_dot = dot(x,y)
                #contiuety equation
                constr += [g1x +g2y == -rho_dot]  #,g2x -g1y == 0]
                #minimize curl
                #cost += (g1y-g2x)**2 #(g1x+g2y+rho_dot)**2 #+ 

        for j in range(self.N+1):
            constr += [
                self.G1[0,j] == 0,
                self.G1[-1,j] == 0,
                self.G1[j,0] == 0,
                self.G1[j,0-1] == 0,
                self.G2[0,j] == 0,
                self.G2[-1,j] == 0,
                self.G2[j,0] == 0,   
                self.G2[j,-1] ==0
                ]
            
        #minimize norm
        #cost += cp.pnorm(self.G1) + cp.pnorm(self.G2)

        problem = cp.Problem(cp.Minimize(cost),constraints=constr)
        sol = problem.solve(verbose=False)
        print(problem.status)
        print(sol)

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
    w1 = torch.Tensor([[0,0],[0,1]]).double()
    w2 = torch.Tensor([[1,0],[0,0]]).double()


    h=(t)/(T)
    return (w1*(1-h) + h*w2).double()



def sine_weights_dot(t: float):
    
    w1 = torch.Tensor([[0,0],
                       [0,1]]).double()
    w2 = torch.Tensor([[1,0],
                       [0,0]]).double()
    return (-w1/(T) + w2/(T)).double()

    


if __name__ == "__main__":
    #test_diff()
    T = 10
    L = 10
    N = 250
    print(rho_grad_x(1,1))
    params = TimevaryingParams(sine_weights,sine_weights_dot)
    means = sine_means()
    f_d = GridGMM(means,params,L)
    solver = ContEQSolver(L,f_d,N)
    x= torch.tensor([30,10]).double()
    solver.solve_rho(0.1)
    #solver.solve_pytorch(t=0.1)
    xx,yy,uu,vv = solver.get_field(n=50)
    plt.quiver(xx,yy,uu,vv)
    plt.savefig("./images/fd_pde.jpg")
    solver.check_sol()





                