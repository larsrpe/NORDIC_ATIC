import numpy as np
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
    Rho_grad: array of shape (N+1,N+1,2) where N+1 Rho_ij corresponds to the spatial gradient of rho evaluated at x = L/N*j and y = L/N*i 
    """
    
    def __init__(self,L: float, Rho: np.ndarray,Rho_dot: np.ndarray, Rho_grad: np.ndarray) -> None:
        self.L = L
        self.N = Rho.shape[0]-1
        self.h = L/self.N
        self.V1 = None
        self.V2 = None
        self.G1 = None
        self.G2 = None

        self.Rho = Rho
        self.Rho_dot = Rho_dot
        self.Rho_grad = Rho_grad
        self.xs = np.linspace(0,L,self.N+1)
        self.ys = np.linspace(0,L,self.N+1)


    def get_spars_AB(self):
        "calculates the system matrixes Ax=B for every grid point using central differences as gradient approximation."

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
        
        #Rho part of equation (rho*DIV(v))
        #calculate rho on every gridpoint except the border, since central differnce is not defined here.
        rho = self.Rho[1:self.N,1:self.N]
        
        # multiply with the DIV opertaor rho*DIV
        rhoDiag = scipy.sparse.csc_matrix(np.diag(rho.flatten()))
        rhoDIV = rhoDiag@DIV
        
        #Rho_grad part of equation (grad_x*v1 + grad_y*v2)
        
        # get border idx 
        border_grid = np.zeros((rows,cols))
        border_grid[:,0]=1
        border_grid[:,-1]=1
        border_grid[0,:]=1
        border_grid[-1,:]=1

        not_border_mask = border_grid.flatten()==0

        #get gradient and make diagonal matrix 
        grad_x = self.Rho_grad[:,:,0]
        grad_x_diag = np.diag(grad_x.flatten())
        grad_y = self.Rho_grad[:,:,1]
        grad_y_diag = np.diag(grad_y.flatten())

        

       
        #remove rows that are connected to divergence at the border since its not defined due to central differences
        grad_x_diag=grad_x_diag[not_border_mask,:]
        grad_y_diag=grad_y_diag[not_border_mask,:]

        #concatenate 
        grad = scipy.sparse.hstack([scipy.sparse.csc_matrix(grad_x_diag),scipy.sparse.csc_matrix(grad_y_diag)])
        
        #Define A and B in A@[v1.flatten();v2.flatten()]=B

        #Central differnece approx of continuety equation -Rho_dot.flatten() = A_cont@[v1.flatten();v2_flatten()]
        A_cont = grad + rhoDIV
        #add border constraints
        A = scipy.sparse.vstack([A_cont,Border])
        #create B
        B = np.zeros(A.shape[0])
        rho_dot = self.Rho_dot[1:self.N,1:self.N]
        B[:rho_dot.size] = -rho_dot.flatten()


        #cost matrix 
        #costdx =  scipy.sparse.block_diag((dx,dx))
        #costdy =  scipy.sparse.block_diag((dy,dy))
        #C_sqrt = scipy.sparse.vstack([costdx,costdy])
        #C = (C_sqrt.transpose())@C_sqrt 
        #scipy.sparse.linalg.inv(C)
        ##scipy.sparse.linalg.inv(A)
        #KKTmat1 = scipy.sparse.hstack([C,A.transpose()],format='csc')
        #zeros = scipy.sparse.csc_matrix(np.zeros((A.shape[0],A.shape[0])))
        #KKTmat2 = scipy.sparse.hstack([A,zeros],format='csc')
        #KKTmat = scipy.sparse.vstack([KKTmat1,KKTmat2],format='csc')
        #KKTvec = np.zeros(KKTmat.shape[0])
        #KKTvec[C.shape[1]:] = B
        #print(A.shape)
        return A,B,#KKTmat,KKTvec


    def solve(self,x0: np.ndarray=None):
        rows,cols = self.N+1,self.N+1
        A,B = self.get_spars_AB()

        if x0 is not None:
            v_vec = scipy.sparse.linalg.lsqr(A,B,x0=x0)[0]
        else: v_vec = scipy.sparse.linalg.lsqr(A,B)[0]

        self.G1 = (v_vec[:rows*cols]*self.Rho.flatten()).reshape(rows,cols)
        self.G2 = (v_vec[rows*cols:]*self.Rho.flatten()).reshape(rows,cols)
        
        self.V1 = v_vec[:rows*cols].reshape(rows,cols)
        self.V2 = v_vec[rows*cols:].reshape(rows,cols)
    
    def solve2(self):
        rows,cols = self.N+1,self.N+1
        _,_,KKTmat,KKTvec = self.get_spars_AB()

        x = scipy.sparse.linalg.spsolve(KKTmat,KKTvec)[0]
        v_vec =x[:rows*cols]
        self.G1 = (v_vec[:rows*cols]*self.Rho.flatten()).reshape(rows,cols)
        self.G2 = (v_vec[rows*cols:]*self.Rho.flatten()).reshape(rows,cols)
        
        self.V1 = v_vec[:rows*cols].reshape(rows,cols)
        self.V2 = v_vec[rows*cols:].reshape(rows,cols)


    def check_sol(self):
        for i in range(1,self.N):
            for j in range(1,self.N):
                x,y = self.xs[i],self.ys[j]
                v1 = self.V1[j,i]
                v2 = self.V2[j,i]
                rho = rho_f(x,y)
                rho_dot = dot(x,y)
                v1x = (self.V1[j,i+1]-self.V1[j,i-1])/(2*self.h)
                v2y = (self.V2[j+1,i]-self.V2[j-1,i])/(2*self.h)
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
                v1 = self.V1[j,i]
                v2 = self.V2[j,i]
                xx[k] = x
                yy[k] = y
                rho = self.Rho[j,i]
                uu[k] = rho*v1
                vv[k] = rho*v2
                k+=1
        
        return xx[:k],yy[:k],uu[:k],vv[:k]
    



if __name__ == "__main__":

    #example of use

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


        
    T = 10
    L = 10
    N = 50

    from densities import GridGMM
    import torch

    f_d = GridGMM.showtime('lena','roy_safari',(256,256),0,T,L,sigma_pixels=2)

    xs = np.linspace(0,L,N+1)
    ys = np.linspace(0,L,N+1)

    Rho = np.zeros((N+1,N+1))
    Rho_dot = np.zeros((N+1,N+1))
    Rho_grad = np.zeros((N+1,N+1,2))
    t = 1
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            r = torch.Tensor([x.item(),y.item()]).double()
            Rho[j,i] = f_d.eval(t,r).item()
            Rho_dot[j,i] = f_d.dot(t,r).item()
            Rho_grad[j,i,:] =f_d.grad(t,r).numpy()
    #Rho_dot = dot(xx,yy)
    #Rho_grad = np.concatenate((rho_grad_x(xx,yy).reshape(N+1,N+1,1),rho_grad_y(xx,yy).reshape(N+1,N+1,1)),axis=2)
    
    solver = ContEQSolver(L,Rho,Rho_dot,Rho_grad)
    solver.solve()
#
    xx,yy,uu,vv = solver.get_field(n=N)
    plt.quiver(xx,yy,uu,vv,scale=1)
    plt.savefig("./images/fd_pde_test.jpg")
    solver.check_sol()





                