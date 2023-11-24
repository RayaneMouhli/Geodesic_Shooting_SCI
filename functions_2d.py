import torch
import matplotlib.pyplot as plt

def generate_disk_2d(hei, wid, radius):
    """ 
    Generate a 2D disk using the equation x^2 + y^2 <= radius^2

    Input : 
        - wight, height : Dimension of the disk generated (type : int,int)
        - radius : Radius of the disk (type : float)
    Output :
        - 2D disk (type : torch.Tensor of shape (height, width))
    """
    x = torch.linspace(-1, 1, hei)
    y = torch.linspace(-1, 1, wid)
    X, Y = torch.meshgrid(x, y)
    distance = torch.sqrt(X**2 + Y**2)
    disk = (distance <= radius).float()
    return disk

def generate_ellipse_2d(hei, wid, a, b):
    """ 
    Generate a 3D ellipsoid using the equation (x/a)^2 + (y/b)^2 <= 1

    Input : 
        - height, width : Dimension of the ellipsoid generated (type : int,int)
        - a,b : parameters of the ellipsoid (type : float, float)
    Output :
        - 2D ellipsoid (type : torch.Tensor of shape (height, width))
    """
    x = torch.linspace(-1, 1, hei)
    y = torch.linspace(-1, 1, wid)
    X, Y = torch.meshgrid(x, y)
    ellipse = ((X / a) ** 2 + (Y / b) ** 2 <= 1).float()
    return ellipse

def get_idty_2d(size_h, size_w): 
    """
    Generate the identity diffeomorphism in 2D.
    
    Input : size_h, size_w : Dimension of the diffeomorphism generated (type : int,int)

    Output : Identity diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))
    """
    HH, WW = torch.meshgrid([torch.arange(size_h, dtype=torch.double), torch.arange(size_w, dtype=torch.double)])
    return torch.stack((HH, WW))

def imageGradient_2d(img):
    """
    Compute the gradient of a 2D image.

    Input : img : Image (type : torch.Tensor of shape (size_h, size_w))

    Output : Gradient of the image (type : torch.Tensor of shape (2, size_h, size_w))
    """
    [gx, gy] = torch.gradient(img)
    G = torch.vstack([gy.unsqueeze(0), gx.unsqueeze(0)])

    return G
    
def get_jacobian_matrix_2d(h):
    """
    Compute the Jacobian matrix of a 2D diffeomorphism h.

    Input : h : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))

    Output : Jacobian matrix of h (type : torch.Tensor of shape (2,2, size_h, size_w))
    """
    dim = h.shape[0]
    J = torch.zeros([dim] + list(h.shape)).to(h.device)

    J[0] = imageGradient_2d(h[0].squeeze())
    J[1] = imageGradient_2d(h[1].squeeze())

    return J


def detjacob_2d(g):
    """
    Compute the determinant of the Jacobian matrix of a 2D diffeomorphism g.

    Input : g : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))

    Output : Determinant of the Jacobian matrix of g (type : torch.Tensor of shape (size_h, size_w))
    """
    jacob_g=get_jacobian_matrix_2d(g)
    a11,a12,a21,a22=jacob_g[0,0,:,:],jacob_g[0,1,:,:],jacob_g[1,0,:,:],jacob_g[1,1,:,:]
    det = a11*a22-a12*a21
    return det



# def compose_function_2d(f, diffeo, mode='border'):  # f: N x m x n  diffeo: 2 x m x n
#     f = f.permute(f.dim()-2, f.dim()-1, *range(f.dim()-2))  # change the size of f to m x n x ...
    
#     size_h, size_w = f.shape[:2]
#     Ind_diffeo = torch.stack((torch.floor(diffeo[1]).long()%size_h, torch.floor(diffeo[0]).long()%size_w))
#     F = torch.zeros(size_h+1, size_w+1, *f.shape[2:], dtype=torch.double)
    
#     if mode=='border':
#         F[:size_h,:size_w] = f
#         F[-1, :size_w] = f[-1]
#         F[:size_h, -1] = f[:, -1]
#         F[-1, -1] = f[-1,-1]
#     elif mode =='periodic':
#         # extend the function values periodically (1234 1)
#         F[:size_h,:size_w] = f
#         F[-1, :size_w] = f[0]
#         F[:size_h, -1] = f[:, 0]
#         F[-1, -1] = f[0,0]

#     # use the bilinear interpolation method
#     F00 = F[Ind_diffeo[0], Ind_diffeo[1]].permute(*range(2, f.dim()), 0, 1)  # change the size to ...*m*n
#     F01 = F[Ind_diffeo[0], Ind_diffeo[1]+1].permute(*range(2, f.dim()), 0, 1)
#     F10 = F[Ind_diffeo[0]+1, Ind_diffeo[1]].permute(*range(2, f.dim()), 0, 1)
#     F11 = F[Ind_diffeo[0]+1, Ind_diffeo[1]+1].permute(*range(2, f.dim()), 0, 1)

#     C = diffeo[0] - Ind_diffeo[1].type(torch.DoubleTensor)
#     D = diffeo[1] - Ind_diffeo[0].type(torch.DoubleTensor)

#     F0 = F00 + (F01 - F00)*C
#     F1 = F10 + (F11 - F10)*C
#     return F0 + (F1 - F0)*D

def compose_function_2d(img, diffeo): 

    """
    Compose an image img with a diffeomorphism diffeo using the bilinear interpolation method. Can also work to compose a vector field with a diffeomorphism.

    Input :
        - img    : Image (type : torch.Tensor of shape (n,size_h, size_w) with n the number of images. Can also work with shape (size_h, size_w)) 
        - diffeo : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))
    
    Output : Image composed with the diffeomorphism(type : torch.Tensor of shape (n,size_h, size_w) or (size_h, size_w))
    """
    dev=img.device
    img = img.permute(img.dim() - 2, img.dim() - 1, *range(img.dim() - 2))  # change the size of img to (size_h, size_w, size_d,n)
    size_h, size_w = img.shape[:2]

    Ind_diffeo = torch.stack(((diffeo[0]).int(),
                              (diffeo[1]).int())).to(device=dev)
    Ind_diffeop1=Ind_diffeo+1

    deltax=diffeo[0]-Ind_diffeo[0].float()
    deltay=diffeo[1]-Ind_diffeo[1].float()

    ## Boundaries condition

    ## If deltax<0 or Ind_diffeo[0]<0:
    ##      deltax+=1
    ##      Ind_diffeo[0]=(Ind_diffeo[0]-1)%size_h
    ##      Ind_diffeop1[0]=(Ind_diffeop1[0]-1)%size_h
    ## elif Ind_diffeo[0]>=size_h :
    ##      Ind_diffeo[0] = Ind_diffeo[0]%size_h
    ##      Ind_diffeop1[0]=Ind_diffeop1[0]%size_h
    ## elif Ind_diffeop1[0]>=size_h :
    ##      Ind_diffeop1[0]=Ind_diffeop1[0]%size_h

    ax = torch.logical_or(deltax<0,Ind_diffeo[0]<0)
    bx= Ind_diffeo[0]>=size_h
    cx = Ind_diffeop1[0]>=size_h

    deltax=torch.where(ax,deltax+1,deltax)
    Ind_diffeo[0]=torch.where(ax,(Ind_diffeo[0]-1)%size_h,torch.where(bx,Ind_diffeo[0]%size_h,Ind_diffeo[0]))
    Ind_diffeop1[0]=torch.where(ax,(Ind_diffeop1[0]-1)%size_h,torch.where(bx,Ind_diffeop1[0]%size_h,torch.where(cx,Ind_diffeop1[0]%size_h,Ind_diffeop1[0])))

    ay = torch.logical_or(deltay<0,Ind_diffeo[1]<0)
    by = Ind_diffeo[1]>=size_w
    cy = Ind_diffeop1[1]>=size_w

    deltay=torch.where(ay,deltay+1,deltay)
    Ind_diffeo[1]=torch.where(ay,(Ind_diffeo[1]-1)%size_w,torch.where(by,Ind_diffeo[1]%size_w,Ind_diffeo[1]))
    Ind_diffeop1[1]=torch.where(ay,(Ind_diffeop1[1]-1)%size_w,torch.where(by,Ind_diffeop1[1]%size_w,torch.where(cy,Ind_diffeop1[1]%size_w,Ind_diffeop1[1])))

    ## Bilinear interpolation
    interp=torch.zeros(*img.shape[2:],size_h,size_w,device=dev)
    interp+=img[Ind_diffeo[0],Ind_diffeo[1]].permute(*range(2, img.dim()), 0, 1)*(1-deltax)*(1-deltay)
    interp+=img[Ind_diffeo[0],Ind_diffeop1[1]].permute(*range(2, img.dim()), 0, 1)*(1-deltax)*(deltay)
    interp+=img[Ind_diffeop1[0],Ind_diffeo[1]].permute(*range(2, img.dim()), 0, 1)*(deltax)*(1-deltay)
    interp+=img[Ind_diffeop1[0],Ind_diffeop1[1]].permute(*range(2, img.dim()), 0, 1)*(deltax)*(deltay)

    torch.cuda.empty_cache()

    return interp


def compose_diffeo_2d(diffeo1,diffeo2):
    """
    Compose two diffeomorphisms diffeo1 and diffeo2 by defining the displacement field u = diff - idty.

    Input : 
        - diffeo1 : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))
        - diffeo2 : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))

    Output : Composition of the diffeomorphisms (type : torch.Tensor of shape (2, size_h, size_w))
    """
    idty=get_idty_2d(diffeo1.shape[1],diffeo1.shape[2])
    udiffeo1t = diffeo1 - idty
    diffeo1t = compose_function_2d(udiffeo1t,diffeo2) + diffeo2
    return diffeo1t


def laplace_inverse_2d(u,alpha,gamma):
    '''
    Computes the laplacian inverse of a vector field u of size (2, size_h, size_w)
    
    Input : 
        - Vector field (type : torch.Tensor of shape (2, size_h, size_w))
        - alpha, gamma : Parameters of the laplacian inverse (type : float, float)

    Output: Vector field (type : torch.Tensor of shape (2, size_h, size_w))
    '''
    size_h, size_w = u.shape[-2:]
    idty = get_idty_2d(size_h, size_w)
    lap = alpha * (4. - 2. * (torch.cos(2. * torch.pi * idty[0] / size_w) +
                     torch.cos(2. * torch.pi * idty[1] / size_h))) + gamma
    lapinv = 1. / (lap**2)

    fx = torch.fft.fftn(u[0])
    fy = torch.fft.fftn(u[1])
    fx *= lapinv
    fy *= lapinv
    vx = torch.real(torch.fft.ifftn(fx))
    vy = torch.real(torch.fft.ifftn(fy))

    return torch.stack((vx, vy))#.to(device=torch.device('cuda'))


def coad_2d(m,g):
    """
    Compute the coadjoint Ad^*_g m = det(Dg)(Dg)^T (m o g).

    Input :
        - m : Momenta (type : torch.Tensor of shape (2, size_h, size_w))
        - g : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))

    Output : Coadjoint  (type : torch.Tensor of shape (2, size_h, size_w))
    """
    jacob_g=get_jacobian_matrix_2d(g)
    comp = compose_function_2d(m,g) 
    v1,v2 = jacob_g[0,0,:,:]*comp[0,:,:]+jacob_g[1,0,:,:]*comp[1,:,:],jacob_g[0,1,:,:]*comp[0,:,:]+jacob_g[1,1,:,:]*comp[1,:,:]
    V=torch.stack((v1,v2))
    det = jacob_g[0,0,:,:]*jacob_g[1,1,:,:]-jacob_g[0,1,:,:]*jacob_g[1,0,:,:]
    return det*V  

def coad_inf_2d(m,v):
    """
    Compute the infinitesimal coadjoint ad^*_v m = (Dv)Tm + Dmv + div(v)m.

    Input :
        - m : Momenta (type : torch.Tensor of shape (2, size_h, size_w))
        - v : Vector field (type : torch.Tensor of shape (2, size_h, size_w))
    
    Output : Infinitesimal coadjoint (type : torch.Tensor of shape (2, size_h, size_w))
    """
    jacobv = get_jacobian_matrix_2d(v)
    jacobm = get_jacobian_matrix_2d(m)
    divv = jacobv[0,0,:,:]+jacobv[1,1,:,:]
    DvTm0,DvTm1=jacobv[0,0,:,:]*m[0,:,:]+jacobv[0,1,:,:]*m[1,:,:],jacobv[1,0,:,:]*m[0,:,:]+jacobv[1,1,:,:]*m[1,:,:]
    Dmv0,Dmv1 = jacobm[0,0,:,:]*v[0,:,:]+jacobm[1,0,:,:]*v[1,:,:],jacobm[0,1,:,:]*v[0,:,:]+jacobm[1,1,:,:]*v[1,:,:]
    return torch.stack((DvTm0,DvTm1))  + torch.stack((Dmv0,Dmv1)) + divv*m



def fwd_2d(m,K,timesteps):  
    """
    Forward shooting of the vector momenta method using the coadjoint. The flow is computed using the Euler scheme.

    Input :
        - m : Momenta (type : torch.Tensor of shape (2, size_h, size_w))
        - K : Operator (kernel) (type : function : torch.Tensor of shape (2, size_h, size_w) -> torch.Tensor of shape (2, size_h, size_w))
        - phi : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))
        - phiinv : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))
        - timesteps : Number of iterations for the shooting (type : int)
    Output :
        - phi : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))
        - phiinv : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))
    """
    n,h,w=m.shape
    deltat=1/timesteps
    idty=get_idty_2d(h,w)
    phi,phiinv=get_idty_2d(h,w).to(m.device),get_idty_2d(h,w).to(m.device)

    for t in range(timesteps):
        v=deltat*K(m)
        phi= phi + compose_function_2d(v,phi)
        phiinv = compose_diffeo_2d(phiinv,idty-v)
        m =  coad_2d(m,phiinv)
    return(phiinv)

def fwd_inf_2d(m,K,timesteps): 
    """
    Forward shooting of the vector momenta method using the infinitesimal coadjoint. The flow is computed using the Euler scheme.
    Input :
        - m : Momenta (type : torch.Tensor of shape (2, size_h, size_w))
        - K : Operator (kernel) (type : function : torch.Tensor of shape (2, size_h, size_w) -> torch.Tensor of shape (2, size_h, size_w))
        - phi : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))
        - phiinv : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))
        - timesteps : Number of iterations for the shooting (type : int)
    Output :
        - phi : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))
        - phiinv : Diffeomorphism (type : torch.Tensor of shape (2, size_h, size_w))
    """
    n,h,w=m.shape
    deltat=1/timesteps
    idty=get_idty_2d(h,w).to(m.device)
    phi,phiinv=get_idty_2d(h,w).to(m.device),get_idty_2d(h,w).to(m.device)
    for t in range(timesteps):
        v=deltat*K(m)
        phi= phi + compose_function_2d(v,phi)
        phiinv = compose_diffeo_2d(phiinv,idty-v)
        m = m -  coad_inf_2d(m,v)*deltat
    return(phiinv)

def plot_diffeo_2d(diffeo,ax):
    diffeo=diffeo.detach().numpy()
    ax.plot(diffeo[0,:,:,],diffeo[1,:,:,],'k')
    ax.plot(diffeo[0,:,:].T,diffeo[1,:,:,].T,'k')


def splat_2d(img,diffeo):
    def f(img,interp):
        return torch.sum(torch.mul(img,interp))
    interp=compose_function_2d(img,diffeo)
    interpimg=f(img,interp)
    interpimg.backward()
    return img.grad - interp

