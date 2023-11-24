import torch
import matplotlib.pyplot as plt

def generate_ball_3d(height, width, depth,radius):
    """ 
    Generate a 3D ball using the equation x^2 + y^2 + z^2 <= radius^2

    Input : 
        - wight, height, depth : Dimension of the volume generated (type : int,int,int)
        - radius : Radius of the ball (type : float)
    Output :
        - Volume representing a 3D ball (type : torch.Tensor of shape (width, height, depth)))
       """
    x = torch.linspace(-1, 1, height)
    y = torch.linspace(-1, 1, width)
    z = torch.linspace(-1, 1, depth)
    X, Y, Z = torch.meshgrid(x, y, z)
    distance = torch.sqrt(X**2 + Y**2+Z**2)
    disk = (distance <= radius).float()
    return disk


def generate_ellipsoid_3d(height, width,depth, a,b,c):
    """ 
    Generate a 3D ellipsoid using the equation (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1

    Input : 
        - wight, height, depth : Dimension of the volume generated (type : int,int,int)
        - a,b,c : parameters of the ellipsoid (type : float, float, float)
    Output :
        - Volume representing a 3D ellipsoid (type : torch.Tensor of shape (width, height, depth)))
    """
    x = torch.linspace(-1, 1, height)
    y = torch.linspace(-1, 1, width)
    z = torch.linspace(-1, 1, depth)
    X, Y, Z = torch.meshgrid(x, y, z)
    ellipse = ((X / a) ** 2 + (Y / b) ** 2 + (Z/c)**2 <= 1).float()
    return ellipse

def laplace_inverse_3d(u,alpha,gamma):
    '''
    Computes the laplacian inverse of a vector field u of size 3 x size_h x size_w x size_d
    
    Input : 
        - Vector field (type : torch.Tensor of shape = [3, h, w, d])
        - alpha, gamma : Parameters of the laplacian inverse (type : float, float)

    Output: Vector field (type : torch.Tensor of shape = [3, h, w, d])
    '''

    ### Personal note : Theoretically, the equations below should be correct.
    #  However, the results are not really satisfying. I advice to use a Kernel Gaussian instead using an existing library.
    size_h, size_w, size_d = u.shape[-3:]
    idty = get_idty_3d(size_h, size_w, size_d)
    lap = alpha *(6. - 2. * (torch.cos(2. * torch.pi * idty[0] / size_h) +
                     torch.cos(2. * torch.pi * idty[1] / size_w) +
                     torch.cos(2. * torch.pi * idty[2] / size_d))) + gamma
    # lap[0, 0] = 1.
    lapinv = 1. / lap
    # lap[0, 0] = 0.
    # lapinv[0, 0] = 1.

    fx = torch.fft.fftn(u[0])
    fy = torch.fft.fftn(u[1])
    fz = torch.fft.fftn(u[2])
    fx *= lapinv
    fy *= lapinv
    fz *= lapinv
    vx = torch.real(torch.fft.ifftn(fx))
    vy = torch.real(torch.fft.ifftn(fy))
    vz = torch.real(torch.fft.ifftn(fz))

    return torch.stack((vx, vy, vz))



def get_idty_3d(size_h, size_w, size_d):
    """
    Generate the identity diffeomorphism in 3D.
    
    Input : size_h, size_w, size_d : Dimension of the diffeomorphism generated (type : int,int,int)

    Output : Identity diffeomorphism (type : torch.Tensor of shape (3, size_h, size_w, size_d))
    """
    HH, WW, DD = torch.meshgrid([torch.arange(size_h),
                                 torch.arange(size_w),
                                 torch.arange(size_d)])
    return torch.stack((HH, WW, DD)).float()



def imageGradient_3d(img):
    """
    Compute the gradient of a 3D image.

    Input : img : Image (type : torch.Tensor of shape (size_h, size_w, size_d))

    Output : Gradient of the image (type : torch.Tensor of shape (3, size_h, size_w, size_d))
    """
    [gx, gy, gz] = torch.gradient(img)
    G = torch.vstack([gz.unsqueeze(0), gy.unsqueeze(0), gx.unsqueeze(0)])

    return G
    
def get_jacobian_matrix_3d(h):
    """
    Compute the Jacobian matrix of a 3D diffeomorphism h.

    Input : h : Diffeomorphism (type : torch.Tensor of shape (3, size_h, size_w, size_d))

    Output : Jacobian matrix of h (type : torch.Tensor of shape (3, 3, size_h, size_w, size_d))
    """
    dim = h.shape[0]
    J = torch.zeros([dim] + list(h.shape)).to(h.device)

    J[0] = imageGradient_3d(h[0].squeeze())
    J[1] = imageGradient_3d(h[1].squeeze())
    J[2] = imageGradient_3d(h[2].squeeze())

    return J

def detjacob_3d(g):
    """
    Compute the determinant of the Jacobian matrix of a 3D diffeomorphism g.

    Input : g : Diffeomorphism (type : torch.Tensor of shape (3, size_h, size_w, size_d))

    Output : Determinant of the Jacobian matrix of g (type : torch.Tensor of shape (size_h, size_w, size_d))
    """
    jacob_g = get_jacobian_matrix_3d(g)
    
    # Extract components of the Jacobian matrix
    a11, a12, a13 = jacob_g[0, 0, :, :], jacob_g[0, 1, :, :], jacob_g[0, 2, :, :]
    a21, a22, a23 = jacob_g[1, 0, :, :], jacob_g[1, 1, :, :], jacob_g[1, 2, :, :]
    a31, a32, a33 = jacob_g[2, 0, :, :], jacob_g[2, 1, :, :], jacob_g[2, 2, :, :]
    
    # Calculate the determinant of the Jacobian matrix
    det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
    
    return det



def compose_function_3d(img, diffeo): 

    """
    Compose an image img with a diffeomorphism diffeo using the trilinear interpolation method. Can also work to compose a vector field with a diffeomorphism.

    This function has been inspired by Klas' Modin code : https://github.com/klasmodin/ddmatch/blob/master/ddmatch/core.py

    Input :
        - img    : Image (type : torch.Tensor of shape (n,size_h, size_w, size_d) with n the number of images. Can also work with shape (size_h, size_w, size_d) ) 
        - diffeo : Diffeomorphism (type : torch.Tensor of shape (3, size_h, size_w, size_d))
    
    Output : Image composed with the diffeomorphism(type : torch.Tensor of shape (n,size_h, size_w, size_d) or (size_h, size_w, size_d))
    """
    dev=img.device
    img = img.permute(img.dim() - 3, img.dim() - 2, img.dim() - 1, *range(img.dim() - 3))  # change the size of img to (size_h, size_w, size_d,n)
    size_h, size_w, size_d = img.shape[:3]

    Ind_diffeo = torch.stack(((diffeo[0]).int(),
                              (diffeo[1]).int(),
                              (diffeo[2]).int())).to(device=dev)
    Ind_diffeop1=Ind_diffeo+1

    deltax=diffeo[0]-Ind_diffeo[0].float()
    deltay=diffeo[1]-Ind_diffeo[1].float()
    deltaz=diffeo[2]-Ind_diffeo[2].float()

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

    az = torch.logical_or(deltaz<0,Ind_diffeo[2]<0)
    bz = Ind_diffeo[2]>=size_d
    cz = Ind_diffeop1[2]>=size_d

    deltaz=torch.where(az,deltaz+1,deltaz)
    Ind_diffeo[2]=torch.where(az,(Ind_diffeo[2]-1)%size_d,torch.where(bz,Ind_diffeo[2]%size_d,Ind_diffeo[2]))
    Ind_diffeop1[2]=torch.where(az,(Ind_diffeop1[2]-1)%size_d,torch.where(bz,Ind_diffeop1[2]%size_d,torch.where(cz,Ind_diffeop1[2]%size_d,Ind_diffeop1[2])))

    ## Trilinear interpolation
    interp=torch.zeros(*img.shape[3:],size_h,size_w,size_d,device=dev)
    interp+=img[Ind_diffeo[0],Ind_diffeo[1],Ind_diffeo[2]].permute(*range(3, img.dim()), 0, 1, 2)*(1-deltax)*(1-deltay)*(1-deltaz)
    interp+=img[Ind_diffeo[0],Ind_diffeo[1],Ind_diffeop1[2]].permute(*range(3, img.dim()), 0, 1, 2)*(1-deltax)*(1-deltay)*(deltaz)
    interp+=img[Ind_diffeo[0],Ind_diffeop1[1],Ind_diffeo[2]].permute(*range(3, img.dim()), 0, 1, 2)*(1-deltax)*(deltay)*(1-deltaz)
    interp+=img[Ind_diffeo[0],Ind_diffeop1[1],Ind_diffeop1[2]].permute(*range(3, img.dim()), 0, 1, 2)*(1-deltax)*(deltay)*(deltaz)
    interp+=img[Ind_diffeop1[0],Ind_diffeo[1],Ind_diffeo[2]].permute(*range(3, img.dim()), 0, 1, 2)*(deltax)*(1-deltay)*(1-deltaz)
    interp+=img[Ind_diffeop1[0],Ind_diffeo[1],Ind_diffeop1[2]].permute(*range(3, img.dim()), 0, 1, 2)*(deltax)*(1-deltay)*(deltaz)
    interp+=img[Ind_diffeop1[0],Ind_diffeop1[1],Ind_diffeo[2]].permute(*range(3, img.dim()), 0, 1, 2)*(deltax)*(deltay)*(1-deltaz)
    interp+=img[Ind_diffeop1[0],Ind_diffeop1[1],Ind_diffeop1[2]].permute(*range(3, img.dim()), 0, 1, 2)*(deltax)*(deltay)*(deltaz)

    torch.cuda.empty_cache()

    return interp

def compose_diffeo_3d(diffeo1,diffeo2):
    """
    Compose two diffeomorphisms diffeo1 and diffeo2 by defining the displacement field u = diff - idty.

    Input : 
        - diffeo1 : Diffeomorphism (type : torch.Tensor of shape (3, size_h, size_w, size_d))
        - diffeo2 : Diffeomorphism (type : torch.Tensor of shape (3, size_h, size_w, size_d))

    Output : Composition of the diffeomorphisms (type : torch.Tensor of shape (3, size_h, size_w, size_d))
    """

    ### Personal note : Composing two diffeos and composing a diffeo with an image is not really the same since we can 
    # get rid of the boundaries conditions for the images (or vector field) by padding with 0 for example.   
    # Here, to deal with the boundaries conditions, we work with the displacement field that should be small at the boundaries (if small deformations)
    
    idty=get_idty_3d(diffeo1.shape[1],diffeo1.shape[2],diffeo1.shape[3]).to(diffeo1.device)

    ## Define the displacement field
    udiffeo1t = diffeo1 - idty
    diffeo1t = compose_function_3d(udiffeo1t,diffeo2) + diffeo2
    return diffeo1t


def coad_3d(m,g):
    """
    Compute the coadjoint Ad^*_g m = det(Dg)(Dg)^T (m o g). We have the relation ad^*v_m = d(Ad^*_g m)/dt.

    Input :
        - m : Vector momenta (type : torch.Tensor of shape (3, size_h, size_w, size_d))
        - g : Diffeomorphism (type : torch.Tensor of shape (3, size_h, size_w, size_d))

    Output : Coadjoint  (type : torch.Tensor of shape (3, size_h, size_w, size_d))
    """
    jacob_g=get_jacobian_matrix_3d(g)
    comp = compose_function_3d(m,g) 

    ## Compute Dg^T(mog)
    v1 = jacob_g[0,0,:,:,:]*comp[0,:,:]+jacob_g[1,0,:,:]*comp[1,:,:] + jacob_g[2,0,:,:]*comp[2,:,:]
    v2 = jacob_g[0,1,:,:]*comp[0,:,:]+jacob_g[1,1,:,:]*comp[1,:,:]+ jacob_g[2,1,:,:]*comp[2,:,:]
    v3 = jacob_g[0,2,:,:]*comp[0,:,:]+jacob_g[1,2,:,:]*comp[1,:,:]+ jacob_g[2,2,:,:]*comp[2,:,:]
    V=torch.stack((v1,v2,v3))

    ## Compute det(Dg)
    a11, a12, a13 = jacob_g[0, 0, :, :], jacob_g[0, 1, :, :], jacob_g[0, 2, :, :]
    a21, a22, a23 = jacob_g[1, 0, :, :], jacob_g[1, 1, :, :], jacob_g[1, 2, :, :]
    a31, a32, a33 = jacob_g[2, 0, :, :], jacob_g[2, 1, :, :], jacob_g[2, 2, :, :]
    det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)

    return det*V  

def coad_inf_3d(m,v):
    """
    Compute the infinitesimal coadjoint ad^*_v m = (Dv)Tm + Dmv + div(v)m. We have the relation ad^*v_m = d(Ad^*_g m)/dt.

    Input :
        - m : Momenta (type : torch.Tensor of shape (3, size_h, size_w, size_d))
        - v : Vector field (type : torch.Tensor of shape (3, size_h, size_w, size_d))
    
    Output : Infinitesimal coadjoint (type : torch.Tensor of shape (3, size_h, size_w, size_d))
    """
    jacobv = get_jacobian_matrix_3d(v)
    jacobm = get_jacobian_matrix_3d(m)

    ## Compute div(v)
    divv = jacobv[0,0,:,:]+jacobv[1,1,:,:]+jacobv[2,2,:,:]
    divvm0,divvm1,divvm2=divv*m[0],divv*m[1],divv*m[2]
    divvm = torch.stack((divvm0,divvm1,divvm2))

    ## Compute (Dv)Tm
    DvTm0= jacobv[0,0,:,:]*m[0,:,:]+jacobv[1,0,:,:]*m[1,:,:]+jacobv[2,0,:,:]*m[2,:,:]
    DvTm1 = jacobv[0,1,:,:]*m[0,:,:]+jacobv[1,1,:,:]*m[1,:,:]+jacobv[2,1,:,:]*m[2,:,:]
    DvTm2 = jacobv[0,2,:,:]*m[0,:,:]+jacobv[1,2,:,:]*m[1,:,:]+jacobv[2,2,:,:]*m[2,:,:]

    ## Compute Dmv
    Dmv0 = jacobm[0,0,:,:]*v[0,:,:]+jacobm[0,1,:,:]*v[1,:,:] + jacobm[0,2,:,:]*v[2,:,:]
    Dmv1 = jacobm[1,0,:,:]*v[0,:,:]+jacobm[1,1,:,:]*v[1,:,:] + jacobm[1,2,:,:]*v[2,:,:]
    Dmv2 = jacobm[2,0,:,:]*v[0,:,:]+jacobm[2,1,:,:]*v[1,:,:] + jacobm[2,2,:,:]*v[2,:,:]
    return torch.stack((DvTm0,DvTm1,DvTm2))  + torch.stack((Dmv0,Dmv1,Dmv2)) + divvm



def fwd_3d(m,K,timesteps): 
    """
    Forward shooting of the vector momenta method using the coadjoint. The flow is computed using the Euler scheme.

    Input :
        - m : Momenta (type : torch.Tensor of shape (3, size_h, size_w, size_d))
        - K : Operator (kernel) (type : function : torch.Tensor of shape (3, size_h, size_w, size_d) -> torch.Tensor of shape (3, size_h, size_w, size_d))
        - timesteps : Number of iterations for the shooting (type : int)
    Output :
        - phiinv : Diffeomorphism (type : torch.Tensor of shape (3, size_h, size_w, size_d))
    """
    n,h,w,d=m.shape
    deltat=1/timesteps
    idty=get_idty_3d(h,w,d)
    phi,phiinv=get_idty_3d(h,w,d).to(m.device),get_idty_3d(h,w,d).to(m.device)
    for t in range(timesteps):
        v=deltat*K(m)
        phi= phi + compose_function_3d(v,phi)
        phiinv = compose_diffeo_3d(phiinv,idty-v)
        m =  coad_3d(m,phiinv)
    return(phiinv)

def fwd_inf_3d(m,K,timesteps):
    """
    Forward shooting of the vector momenta method using the infinitesimal coadjoint. The flow is computed using the Euler scheme.
    Input :
        - m : Momenta (type : torch.Tensor of shape (3, size_h, size_w, size_d))
        - K : Operator (kernel) (type : function : torch.Tensor of shape (3, size_h, size_w, size_d) -> torch.Tensor of shape (3, size_h, size_w, size_d))
        - timesteps : Number of iterations for the shooting (type : int)
    Output :
        - phiinv : Diffeomorphism (type : torch.Tensor of shape (3, size_h, size_w, size_d))
    """
    n,h,w,d=m.shape
    deltat=1/timesteps
    idty=get_idty_3d(h,w,d).to(m.device)
    phi,phiinv=get_idty_3d(h,w,d).to(m.device),get_idty_3d(h,w,d).to(m.device)
    for t in range(timesteps):
        v=K(m)
        phi= phi + compose_function_3d(v,phi)*deltat
        phiinv = compose_diffeo_3d(phiinv,idty-v*deltat)
        m = m -  coad_inf_3d(m,v)*deltat
    return(phiinv)

def fwd_inf_multiple_3d(m,K,timestep1,timestep2,timestep3,alpha1,alpha2,alpha3):
    """
    Forward shooting of the vector momenta method for 3 images using the infinitesimal coadjoint. The flow is computed using the Euler scheme.
    Input :
        - m : Momenta (type : torch.Tensor of shape (3, size_h, size_w, size_d))
        - K : Operator (kernel) (type : function : torch.Tensor of shape (3, size_h, size_w, size_d) -> torch.Tensor of shape (3, size_h, size_w, size_d))
        - timestepk : Number of iterations for the shooting between images k and k+1 (type : int)
        - alphak : Parameter controlling the duration of the shooting between images k and k+1 (type : float)
    Output :
        - phiinvk : Diffeomorphism representing the deformation between the image source and the image k (type : torch.Tensor of shape (3, size_h, size_w, size_d))
    """
    n,h,w,d=m.shape
    deltat1=1/timestep1
    deltat2=1/timestep2
    deltat3=1/timestep3
    idty=get_idty_3d(h,w,d).to(m.device)
    phi,phiinv=get_idty_3d(h,w,d).to(m.device),get_idty_3d(h,w,d).to(m.device)
    for t in range(timestep1): #Shooting from the source to img1
        v=K(m)*deltat1*alpha1
        phi1= phi + compose_function_3d(v,phi)
        phiinv1 = compose_diffeo_3d(phiinv,idty-v)
        m = m -  coad_inf_3d(m,v)
    for t in range(timestep1,timestep2): #Shooting from the img1 to img2
        v=K(m)*deltat2*alpha2
        phi2 = phi1 + compose_function_3d(v,phi1)
        phiinv2 = compose_diffeo_3d(phiinv1,idty-v)
        m = m -  coad_inf_3d(m,v)
    for t in range(timestep2,timestep3): #Shooting from the img2 to img3
        v=K(m)*deltat3*alpha3
        phi3 = phi2 + compose_function_3d(v,phi2)
        phiinv3 = compose_diffeo_3d(phiinv2,idty-v)
        m = m -  coad_inf_3d(m,v)
    return(phiinv1,phiinv2,phiinv3)
