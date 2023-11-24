import functions_3d as f3d
import torch
import matplotlib.pyplot as plt
import numpy as np
import kornia
import nibabel as nib

dev=['cuda' if torch.cuda.is_available() else 'cpu'][0]


## Generating toy data
hei,wid,dep=64,64,64
img1 = f3d.generate_ball_3d(hei,wid,dep,0.4).to(dev)  # A 3D ball
img2=f3d.generate_ellipsoid_3d(hei,wid,dep,0.3,0.4,0.5).to(dev) # A 3D ellipsoid

### Blurring the images with Gaussian
ksizeimg,sigmaimg=11,1
kernelimg=kornia.filters.get_gaussian_kernel3d((ksizeimg,ksizeimg,ksizeimg),(sigmaimg,sigmaimg,sigmaimg),device=dev)
Kimg = lambda k : kornia.filters.filter3d(k[None,:,:,:,:],kernelimg)[0]
Kt = Kimg(torch.stack((img1,img2)))
img1 = Kt[0]
img2 = Kt[1]

### Saving the images as nifti files
img1nii=nib.Nifti1Image(img1.cpu().detach().numpy(),affine=np.eye(4))
nib.save(img1nii,'./data/img1.nii')
img2nii=nib.Nifti1Image(img2.cpu().detach().numpy(),affine=np.eye(4))
nib.save(img2nii,'./data/img2.nii')

### Define the parameters
m = torch.zeros(3,hei,wid,dep,requires_grad=True).to(dev)
idty= f3d.get_idty_3d(hei,wid,dep).to(dev)
phi,phiinv=f3d.get_idty_3d(hei,wid,dep).to(dev),f3d.get_idty_3d(hei,wid,dep).to(dev)


### Defining the Gaussian kernel for the vector field
ksize,sigma=25,5
kernel=kornia.filters.get_gaussian_kernel3d((ksize,ksize,ksize),(sigma,sigma,sigma),device=dev)
K = lambda k : kornia.filters.filter3d(k[None,:,:,:,:],kernel)[0]

errors=[]
mlist=[]
ls=[]

N=100
lamb = 1e-2
lr=1e-1
for i in range(N):
    print(i)
    m.detach_(), phi.detach_(), phiinv.detach_()
    m.requires_grad=True
    phi=idty+K(m)
    phiinv = idty-K(m)
    Ideformed = f3d.compose_function_3d(img1,idty-K(m))
    error=(1/lamb)*torch.sum((Ideformed-img2)**2)+ torch.sum(m*K(m))

    print(error.item(),(1/lamb)*torch.sum((Ideformed-img2)**2).item(), torch.sum(m*K(m)).item())
    
    mlist.append(torch.sum(m*K(m)).cpu().detach().numpy())
    ls.append((1/lamb)*torch.sum((Ideformed-img2)**2).cpu().detach().numpy())
    errors.append(error.cpu().detach().numpy())
    if i%10==0:
        plt.figure()
        plt.title('Shooting error')
        plt.plot(errors,label='energy')
        plt.plot(mlist,label='m')
        plt.plot(ls,label='ls1')
        plt.legend()
        plt.savefig('errorplot.jpg')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    error.backward()
    with torch.no_grad():
        m -= lr * m.grad
        m.grad.zero_()

slice=32
Ideformed = f3d.compose_function_3d(img1,phiinv)
fig,axs = plt.subplots(2,2)
fig.suptitle('Deformed original and Target')
im00=axs[0,0].imshow(img1[slice,:,:].cpu().detach().numpy())
axs[0,0].set_title('Original')
im01=axs[0,1].imshow(Ideformed.cpu().detach().numpy()[slice,:,:])
axs[0,1].set_title('Deformed original')
im10=axs[1,0].imshow(img2.cpu().detach().numpy()[slice,:,:])
axs[1,0].set_title('Target')
im11=axs[1,1].imshow(Ideformed.cpu().detach().numpy()[slice,:,:]-img2.cpu().detach().numpy()[slice,:,:])
axs[1,1].set_title('Difference')
plt.colorbar(im11,ax=axs[1,1])
plt.savefig('fig3d.png')
plt.show()

plt.figure()
plt.title('Elastic Registration')
plt.plot(errors,label='energy')
plt.plot(mlist,label='m')
plt.plot(ls,label='ls')
plt.legend()
plt.savefig('error3d.jpg')
plt.show()
