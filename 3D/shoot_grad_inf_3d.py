import functions_3d as f3d
import torch
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import kornia
import copy

dev=['cuda' if torch.cuda.is_available() else 'cpu'][0]

### Loading the images
img1nii=nib.load('./data/registration1norm.nii')
img1=torch.tensor(img1nii.get_fdata()).to(dev)
img2nii=nib.load('./data/registration6norm.nii')
img2=torch.tensor(img2nii.get_fdata()).to(dev)
setup=copy.deepcopy(img2nii)  # This line will be useful later when we will want to save the image (tensor format) in a nifti format.
hei,wid,dep=img1.shape

### If we want to try with toy data, uncomment the following lines:

# hei,wid,dep=64,64,64
# img1 = f3d.generate_ball_3d(hei,wid,dep,0.4).to(dev)
# img2=f3d.generate_ellipsoid_3d(hei,wid,dep,0.3,0.4,0.5).to(dev)
## Blurring with Gaussian
# ksizeimg,sigmaimg=11,1
# kernelimg=kornia.filters.get_gaussian_kernel3d((ksizeimg,ksizeimg,ksizeimg),(sigmaimg,sigmaimg,sigmaimg),device=dev)
# Kimg = lambda k : kornia.filters.filter3d(k[None,:,:,:,:],kernelimg)[0]
# Kt = Kimg(torch.stack((img1,img2)))
# img1 = Kt[0]
# img2 = Kt[1]
# img1nii=nib.Nifti1Image(img1.cpu().detach().numpy(),affine=np.eye(4))
# nib.save(img1nii,'./data/img1.nii')
# img2nii=nib.Nifti1Image(img2.cpu().detach().numpy(),affine=np.eye(4))
# nib.save(img2nii,'./data/img2.nii')

### Define the variables
m = torch.zeros(3,hei,wid,dep,requires_grad=True).to(dev)
idty= f3d.get_idty_3d(hei,wid,dep).to(dev)
phi,phiinv=f3d.get_idty_3d(hei,wid,dep).to(dev),f3d.get_idty_3d(hei,wid,dep).to(dev)


## Define the Gaussian kernel for the vector field
ksize,sigma=25,1
kernel=kornia.filters.get_gaussian_kernel3d((ksize,ksize,ksize),(sigma,sigma,sigma),device=dev)
K = lambda k : kornia.filters.filter3d(k[None,:,:,:,:],kernel)[0]

errors=[]
ls=[]
mlist=[]

N=800
timesteps=5
lamb = 1e-7
lr=1e-1
for i in range(N):
    print(i)
    m.detach_(), phi.detach_(), phiinv.detach_()
    m.requires_grad=True

    phiinv=f3d.fwd_inf_3d(m,K,timesteps)
    Ideformed = f3d.compose_function_3d(img1,phiinv)
    error=torch.sum((Ideformed-img2)**2) + lamb*torch.sum(m*K(m))

    mlist.append(lamb*torch.sum(m*K(m)).cpu().detach().numpy())
    ls.append(torch.sum((Ideformed-img2)**2).cpu().detach().numpy())
    errors.append(error.cpu().detach().numpy())
    print(error.item(),torch.sum((Ideformed-img2)**2).item(), lamb*torch.sum(m*K(m)).item())

    if i==N-1:
        torch.save(m.cpu(),'./data/m')
        torch.save(Ideformed.cpu(),'./data/imgdeformed')
        ideformednii=nib.Nifti1Image(Ideformed.cpu().detach().numpy(),setup.affine,setup.header)
        nib.save(ideformednii,'./data/imgdeformed.nii')
    
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


plt.figure()
plt.title('Shooting')
plt.plot(errors,label='energy')
plt.plot(mlist,label='m')
plt.plot(ls,label='ls')
plt.legend()
plt.savefig('error.jpg')
plt.show()

# slice=58
# Ideformed = f3d.compose_function_3d(img1,phiinv)
# fig,axs = plt.subplots(2,2)
# fig.suptitle('Deformed original and Target')
# im00=axs[0,0].imshow(img1[:,slice,:].cpu().detach().numpy())
# axs[0,0].set_title('Original')
# plt.colorbar(im00,ax=axs[0,0])
# im01=axs[0,1].imshow(Ideformed.cpu().detach().numpy()[:,slice,:])
# axs[0,1].set_title('Deformed original')
# plt.colorbar(im01,ax=axs[0,1])
# im10=axs[1,0].imshow(img2.cpu().detach().numpy()[:,slice,:])
# axs[1,0].set_title('Target')
# plt.colorbar(im10,ax=axs[1,0])
# im11=axs[1,1].imshow(Ideformed.cpu().detach().numpy()[:,slice,:]-img2.cpu().detach().numpy()[:,slice,:])
# axs[1,1].set_title('Difference')
# plt.colorbar(im11,ax=axs[1,1])
# plt.savefig('fig1to6.png')
# plt.show()
