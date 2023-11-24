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
img2nii=nib.load('./data/registration2norm.nii')
img2=torch.tensor(img2nii.get_fdata()).to(dev)
img3nii=nib.load('./data/registration4norm.nii')
img3=torch.tensor(img3nii.get_fdata()).to(dev)
img4nii=nib.load('./data/registration5norm.nii')
img4=torch.tensor(img4nii.get_fdata()).to(dev)
img3=torch.clone(img2)
img4=torch.clone(img2)
setup=copy.deepcopy(img2nii) # This line will be useful later when we will want to save the image (tensor format) in a nifti format.
hei,wid,dep=img1.shape

### Define the parameters
m = torch.zeros(3,hei,wid,dep,requires_grad=True).to(dev)
idty= f3d.get_idty_3d(hei,wid,dep).to(dev)
phi,phiinv=f3d.get_idty_3d(hei,wid,dep).to(dev),f3d.get_idty_3d(hei,wid,dep).to(dev)
phi1,phiinv1=f3d.get_idty_3d(hei,wid,dep).to(dev),f3d.get_idty_3d(hei,wid,dep).to(dev)
phi2,phiinv2=f3d.get_idty_3d(hei,wid,dep).to(dev),f3d.get_idty_3d(hei,wid,dep).to(dev)

### Define the Gaussian kernel for the vector field
ksize,sigma=25,1
kernel=kornia.filters.get_gaussian_kernel3d((ksize,ksize,ksize),(sigma,sigma,sigma),device=dev)
K = lambda k : kornia.filters.filter3d(k[None,:,:,:,:],kernel)[0]

errors=[]
ls1,ls2,ls3=[],[],[]
alphalist2,alphalist3=[],[]
mlist=[]


N=1000
timestep1,timestep2,timestep3=1,2,3
lamb = 1e-4
lr, lr1, lr2 =1e-1, 1e-6, 1e-5
alpha1,alpha2,alpha3=torch.tensor([1.],device=dev),torch.tensor([1.],requires_grad=True,device=dev),torch.tensor([1.],requires_grad=True,device=dev)
for i in range(N):
    print(i)
    m.detach_(), phi.detach_(), phiinv.detach_(), alpha2.detach_(),alpha3.detach_(),phiinv1.detach_(),phi1.detach_(),phi2.detach_(),phiinv2.detach_()
    m.requires_grad=True

    alpha2.requires_grad=True
    alpha3.requires_grad=True
    phiinv1,phiinv2,phiinv3=f3d.fwd_inf_multiple_3d(m,K,timestep1,timestep2,timestep3,alpha1,alpha2,alpha3)
    Ideformed1 = f3d.compose_function_3d(img1,phiinv1)
    Ideformed2 = f3d.compose_function_3d(img1,phiinv2)
    Ideformed3 = f3d.compose_function_3d(img1,phiinv3)
    error=torch.sum((Ideformed1-img2)**2)+torch.sum((Ideformed2-img3)**2) + torch.sum((Ideformed3-img4)**2)+lamb*torch.sum(m*K(m))*(alpha1**2+alpha2**2+alpha3**2)

    mlist.append(lamb*torch.sum(m*K(m)).cpu().detach().numpy())
    ls1.append(torch.sum((Ideformed1-img2)**2).cpu().detach().numpy())
    ls2.append(torch.sum((Ideformed2-img3)**2).cpu().detach().numpy())
    ls3.append(torch.sum((Ideformed3-img4)**2).cpu().detach().numpy())
    alphalist2.append(alpha2.cpu().detach().numpy())
    alphalist3.append(alpha3.cpu().detach().numpy())
    errors.append(error.cpu().detach().numpy())

    print(error.item(),torch.sum((Ideformed1-img2)**2).item(),torch.sum((Ideformed2-img3)**2).item(), torch.sum((Ideformed3-img4)**2).item(),((alpha1**2+alpha2**2+alpha3**2)*lamb*torch.sum(m*K(m))).item())
    print(alpha1.item(),alpha2.item(),alpha3.item())

    if i==N-1:
        torch.save(m.cpu(),'./data/mmultiple')
        torch.save(Ideformed1.cpu(),'./data/img1deformedmultiple')
        torch.save(Ideformed2.cpu(),'./data/img2deformedmultiple')
        torch.save(Ideformed3.cpu(),'./data/img3deformedmultiple')
        ideformednii1=nib.Nifti1Image(Ideformed1.cpu().detach().numpy(),setup.affine,setup.header)
        nib.save(ideformednii1,'./data/img1deformedmultiple.nii')
        ideformednii2=nib.Nifti1Image(Ideformed2.cpu().detach().numpy(),setup.affine,setup.header)
        nib.save(ideformednii2,'./data/img1deformedmultiple.nii')
        ideformednii3=nib.Nifti1Image(Ideformed3.cpu().detach().numpy(),setup.affine,setup.header)
        nib.save(ideformednii3,'./data/img1deformedmultiple.nii')
    if i%10==0:
        plt.figure()
        plt.title('Shooting error')
        plt.plot(errors,label='energy')
        plt.plot(mlist,label='m')
        plt.plot(ls1,label='ls1')
        plt.plot(ls2,label='ls2')
        plt.plot(ls3,label='ls3')
        plt.legend()
        plt.savefig('errorplot.jpg')
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        plt.figure()
        plt.title('Shooting alpha')
        plt.plot(alphalist2,label='alpha2')
        plt.plot(alphalist3,label='alpha3')
        plt.legend()
        plt.savefig('alphaplot.jpg')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    error.backward()
    with torch.no_grad():
        m -= lr * m.grad
        alpha2 -= lr1 * alpha2.grad
        alpha3 -= lr2 * alpha3.grad
        m.grad.zero_()
        alpha2.grad.zero_()
        alpha3.grad.zero_()


plt.figure()
plt.title('Shooting')
plt.plot(errors,label='energy')
plt.plot(mlist,label='m')
plt.plot(ls1,label='ls1')
plt.plot(ls2,label='ls2')
plt.plot(ls3,label='ls3')
plt.legend()
plt.savefig('errorplot.jpg')
plt.show()

plt.figure()
plt.title('Shooting alpha')
plt.plot(alphalist2,label='alpha2')
plt.plot(alphalist3,label='alpha3')
plt.legend()
plt.savefig('alphaplot.jpg')
plt.show()
