import functions_2d as f2d
import torch
import matplotlib.pyplot as plt
import numpy as np
import kornia
import torchvision

dev=['cuda' if torch.cuda.is_available() else 'cpu'][0]

#Loading images
img1=torchvision.io.read_image('./data/DS0002AxialSlice80.png')[0]
img2=torchvision.io.read_image('./data/DS0003AxialSlice80.png')[0]
hei,wid=img1.shape

### Define the variables
m = torch.zeros(2,hei,wid,requires_grad=True)
idty= f2d.get_idty_2d(hei,wid)
phi,phiinv=f2d.get_idty_2d(hei,wid),f2d.get_idty_2d(hei,wid)

### If we want to try with toy data, uncomment the following lines:
# hei,wid=64,64
# ellipse=f2d.generate_ellipse_2d(hei,wid,a=0.3,b=0.4)
# disk = f2d.generate_disk_2d(hei,wid,radius=0.4)
## Blurring with Gaussian
# ksizeimg,sigmaimg=11,1
# kernelimg=kornia.filters.get_gaussian_kernel2d((ksizeimg,ksizeimg),(sigmaimg,sigmaimg))
# Kimg = lambda k : kornia.filters.filter2d(k[None,:,:,:],kernelimg)[0]
# Ktimg = Kimg(torch.stack((ellipse,disk)))
# img1 = Ktimg[0]
# img2 = Ktimg[1]

### Defining the Gaussian kernel for the vector fields
ksize,sigma=25,5
kernel=kornia.filters.get_gaussian_kernel2d((ksize,ksize),(sigma,sigma))
K = lambda k : kornia.filters.filter2d(k[None,:,:,:],kernel)[0]


errors=[]
mlist=[]
ls=[]

N=100
timesteps=1
lamb = 1e-5
lr=1e-4
for i in range(N):
    print(i)
    m.detach_(), phi.detach_(), phiinv.detach_()
    m.requires_grad=True

    phiinv=f2d.fwd_2d(m,K,timesteps)
    Ideformed = f2d.compose_function_2d(img1,phiinv)
    error=torch.sum((Ideformed-img2)**2)+ lamb*torch.sum(m*K(m))

    print(error.item(),torch.sum((Ideformed-img2)**2).item(), lamb*torch.sum(m*K(m)).item())

    mlist.append(lamb*torch.sum(m*K(m)).detach().numpy())
    ls.append(torch.sum((Ideformed-img2)**2).detach().numpy())
    errors.append(error.detach().numpy())
    
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


Ideformed = f2d.compose_function_2d(img1,phiinv)
fig,axs = plt.subplots(3,4)
fig.suptitle('Deformed original and Target')
im00=axs[0,0].imshow(img1)
axs[0,0].set_title('Original')
im01=axs[0,1].imshow(Ideformed.detach().numpy())
axs[0,1].set_title('Deformed original')
im02=axs[0,2].imshow(img2)
axs[0,2].set_title('Target')
im03=axs[0,3].imshow(Ideformed.detach().numpy()-img2.numpy())
axs[0,3].set_title('Difference')
plt.colorbar(im03,ax=axs[0,3])
im10=f2d.plot_diffeo_2d(phi,axs[1,0])
axs[1,0].set_title('phi')
im11=f2d.plot_diffeo_2d(phiinv,axs[1,1])
axs[1,1].set_title('phiinv')
im12=axs[1,2].imshow(f2d.detjacob_2d(phiinv).detach().numpy())
axs[1,2].set_title('detjacob')
plt.colorbar(im12,ax=axs[1,2])
im13=axs[1,3].plot(errors)
axs[1,3].set_title('error')
im20=axs[2,0].imshow(torch.gradient(phiinv[0,:,:])[0].detach().numpy())
axs[2,0].set_title('gradxxphiinv')
plt.colorbar(im20,ax=axs[2,0])
im21=axs[2,1].imshow(torch.gradient(phiinv[0,:,:])[1].detach().numpy())
axs[2,1].set_title('gradxyphiinv')
plt.colorbar(im21,ax=axs[2,1])
im22=axs[2,2].imshow(torch.gradient(phiinv[1,:,:])[0].detach().numpy())
axs[2,2].set_title('gradyxphiinv')
plt.colorbar(im22,ax=axs[2,2])
im23=axs[2,3].imshow(torch.gradient(phiinv[1,:,:])[1].detach().numpy())
axs[2,3].set_title('gradyyphiinv')
plt.colorbar(im23,ax=axs[2,3])
plt.savefig('fig2.png')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.imshow(img1)
# plt.colorbar()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.imshow(Ideformed.detach().numpy())
# plt.colorbar()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.imshow(img2)
# plt.colorbar()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.imshow(Ideformed.detach().numpy()-img2.numpy())
# plt.colorbar()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.imshow(f2d.detjacob(phiinv).detach().numpy())
# plt.colorbar()
# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
f2d.plot_diffeo_2d(phi,ax)
ax.set_title('phi')
plt.savefig('phi2.jpg')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
f2d.plot_diffeo_2d(phiinv,ax)
ax.set_title('phiinv')
plt.savefig('phiinv2.jpg')
plt.show()


plt.figure()
plt.plot(errors,label='energy')
plt.plot(mlist,label='m')
plt.plot(ls,label='ls')
plt.legend()
plt.savefig('error2.jpg')
plt.show()
