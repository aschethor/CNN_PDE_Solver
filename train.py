import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

class PDE_CNN(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(6,10,kernel_size=[3,3],padding=[1,1])
		self.conv2 = nn.Conv2d(10,10,kernel_size=[7,7],padding=[3,3])
		self.conv3 = nn.Conv2d(10,3,kernel_size=[3,3],padding=[1,1])
	
	def forward(self,v_old,p_old,mask,v_cond):
		x = torch.cat([v_old,p_old,mask,v_cond],dim=1)#use gating mechanism for mask
		x = torch.sigmoid(self.conv1(x))
		x = torch.sigmoid(self.conv2(x))
		x = self.conv3(x)
		v_new, p_new = x[:,0:2], x[:,2:3]
		return v_new,p_new

dx_kernel = torch.Tensor([-1,0,1]).unsqueeze(0).unsqueeze(1).unsqueeze(2).cuda()
def dx(v):
	return F.conv2d(v,dx_kernel,padding=(0,1))

dy_kernel = torch.Tensor([-1,0,1]).unsqueeze(0).unsqueeze(1).unsqueeze(3).cuda()
def dy(v):
	return F.conv2d(v,dy_kernel,padding=(1,0))

laplace_kernel = torch.Tensor([[0,0.5,0],[0.5,-2,0.5],[0,0.5,0]]).unsqueeze(0).unsqueeze(1).cuda()
def laplace(v):
	return F.conv2d(v,laplace_kernel,padding=(1,1))

"""
v = torch.Tensor([[0,0,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]]).unsqueeze(0).unsqueeze(0).cuda()

print(f"v:{v};")
print(f"dx(v):{dx(v)}")
print(f"dy(v):{dy(v)}")
print(f"laplace(v):{laplace(v)}")
"""

#Attention: x/y are swapped (x-dimension=1; y-dimension=0)

pde_cnn = PDE_CNN().cuda()
optimizer = Adam(pde_cnn.parameters(),lr=0.00005)#0.000002)

mu = 3
rho = 1
alpha=200*100
beta=15*3#0#150

for i in range(100):
	
	w,h = 300,100#TODO: generate batch of multiple instances
	x_disp = np.random.randint(-20,20)
	y_disp = np.random.randint(-10,10)
	mask = torch.zeros([1,1,h,w]).cuda()
	mask[:,:,0,:]=1
	mask[:,:,h-1,:]=1
	mask[:,:,:,0:5]=1
	mask[:,:,:,w-5:w]=1
	mask[:,:,(h//2-10+y_disp):(h//2+10+y_disp),(w//3-5+x_disp):(w//3+5+x_disp)] = 1

	v = 1+torch.randn(1)
	v_cond = torch.zeros([1,2,h,w]).cuda()
	v_cond[:,1,10:(h-10),0:5]=v
	v_cond[:,1,10:(h-10),w-5:w]=v

	v_old = torch.zeros([1,2,h,w]).cuda()
	p_old = torch.zeros([1,1,h,w]).cuda()

	for t in range(10000):
		v_new,p_new = pde_cnn(v_old,p_old,mask,v_cond)
		v_new = mask*v_cond+(1-mask)*v_new
		v = v_new#(v_new+v_old)/2#
		
		loss_bound = torch.mean(mask*(v_new-v_cond)**2)
		loss_cont = alpha*torch.mean((1-mask)*(dx(v_new[:,1:2])+dy(v_new[:,0:1]))**2)
		loss_nav = beta*(torch.mean((1-mask)*(rho*((v_new[:,1:2]-v_old[:,1:2])+v[:,1:2]*dx(v[:,1:2]))+dx(p_new)-mu*laplace(v[:,1:2]))**2)+\
						 torch.mean((1-mask)*(rho*((v_new[:,0:1]-v_old[:,0:1])+v[:,0:1]*dy(v[:,0:1]))+dy(p_new)-mu*laplace(v[:,0:1]))**2))#I think, here is a mistake...
		#loss_nav = beta*(torch.mean((1-mask)*(rho*((v_new[:,1:2]-v_old[:,1:2]))+dx(p_new)-mu*laplace(v[:,1:2]))**2)+\
		#				 torch.mean((1-mask)*(rho*((v_new[:,0:1]-v_old[:,0:1]))+dy(p_new)-mu*laplace(v[:,0:1]))**2))#I think, here is a mistake...
		loss = loss_bound + loss_cont + loss_nav
		
		if t>20:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		v_old,p_old = v_new.detach(),(p_new-torch.mean(p_new)).detach()
		
		if t%1000==0:
			print(f"{i}: t:{t}: loss: {loss.detach().cpu().numpy()}; loss_bound: {loss_bound.detach().cpu().numpy()}; loss_cont: {loss_cont.detach().cpu().numpy()}; loss_nav: {loss_nav.detach().cpu().numpy()};")

	plt.figure(1)
	plt.imshow((v_new)[0,1].cpu().detach().numpy())
	plt.draw()
	plt.pause(0.001)
	
	plt.figure(2)
	plt.imshow((v_new)[0,0].cpu().detach().numpy())
	plt.draw()
	plt.pause(0.001)
	
	plt.figure(3)
	loss_cont = (1-mask)*(dx(v_new[:,1:2])+dy(v_new[:,0:1]))**2
	plt.imshow((torch.log(loss_cont))[0,0].cpu().detach().numpy())
	plt.draw()
	plt.pause(0.001)

input("end program...")
