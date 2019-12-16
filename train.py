import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from derivatives import dx,dy,laplace
from setups import setups
import sys
sys.path.insert(0, '/home/bigboy/Nils/Pytorch/Projects/Library')
sys.path.insert(0, '../Library')
from Logger import Logger,t_step

torch.manual_seed(0)
np.random.seed(0)

logger = Logger("CNN_PDE_Solver",use_csv=False,use_tensorboard=False)

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

#Attention: x/y are swapped (x-dimension=1; y-dimension=0)

pde_cnn = PDE_CNN().cuda()
optimizer = Adam(pde_cnn.parameters(),lr=0.00005)#0.000002)

batch_size = 10
mu = 1
rho = 1
alpha=200*10
beta=15*30#0#150
w,h = 300,100

for i in range(100):
	setup = setups(h,w,batch_size)
	
	v_old = torch.zeros([batch_size,2,h,w]).cuda()
	p_old = torch.zeros([batch_size,1,h,w]).cuda()

	for t in range(10000):
		mask,v_cond = setup.get()
		
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

	logger.save_state(pde_cnn,optimizer,i+1)
	
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

	plt.figure(4)
	loss_cont = (1-mask)*(dx(v_new[:,1:2])+dy(v_new[:,0:1]))**2
	plt.imshow((p_new)[0,0].cpu().detach().numpy())
	plt.draw()
	plt.pause(0.001)

input("end program...")
