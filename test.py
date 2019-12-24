import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/bigboy/Nils/Pytorch/Projects/Library')
sys.path.insert(0, '../Library')
from Logger import Logger,t_step
from pde_cnn import PDE_CNN,PDE_UNet,PDE_UNet2
import torch
import numpy as np
from setups import setups
from derivatives import dx,dy,laplace
from torch.optim import Adam

torch.manual_seed(0)
np.random.seed(0)

batch_size = 1
mu = 1
rho = 1
alpha=200*10
beta=15*30#0#150
w,h = 300,100#1000,300#500,150#
plot = True

#logger = Logger(f"CNN_PDE_Solver_mu_{mu}_rho_{rho}",use_csv=False,use_tensorboard=False)
logger = Logger(f"UNet2_PDE_Solver_mu_{mu}_rho_{rho}",use_csv=False,use_tensorboard=False)
setup = setups(h,w,batch_size)

#pde_cnn = PDE_CNN().cuda()
pde_cnn = PDE_UNet2().cuda()
optimizer = Adam(pde_cnn.parameters(),lr=0.00002)#0.00005)#0.000002)
date_time,index = logger.load_state(pde_cnn,None)
rho = 1
print(f"date_time: {date_time}; index: {index}")

v_old = torch.zeros([batch_size,2,h,w]).cuda()
p_old = torch.zeros([batch_size,1,h,w]).cuda()

for t in range(50000):
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

	optimizer.zero_grad()
	loss.backward()
	#optimizer.step()#somehow, it doesn't work without the optimizer?! :/
	
	v_old,p_old = v_new.detach(),(p_new-torch.mean(p_new)).detach()
	
	if t%10==0:
		print(f"t:{t}: loss: {loss.detach().cpu().numpy()}; loss_bound: {loss_bound.detach().cpu().numpy()}; loss_cont: {loss_cont.detach().cpu().numpy()}; loss_nav: {loss_nav.detach().cpu().numpy()};")

		plt.figure(1)
		plt.imshow((v_new)[0,1].cpu().detach().numpy())
		plt.draw()
		plt.pause(0.001)
		
		plt.figure(2)
		plt.imshow((v_new)[0,0].cpu().detach().numpy())
		plt.draw()
		plt.pause(0.001)
		"""
		plt.figure(3)
		loss_cont = (1-mask)*(dx(v_new[:,1:2])+dy(v_new[:,0:1]))**2
		plt.imshow((torch.log(loss_cont))[0,0].cpu().detach().numpy())
		plt.draw()
		plt.pause(0.001)
		"""

		plt.figure(4)
		plt.imshow((p_new)[0,0].cpu().detach().numpy())
		plt.draw()
		plt.pause(1)
