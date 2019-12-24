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
from pde_cnn import PDE_CNN,PDE_UNet,PDE_UNet2

torch.manual_seed(0)
np.random.seed(0)

#Attention: x/y are swapped (x-dimension=1; y-dimension=0)

#pde_cnn = PDE_CNN().cuda()
pde_cnn = PDE_UNet().cuda()
pde_cnn = PDE_UNet2().cuda()
optimizer = Adam(pde_cnn.parameters(),lr=0.0001)#0.00002)#0.000001)#0.00005)#0.000002)

batch_size = 100
mu = 1
rho = 1
alpha=200*10*100
beta=15*30*2*10#0#150
w,h = 300,100
plot = False#True
load_latest = False#True#

logger = Logger(f"UNet2_PDE_Solver_mu_{mu}_rho_{rho}",use_csv=False,use_tensorboard=False)
if load_latest:
	date_time,index = logger.load_state(pde_cnn,None)#optimizer)
	print(f"date_time: {date_time}; index: {index}")

#rho = 10

dataset = []

for i in range(100):
	setup = setups(h,w,batch_size)
	
	v_old = torch.zeros([batch_size,2,h,w]).cuda()
	p_old = torch.zeros([batch_size,1,h,w]).cuda()

	for t in range(20000):
		mask,v_cond = setup.get()
		
		v_new,p_new = pde_cnn(v_old,p_old,mask,v_cond)
		
		loss_bound = torch.mean(mask*(v_new-v_cond)**2,dim=(1,2,3))
		
		v_new = mask*v_cond+(1-mask)*v_new
		v = v_new#(v_new+v_old)/2#
		
		loss_cont = alpha*torch.mean((1-mask)*(dx(v_new[:,1:2])+dy(v_new[:,0:1]))**2,dim=(1,2,3))
		loss_nav = beta*(torch.mean((1-mask)*(rho*((v_new[:,1:2]-v_old[:,1:2])+v[:,1:2]*dx(v[:,1:2]))+dx(p_new)-mu*laplace(v[:,1:2]))**2,dim=(1,2,3))+\
						 torch.mean((1-mask)*(rho*((v_new[:,0:1]-v_old[:,0:1])+v[:,0:1]*dy(v[:,0:1]))+dy(p_new)-mu*laplace(v[:,0:1]))**2,dim=(1,2,3)))#double-check this loss
		loss = loss_bound + loss_cont + loss_nav
		loss = torch.mean(torch.log(loss))
		
		if t%1 == 0:#only every 300/ 500th time step to decorrelate samples
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
		v_old,p_old = v_new.detach(),(p_new-torch.mean(p_new)).detach()
		
		if t%1000==0:
			print(f"{i}: t:{t}: loss: {loss.detach().cpu().numpy()}; loss_bound: {torch.mean(loss_bound).detach().cpu().numpy()}; loss_cont: {torch.mean(loss_cont).detach().cpu().numpy()}; loss_nav: {torch.mean(loss_nav).detach().cpu().numpy()};")

		if t%1000 == 0:
			if plot:
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
				plt.imshow((p_new)[0,0].cpu().detach().numpy())
				plt.draw()
				plt.pause(0.001)

	logger.save_state(pde_cnn,optimizer,i+1)
	
if plot:
	input("end program...")
