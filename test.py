import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/bigboy/Nils/Pytorch/Projects/Library')
sys.path.insert(0, '../Library')
from Logger import Logger,t_step
from pde_cnn import PDE_CNN,PDE_UNet,PDE_UNet2,PDE_UNet3
import torch
import numpy as np
from setups import Dataset
from derivatives import dx,dy,laplace
from torch.optim import Adam

#torch.manual_seed(0)
#np.random.seed(0)

batch_size = 1
mu = 1#0.1#
rho = 1#10#
alpha=200*10*1000
beta=15*30*2*10#0#150
w,h = 300,100#600,200#1000,300#500,150#
plot = True

#logger = Logger(f"CNN_PDE_Solver_mu_{mu}_rho_{rho}",use_csv=False,use_tensorboard=False)
#pde_cnn = PDE_CNN().cuda()
#pde_cnn = PDE_UNet2().cuda()
logger = Logger(f"UNet3_PDE_Solver_mu_{mu}_rho_{rho}",use_csv=False,use_tensorboard=False)
pde_cnn = PDE_UNet3().cuda()
date_time,index = logger.load_state(pde_cnn,None)

print(f"date_time: {date_time}; index: {index}")

with torch.no_grad():
	for epoch in range(10):
		dataset = Dataset(w,h,1,1)
		for t in range(5000):
			v_cond,cond_mask,flow_mask,v_old,p_old = dataset.ask()
			v_cond,cond_mask,flow_mask,v_old,p_old = v_cond.cuda(),cond_mask.cuda(),flow_mask.cuda(),v_old.cuda(),p_old.cuda()
			
			v_new,p_new = pde_cnn(v_old,p_old,flow_mask,v_cond,cond_mask)
			
			loss_bound = torch.mean(cond_mask*(v_new-v_cond)**2)
			v_new = cond_mask*v_cond+flow_mask*v_new
			v = v_new#(v_new+v_old)/2#
			loss_cont = alpha*torch.mean(flow_mask*(dx(v_new[:,1:2])+dy(v_new[:,0:1]))**2)
			loss_nav = beta*(torch.mean(flow_mask*(rho*((v_new[:,1:2]-v_old[:,1:2])+v[:,1:2]*dx(v[:,1:2]))+dx(p_new)-mu*laplace(v[:,1:2]))**2)+\
								torch.mean(flow_mask*(rho*((v_new[:,0:1]-v_old[:,0:1])+v[:,0:1]*dy(v[:,0:1]))+dy(p_new)-mu*laplace(v[:,0:1]))**2))#I think, here is a mistake...
			loss = loss_bound + loss_cont + loss_nav

			p_new = (p_new-torch.mean(p_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
			dataset.tell(v_new.detach().cpu(),p_new.detach().cpu())
			
			if t%1000==0:
				print(f"t:{t}: loss: {loss.detach().cpu().numpy()}; loss_bound: {loss_bound.detach().cpu().numpy()}; loss_cont: {loss_cont.detach().cpu().numpy()}; loss_nav: {loss_nav.detach().cpu().numpy()};")

				plt.figure(1)
				plt.clf()
				plt.imshow((v_new)[0,1].cpu().detach().numpy())
				plt.draw()
				plt.pause(0.001)
				
				plt.figure(2)
				plt.clf()
				plt.imshow((v_new)[0,0].cpu().detach().numpy())
				plt.draw()
				plt.pause(0.001)
				
				plt.figure(5)
				plt.clf()
				plt.imshow((p_new)[0,0].cpu().detach().numpy())
				plt.draw()
				plt.pause(0.001)
				
				plt.figure(3)
				plt.clf()
				loss_cont = flow_mask*(dx(v_new[:,1:2])+dy(v_new[:,0:1]))**2
				print(f"max(loss_cont): {torch.max(loss_cont)}; mean(loss_cont): {torch.mean(loss_cont)}")
				plt.imshow((torch.log(loss_cont))[0,0].cpu().detach().numpy())
				plt.draw()
				plt.pause(0.001)
				

				plt.figure(4)
				plt.clf()
				loss_nav = flow_mask*(rho*((v_new[:,1:2]-v_old[:,1:2])+v[:,1:2]*dx(v[:,1:2]))+dx(p_new)-mu*laplace(v[:,1:2]))**2+\
						flow_mask*(rho*((v_new[:,0:1]-v_old[:,0:1])+v[:,0:1]*dy(v[:,0:1]))+dy(p_new)-mu*laplace(v[:,0:1]))**2
				print(f"max(loss_nav): {torch.max(loss_nav)}; mean(loss_nav): {torch.mean(loss_nav)}")
				plt.imshow((torch.log(loss_nav))[0,0].cpu().detach().numpy())
				plt.draw()
				plt.pause(1)
