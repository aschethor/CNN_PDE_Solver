import get_param
import matplotlib.pyplot as plt
from Logger import Logger,t_step
from pde_cnn import PDE_UNet,toCuda,toCpu,params
import torch
import numpy as np
from setups import Dataset
from derivatives import dx,dy,laplace
from torch.optim import Adam

#torch.manual_seed(0)
#np.random.seed(0)

#date_time: 2020-01-08 12:15:30 index 70 gibt verhalten, das ähnlich wie karman street aussieht (bei v=-0.79/-1)

mu = params.mu
rho = params.rho
w,h = params.width,params.height
plot = True

logger = Logger(get_param.get_hyperparam(params),use_csv=False,use_tensorboard=False)
pde_cnn = PDE_UNet().cuda()
date_time,index = logger.load_state(pde_cnn,None,datetime=params.load_date_time,index=params.load_index)

print(f"date_time: {date_time}; index: {index}")

with torch.no_grad():
	for epoch in range(10):
		dataset = Dataset(w,h,1,1)
		for t in range(5000):
			v_cond,cond_mask,flow_mask,v_old,p_old = toCuda(dataset.ask())
			
			v_new,p_new = pde_cnn(v_old,p_old,flow_mask,v_cond,cond_mask)
			
			loss_bound = torch.mean(cond_mask*(v_new-v_cond)**2)
			v_new = cond_mask*v_cond+flow_mask*v_new
			v = v_new#(v_new+v_old)/2#
			loss_cont = torch.mean(flow_mask*(dx(v_new[:,1:2])+dy(v_new[:,0:1]))**2)
			loss_nav = (torch.mean(flow_mask*(rho*((v_new[:,1:2]-v_old[:,1:2])+v[:,1:2]*dx(v[:,1:2]))+dx(p_new)-mu*laplace(v[:,1:2]))**2)+\
								torch.mean(flow_mask*(rho*((v_new[:,0:1]-v_old[:,0:1])+v[:,0:1]*dy(v[:,0:1]))+dy(p_new)-mu*laplace(v[:,0:1]))**2))#I think, here is a mistake...
			loss = params.loss_bound*loss_bound + params.loss_cont*loss_cont + params.loss_nav*loss_nav
			
			p_new = (p_new-torch.mean(p_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
			dataset.tell(toCpu(v_new),toCpu(p_new))
			
			if t%100==0:
				loss,loss_bound,loss_cont,loss_nav = toCpu((loss,loss_bound,loss_cont,loss_nav))
				print(f"t:{t}: loss: {loss.numpy()}; loss_bound: {loss_bound.numpy()}; loss_cont: {loss_cont.numpy()}; loss_nav: {loss_nav.numpy()};")
				
				plt.figure(1)
				plt.clf()
				plt.imshow(toCpu(v_new[0,1]).numpy())
				plt.draw()
				plt.pause(0.001)
				
				plt.figure(2)
				plt.clf()
				plt.imshow(toCpu(v_new[0,0]).numpy())
				plt.draw()
				plt.pause(0.001)
				
				plt.figure(5)
				plt.clf()
				plt.imshow(toCpu(p_new[0,0]).numpy())
				plt.draw()
				plt.pause(0.001)
				
				plt.figure(3)
				plt.clf()
				loss_cont = flow_mask*(dx(v_new[:,1:2])+dy(v_new[:,0:1]))**2
				print(f"max(loss_cont): {torch.max(loss_cont)}; mean(loss_cont): {torch.mean(loss_cont)}")
				plt.imshow(toCpu(torch.log(loss_cont)[0,0]).numpy())
				plt.draw()
				plt.pause(0.001)
				
				plt.figure(4)
				plt.clf()
				loss_nav = flow_mask*(rho*((v_new[:,1:2]-v_old[:,1:2])+v[:,1:2]*dx(v[:,1:2]))+dx(p_new)-mu*laplace(v[:,1:2]))**2+\
						flow_mask*(rho*((v_new[:,0:1]-v_old[:,0:1])+v[:,0:1]*dy(v[:,0:1]))+dy(p_new)-mu*laplace(v[:,0:1]))**2
				print(f"max(loss_nav): {torch.max(loss_nav)}; mean(loss_nav): {torch.mean(loss_nav)}")
				plt.imshow(toCpu(torch.log(loss_nav)[0,0]).numpy())
				plt.draw()
				plt.pause(1)
