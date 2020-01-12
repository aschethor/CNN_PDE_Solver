import torch
from torch import nn
from torch.nn import Parameter
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from derivatives import dx,dy,laplace
from setups import setups
from Logger import Logger,t_step

torch.manual_seed(0)
np.random.seed(0)

#Attention: x/y are swapped (x-dimension=1; y-dimension=0)


batch_size = 1
mu = 1
rho = 1
alpha=200*10
beta=15*30000#0#150
gamma = 1
w,h = 300,100
plot = True

setup = setups(h,w,batch_size)

v_old = torch.zeros([batch_size,2,h,w]).cuda()
p_old = torch.zeros([batch_size,1,h,w]).cuda()

for t in range(1000):
	mask,v_cond = setup.get()
	
	v_new,p_new = Parameter(torch.zeros([batch_size,2,h,w]).cuda()),Parameter(torch.zeros([batch_size,1,h,w]).cuda())
	
	optimizer = Adam([v_new,p_new],lr=0.1)#0.000002)
	
	for i in range(300):
		v_new2 = mask*v_cond+(1-mask)*v_new
		
		v = v_new2#(v_new2+v_old)/2#
		loss_bound = gamma*torch.mean(mask*(v_new2-v_cond)**2)
		loss_cont = alpha*torch.mean((1-mask)*(dx(v_new2[:,1:2])+dy(v_new2[:,0:1]))**2)#
		loss_nav = beta*(torch.mean((1-mask)*(rho*((v_new2[:,1:2]-v_old[:,1:2])+v[:,1:2]*dx(v[:,1:2]))+dx(p_new)-mu*laplace(v[:,1:2]))**2)+\
							torch.mean((1-mask)*(rho*((v_new2[:,0:1]-v_old[:,0:1])+v[:,0:1]*dy(v[:,0:1]))+dy(p_new)-mu*laplace(v[:,0:1]))**2))#I think, here is a mistake...
		loss = loss_bound + loss_cont + loss_nav
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	v_old,p_old = v_new2.data.detach(),(p_new-torch.mean(p_new)).data.detach()
	
	if t%10==0:
		print(f"t:{t}: loss: {loss.detach().cpu().numpy()}; loss_bound: {loss_bound.detach().cpu().numpy()}; loss_cont: {loss_cont.detach().cpu().numpy()}; loss_nav: {loss_nav.detach().cpu().numpy()};")

		if plot:
			plt.figure(1)
			plt.imshow((v_new2)[0,1].cpu().detach().numpy())
			plt.title(f"{t}")
			plt.draw()
			plt.pause(0.001)
			
			plt.figure(2)
			plt.imshow((v_new2)[0,0].cpu().detach().numpy())
			plt.draw()
			plt.pause(0.001)
			
			plt.figure(3)
			loss_cont = (dx(v_new2[:,1:2])+dy(v_new2[:,0:1]))**2
			plt.imshow((torch.log(loss_cont))[0,0].cpu().detach().numpy())
			plt.draw()
			plt.pause(0.001)

			plt.figure(4)
			loss_cont = (dx(v_new2[:,1:2])+dy(v_new2[:,0:1]))**2
			plt.imshow((p_new)[0,0].cpu().detach().numpy())
			plt.draw()
			plt.pause(0.001)

if plot:
	input("end program...")
