import get_param
import matplotlib.pyplot as plt
from Logger import Logger,t_step
from pde_cnn import PDE_UNet,toCuda,toCpu,params
import torch
import numpy as np
from setups import Dataset
from derivatives import dx,dy,laplace,dx_p,dy_p
from torch.optim import Adam
import cv2
import math
import numpy as np
import time

torch.manual_seed(0)
np.random.seed(1)

#date_time: 2020-01-08 12:15:30 index 70 gibt verhalten, das ähnlich wie karman street aussieht (bei v=-0.79/-1)

mu = params.mu
rho = params.rho
w,h = params.width,params.height
plot = True

logger = Logger(get_param.get_hyperparam(params),use_csv=False,use_tensorboard=False)
pde_cnn = PDE_UNet().cuda()
date_time,index = logger.load_state(pde_cnn,None,datetime=params.load_date_time,index=params.load_index)

print(f"date_time: {date_time}; index: {index}")

def vector2HSV(vector):
	"""
	:vector: vector field (size: 2 x height x width)
	:return: hsv (hue: direction of vector; saturation: 1; value: abs value of vector)
	"""
	values = torch.sqrt(torch.sum(torch.pow(vector,2),dim=0)).unsqueeze(0)
	saturation = torch.ones(values.shape).cuda()
	norm = vector/(values+0.000001)
	angles = torch.asin(norm[0])+math.pi/2
	angles[norm[1]<0] = 2*math.pi-angles[norm[1]<0]
	hue = angles.unsqueeze(0)/(2*math.pi)
	hue = (hue*360+100)%360
	values = torch.sqrt(values/torch.max(values))
	hsv = torch.cat([hue,saturation,values])
	return hsv.permute(1,2,0).cpu().numpy()


cv2.namedWindow('color_wheel',cv2.WINDOW_NORMAL)
vector = torch.cat([torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(2).repeat(1,1,200),torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(1).repeat(1,200,1)]).cuda()
image = vector2HSV(vector)
image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
cv2.imshow('color_wheel',image)

eps = 0.00000001

def loss_function(x):
	if params.loss=="square":
		return torch.pow(x,2)
	if params.loss=="exp_square":
		x = torch.pow(x,2)
		return torch.exp(x/torch.max(x).detach()*5)
	if params.loss=="abs":
		return torch.abs(x)
	if params.loss=="log_square":
		return torch.log(torch.pow(x,2)+eps)

flip_diag = False
flip_lr = False
flip_ud = False


with torch.no_grad():
	for epoch in range(20):
		dataset = Dataset(w,h,1,1)
		for t in range(5000):
			v_cond,cond_mask,flow_mask,v_old,p_old = toCuda(dataset.ask())
			if np.random.rand()<0:
				flip_diag = True
				v_cond,cond_mask,flow_mask,v_old,p_old = v_cond.permute(0,1,3,2).flip(1),cond_mask.permute(0,1,3,2),flow_mask.permute(0,1,3,2),v_old.permute(0,1,3,2).flip(1),p_old.permute(0,1,3,2)
			else:
				flip_diag = False
			
			if np.random.rand()<0:
				flip_lr = True
				v_cond,cond_mask,flow_mask,v_old,p_old = v_cond.flip(3),cond_mask.flip(3),flow_mask.flip(3),v_old.flip(3),p_old.flip(3)
				v_cond[:,1,:,:] *=-1
				v_old[:,1,:,:] *=-1
				p_old = torch.cat([p_old[:,:,:,-1:],p_old[:,:,:,:-1]],dim=3)
			else:
				flip_lr = False
			
			if np.random.rand()<0:
				flip_ud = True
				v_cond,cond_mask,flow_mask,v_old,p_old = v_cond.flip(2),cond_mask.flip(2),flow_mask.flip(2),v_old.flip(2),p_old.flip(2)
				v_cond[:,0,:,:] *=-1
				v_old[:,0,:,:] *=-1
				p_old = torch.cat([p_old[:,:,-1:],p_old[:,:,:-1]],dim=2)
			else:
				flip_ud = False
			"""p_old[:,:,:,-1] = 0
			p_old[:,:,:,0] = 0
			p_old[:,:,-1] = 0
			p_old[:,:,0] = 0"""
			
			v_new,p_new = pde_cnn(v_old,p_old,flow_mask,v_cond,cond_mask)
			
			#loss_bound = torch.mean(cond_mask*(v_new-v_cond)**2)
			v_new = cond_mask*v_cond+flow_mask*v_new
			"""
			v = v_new#(v_new+v_old)/2#
			loss_cont = torch.mean(flow_mask*(dx(v_new[:,1:2])+dy(v_new[:,0:1]))**2)
			loss_nav = (torch.mean(flow_mask*(rho*((v_new[:,1:2]-v_old[:,1:2])+v[:,1:2]*dx(v[:,1:2]))+dx(p_new)-mu*laplace(v[:,1:2]))**2)+\
								torch.mean(flow_mask*(rho*((v_new[:,0:1]-v_old[:,0:1])+v[:,0:1]*dy(v[:,0:1]))+dy(p_new)-mu*laplace(v[:,0:1]))**2))#I think, here is a mistake...
			loss = params.loss_bound*loss_bound + params.loss_cont*loss_cont + params.loss_nav*loss_nav
			"""
			p_new = (p_new-torch.mean(p_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
			if flip_ud:
				v_new,p_new = v_new.flip(2),p_new.flip(2)
				v_new[:,0,:,:] *= -1
				p_new = torch.cat([p_new[:,:,1:],p_new[:,:,:1]],dim=2)
			
			if flip_lr:
				v_new,p_new = v_new.flip(3),p_new.flip(3)
				v_new[:,1,:,:] *= -1
				p_new = torch.cat([p_new[:,:,:,1:],p_new[:,:,:,:1]],dim=3)
			
			if flip_diag:
				v_new,p_new = v_new.permute(0,1,3,2).flip(1),p_new.permute(0,1,3,2)
			
			"""p_new[:,:,:,-1] = 0
			p_new[:,:,:,0] = 0
			p_new[:,:,-1] = 0
			p_new[:,:,0] = 0"""
			
			if t%50==0:
				#loss,loss_bound,loss_cont,loss_nav = toCpu((loss,loss_bound,loss_cont,loss_nav))
				#print(f"t:{t}: loss: {loss.numpy()}; loss_bound: {loss_bound.numpy()}; loss_cont: {loss_cont.numpy()}; loss_nav: {loss_nav.numpy()};")
				print(f"t:{t}")
				#time.sleep(1)
				
				"""
				v_x,v_y = v_new[0,1],v_new[0,0]
				#print(f"min(v_x):{torch.min(v_x)}; max(v_x): {torch.max(v_x)}")
				#print(f"min(v_y):{torch.min(v_y)}; max(v_y): {torch.max(v_y)}")
				v_x,v_y = v_x-torch.min(v_x),v_y-torch.min(v_y)
				v_x,v_y = v_x/torch.max(v_x),v_y/torch.max(v_y)
				cv2.imshow('v_x',toCpu(v_x).numpy())
				cv2.imshow('v_y',toCpu(v_y).numpy())
				"""
				
				p = p_new[0,0]
				p = p-torch.min(p)
				p = p/torch.max(p)
				cv2.imshow('p',toCpu(p).numpy())
				
				vector = v_new[0]
				image = vector2HSV(vector)
				image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
				cv2.imshow('hsv',image)
				
				loss_cont = loss_function((dx_p(v_new[:,1:2])+dy_p(v_new[:,0:1]))[0,0,1:-1,1:-1])
				print(f"max(loss_cont):{torch.max(loss_cont)}; mean(loss_cont): {torch.mean(loss_cont)}")
				loss_cont = loss_cont-torch.min(loss_cont)
				loss_cont = loss_cont/(torch.max(loss_cont))
				cv2.imshow('loss_cont',toCpu(loss_cont).numpy())
				
				
				v_new = cond_mask*v_cond+(1-cond_mask)*v_new
				v = v_new#(v_new+v_old)/2#
				loss_nav = loss_function(flow_mask*(rho*((v_new[:,1:2]-v_old[:,1:2])+v[:,1:2]*dx(v[:,1:2]))+dx_p(p_new)-mu*laplace(v[:,1:2])))+\
						 loss_function(flow_mask*(rho*((v_new[:,0:1]-v_old[:,0:1])+v[:,0:1]*dy(v[:,0:1]))+dy_p(p_new)-mu*laplace(v[:,0:1])))#double-check this loss
				print(f"max(loss_nav):{torch.max(loss_nav)}; mean(loss_nav): {torch.mean(loss_nav)}")
				
				
				cv2.waitKey(1)
				"""
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
				"""
			
			dataset.tell(toCpu(v_new),toCpu(p_new))
			
			#cv2.namedWindow('v_x',cv2.WINDOW_NORMAL)
			#cv2.namedWindow('v_y',cv2.WINDOW_NORMAL)
			cv2.namedWindow('p',cv2.WINDOW_NORMAL)
			cv2.namedWindow('hsv',cv2.WINDOW_NORMAL)
			cv2.namedWindow('loss_cont',cv2.WINDOW_NORMAL)
			
