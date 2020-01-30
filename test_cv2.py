import get_param
import matplotlib.pyplot as plt
from Logger import Logger,t_step
from pde_cnn import PDE_UNet,toCuda,toCpu,params
import torch
import numpy as np
from setups import Dataset
from derivatives import dx,dy,laplace
from torch.optim import Adam
import cv2
import math

torch.manual_seed(0)
np.random.seed(1)

#date_time: 2020-01-08 12:15:30 index 70 gibt verhalten, das ähnlich wie karman street aussieht (bei v=-0.79/-1)

mu = params.mu
rho = params.rho
w,h = params.width,params.height
plot = True

logger = Logger(get_param.get_hyperparam(params),use_csv=False,use_tensorboard=False)
pde_cnn = PDE_UNet().cuda()
date_time,index = logger.load_state(pde_cnn,None)

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
	hsv = torch.cat([hue,saturation,values])
	hsv[0] = (hsv[0]*360+100)%360
	return hsv.permute(1,2,0).cpu().numpy()


cv2.namedWindow('color_wheel',cv2.WINDOW_NORMAL)
vector = torch.cat([torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(2).repeat(1,1,200),torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(1).repeat(1,200,1)]).cuda()
image = vector2HSV(vector)
image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
cv2.imshow('color_wheel',image)

with torch.no_grad():
	for epoch in range(20):
		dataset = Dataset(w,h,1,1)
		for t in range(4000):
			v_cond,cond_mask,flow_mask,v_old,p_old = toCuda(dataset.ask())
			
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
			dataset.tell(toCpu(v_new),toCpu(p_new))
			
			cv2.namedWindow('v_x',cv2.WINDOW_NORMAL)
			cv2.namedWindow('v_y',cv2.WINDOW_NORMAL)
			cv2.namedWindow('p',cv2.WINDOW_NORMAL)
			cv2.namedWindow('hsv',cv2.WINDOW_NORMAL)
			
			if t%5==0:
				#loss,loss_bound,loss_cont,loss_nav = toCpu((loss,loss_bound,loss_cont,loss_nav))
				#print(f"t:{t}: loss: {loss.numpy()}; loss_bound: {loss_bound.numpy()}; loss_cont: {loss_cont.numpy()}; loss_nav: {loss_nav.numpy()};")
				print(f"t:{t}")
				v_x,v_y,p = v_new[0,1],v_new[0,0],p_new[0,0]
				#print(f"min(v_x):{torch.min(v_x)}; max(v_x): {torch.max(v_x)}")
				#print(f"min(v_y):{torch.min(v_y)}; max(v_y): {torch.max(v_y)}")
				v_x,v_y,p = v_x-torch.min(v_x),v_y-torch.min(v_y),p-torch.min(p)
				v_x,v_y,p = v_x/torch.max(v_x),v_y/torch.max(v_y),p/torch.max(p)
				cv2.imshow('v_x',toCpu(v_x).numpy())	#cv2.imshow('v_x',toCpu(torch.cat([v_x.unsqueeze(2),v_y.unsqueeze(2),torch.zeros([h,w,1]).cuda()],dim=2)).numpy())
				cv2.imshow('v_y',toCpu(v_y).numpy())
				cv2.imshow('p',toCpu(p).numpy())
				vector = v_new[0]
				image = vector2HSV(vector)
				image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
				cv2.imshow('hsv',image)
				
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
