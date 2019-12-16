import torch
import numpy as np

class setups:
	def __init__(self,h,w,batch_size):
		self.h, self.w = h,w
		self.batch_size = batch_size
		
		
		x_disp = np.random.randint(-20,20,batch_size)
		y_disp = np.random.randint(-10,10,batch_size)
		
		self.mask = torch.zeros([batch_size,1,h,w]).cuda()
		self.mask[:,:,0,:]=1
		self.mask[:,:,h-1,:]=1
		self.mask[:,:,:,0:5]=1
		self.mask[:,:,:,w-5:w]=1
		for i in range(batch_size):
			self.mask[i,:,(h//2-10+y_disp[i]):(h//2+10+y_disp[i]),(w//3-5+x_disp[i]):(w//3+5+x_disp[i])] = 1
		
		v = 1*np.ones(batch_size)+0.5*np.random.randn(batch_size)
		v[0] = 1
		v = v*0.1
		self.v_cond = torch.zeros([batch_size,2,h,w]).cuda()
		for i in range(batch_size):
			self.v_cond[i,1,10:(h-10),0:5]=v[i]
			self.v_cond[i,1,10:(h-10),w-5:w]=v[i]
		
	
	def get(self):
		"""
		return a batch of setups
		(could change over time)
		"""
		return self.mask,self.v_cond
		
		
