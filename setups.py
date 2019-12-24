import torch
import numpy as np

class setups:
	def __init__(self,h,w,batch_size):
		self.h, self.w = h,w
		self.batch_size = batch_size
		
		object_h = np.random.randint(-5,20,batch_size)+10
		object_w = np.random.randint(-5,20,batch_size)+10
		object_h[0],object_w[0] = 10,5
		
		x_disp = np.random.randint(-20,20,batch_size)
		y_disp = np.random.randint(-10,10,batch_size)
		x_disp[0],y_disp[0] = 0,0
		
		self.mask = torch.zeros([batch_size,1,h,w]).cuda()
		self.mask[:,:,0,:]=1
		self.mask[:,:,h-1,:]=1
		self.mask[:,:,:,0:5]=1
		self.mask[:,:,:,w-5:w]=1
		for i in range(batch_size):
			self.mask[i,:,(h//2-object_h[i]+y_disp[i]):(h//2+object_h[i]+y_disp[i]),(w//3-object_w[i]+x_disp[i]):(w//3+object_w[i]+x_disp[i])] = 1
		
		#v = np.ones(batch_size)+0.8*np.random.randn(batch_size)
		v = np.ones(batch_size)+0.9*np.random.rand(batch_size)
		v[0] = 1
		v = v*3#0.1#10 (on /2019-12-18 14:17:34/1.state) showed something that resembled a von karman vortex street
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
		
class setup:
	def __init__(self,h,w):
		self.h,self.w = h,w
		self.mask = torch.zeros([1,1,h,w]).cuda()
		self.mask[:,:,0,:]=1
		self.mask[:,:,h-1,:]=1
		self.mask[:,:,:,0:5]=1
		self.mask[:,:,:,w-5:w]=1
		
		v = 3*2*(np.random.rand()-0.5)
		#TODO: define outflow boundary conditions
		self.v_cond = torch.zeros([1,2,h,w]).cuda()
		self.v_cond[:,1,10:(h-10),0:5]=v
		self.v_cond[:,1,10:(h-10),w-5:w]=v
	
	def get(self):
		return self.mask,self.v_cond

def make_static_setup(self,h,w):
	object_h = np.random.randint(5,20) # object height / 2
	object_w = np.random.randint(5,20) # object width / 2
	object_x = np.random.randint(w//4-10,w//4+10)
	object_y = np.random.randint(h//2-10,h//2+10)
	
	inflow_v = 3*2*(np.random.rand()-0.5)
	
	cond_mask = torch.zeros([1,1,h,w]).cuda()
	cond_mask[:,:,0,:]=1
	cond_mask[:,:,h-1,:]=1
	cond_mask[:,:,:,0:5]=1
	
	cond_mask[:,:,(object_y-object_h):(object_y+object_h),(object_x-object_w):(object_x+object_w)] = 1
	
	flow_mask = 1-cond_mask
	flow_mask[:,:,:,w-5:w] = 0
	
	v_cond = torch.zeros([1,2,h,w]).cuda()
	v_cond[:,1,10:(h-10),0:5]=inflow_v
	
	return v_cond,cond_mask,flow_mask
