import get_param
params = get_param.params()

import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from derivatives import dx,dy,laplace,dx_p,dy_p
from setups import Dataset
from Logger import Logger,t_step
from pde_cnn import PDE_UNet,toCuda,toCpu,params

torch.manual_seed(0)
np.random.seed(0)

print(f"Parameters: {vars(params)}")

pde_cnn = toCuda(PDE_UNet())
optimizer = Adam(pde_cnn.parameters(),lr=params.lr)

mu = params.mu
rho = params.rho

logger = Logger(get_param.get_hyperparam(params),use_csv=False,use_tensorboard=params.log)
if params.load_latest or params.load_date_time is not None or params.load_index is not None:
	load_logger = Logger(get_param.get_hyperparam(params),use_csv=False,use_tensorboard=False)
	if params.load_optimizer:
		params.load_date_time, params.load_index = logger.load_state(pde_cnn,optimizer,params.load_date_time,params.load_index)
	else:
		params.load_date_time, params.load_index = logger.load_state(pde_cnn,None,params.load_date_time,params.load_index)
	params.load_index=int(params.load_index)
	print(f"loaded: {params.load_date_time}, {params.load_index}")
params.load_index = 0 if params.load_index is None else params.load_index

dataset = Dataset(params.width,params.height,params.batch_size)

def loss_function(x):
	if params.loss=="square":
		return torch.pow(x,2)
	if params.loss=="abs":
		return torch.abs(x)

for epoch in range(params.load_index,params.n_epochs):

	for i in range(params.n_batches_per_epoch):
		v_cond,cond_mask,flow_mask,v_old,p_old = toCuda(dataset.ask())
		if np.random.rand()<0.5:
			flip_diag = True
			v_cond,cond_mask,flow_mask,v_old,p_old = v_cond.permute(0,1,3,2).flip(1),cond_mask.permute(0,1,3,2),flow_mask.permute(0,1,3,2),v_old.permute(0,1,3,2).flip(1),p_old.permute(0,1,3,2)
		else:
			flip_diag = False
		
		if np.random.rand()<0.5:
			flip_lr = True
			v_cond,cond_mask,flow_mask,v_old,p_old = v_cond.flip(3),cond_mask.flip(3),flow_mask.flip(3),v_old.flip(3),p_old.flip(3)
			v_cond[:,1,:,:] = -v_cond[:,1,:,:]
			v_old[:,1,:,:] = -v_old[:,1,:,:]
		else:
			flip_lr = False
		
		if np.random.rand()<0.5:
			flip_ud = True
			v_cond,cond_mask,flow_mask,v_old,p_old = v_cond.flip(2),cond_mask.flip(2),flow_mask.flip(2),v_old.flip(2),p_old.flip(2)
			v_cond[:,0,:,:] = -v_cond[:,0,:,:]
			v_old[:,0,:,:] = -v_old[:,0,:,:]
		else:
			flip_ud = False
			
		v_new,p_new = pde_cnn(v_old,p_old,flow_mask,v_cond,cond_mask)
		
		loss_bound = torch.mean(cond_mask*loss_function(v_new-v_cond),dim=(1,2,3))
		
		v_new = cond_mask*v_cond+(1-cond_mask)*v_new
		v = v_new#(v_new+v_old)/2#
		
		loss_cont = torch.mean(loss_function(dx_p(v_new[:,1:2])+dy_p(v_new[:,0:1]))[:,:,1:-1,1:-1],dim=(1,2,3))
		loss_nav = torch.mean(flow_mask*loss_function(rho*((v_new[:,1:2]-v_old[:,1:2])+v[:,1:2]*dx(v[:,1:2]))+dx_p(p_new)-mu*laplace(v[:,1:2])),dim=(1,2,3))+\
						 torch.mean(flow_mask*loss_function(rho*((v_new[:,0:1]-v_old[:,0:1])+v[:,0:1]*dy(v[:,0:1]))+dy_p(p_new)-mu*laplace(v[:,0:1])),dim=(1,2,3))#double-check this loss
		loss = params.loss_bound*loss_bound + params.loss_cont*loss_cont + params.loss_nav*loss_nav
		loss = torch.mean(torch.log(loss))
		
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(pde_cnn.parameters(),1)
		optimizer.step()
	
		p_new = (p_new-torch.mean(p_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))#normalize pressure
		
		if flip_diag:
			v_new,p_new = v_new.permute(0,1,3,2).flip(1),p_new.permute(0,1,3,2)
		
		if flip_lr:
			v_new,p_new = v_new.flip(3),p_new.flip(3)
			v_new[:,1,:,:] = -v_new[:,1,:,:]
		
		if flip_ud:
			v_new,p_new = v_new.flip(2),p_new.flip(2)
			v_new[:,0,:,:] = -v_new[:,0,:,:]
		
		dataset.tell(toCpu(v_new),toCpu(p_new))
		
		loss = toCpu(loss).numpy()
		loss_bound = toCpu(torch.mean(loss_bound)).numpy()
		loss_cont = toCpu(torch.mean(loss_cont)).numpy()
		loss_nav = toCpu(torch.mean(loss_nav)).numpy()
		
		if i%10 == 0:
			logger.log(f"loss_{params.loss}",loss,epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_bound_{params.loss}",loss_bound,epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_cont_{params.loss}",loss_cont,epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_nav_{params.loss}",loss_nav,epoch*params.n_batches_per_epoch+i)
		
		if i%100 == 0:
			print(f"{epoch}: i:{i}: loss: {loss}; loss_bound: {loss_bound}; loss_cont: {loss_cont}; loss_nav: {loss_nav};")
		
	logger.save_state(pde_cnn,optimizer,epoch+1)
