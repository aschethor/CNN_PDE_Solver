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

dataset = Dataset(params.width,params.height,params.batch_size,params.dataset_size,params.average_sequence_length)

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

for epoch in range(params.load_index,params.n_epochs):

	for i in range(params.n_batches_per_epoch):
		v_cond,cond_mask,flow_mask,v_old,p_old = toCuda(dataset.ask())
		
		total_loss = 0
		
		for t in range(params.n_time_steps):
			
			if params.flip:
				if np.random.rand()<0.5:
					flip_diag = True
					v_cond,cond_mask,flow_mask,v_old,p_old = v_cond.permute(0,1,3,2).flip(1),cond_mask.permute(0,1,3,2),flow_mask.permute(0,1,3,2),v_old.permute(0,1,3,2).flip(1),p_old.permute(0,1,3,2)
				else:
					flip_diag = False
				
				if np.random.rand()<0.5:
					flip_lr = True
					v_cond,cond_mask,flow_mask,v_old,p_old = v_cond.flip(3),cond_mask.flip(3),flow_mask.flip(3),v_old.flip(3),p_old.flip(3)
					v_cond[:,1,:,:] *=-1
					v_old[:,1,:,:] *=-1
					p_old = torch.cat([p_old[:,:,:,-1:],p_old[:,:,:,:-1]],dim=3)
				else:
					flip_lr = False
				
				if np.random.rand()<0.5:
					flip_ud = True
					v_cond,cond_mask,flow_mask,v_old,p_old = v_cond.flip(2),cond_mask.flip(2),flow_mask.flip(2),v_old.flip(2),p_old.flip(2)
					v_cond[:,0,:,:] *=-1
					v_old[:,0,:,:] *=-1
					p_old = torch.cat([p_old[:,:,-1:],p_old[:,:,:-1]],dim=2)
				else:
					flip_ud = False
			
			
			v_new,p_new = pde_cnn(v_old,p_old,flow_mask,v_cond,cond_mask)
			
			loss_bound = torch.mean(loss_function(cond_mask*(v_new-v_cond)),dim=(1,2,3))
			
			v_new = cond_mask*v_cond+(1-cond_mask)*v_new
			if params.integrator == "explicit":
				v = v_old
			if params.integrator == "implicit":
				v = v_new
			if params.integrator == "imex":
				v = (v_new+v_old)/2
			
			loss_cont = torch.mean(loss_function(dx_p(v_new[:,1:2])+dy_p(v_new[:,0:1]))[:,:,1:-1,1:-1],dim=(1,2,3))
			loss_nav = torch.mean(loss_function(flow_mask*(rho*((v_new[:,1:2]-v_old[:,1:2])+v[:,1:2]*dx(v[:,1:2]))+dx_p(p_new)-mu*laplace(v[:,1:2]))),dim=(1,2,3))+\
							torch.mean(loss_function(flow_mask*(rho*((v_new[:,0:1]-v_old[:,0:1])+v[:,0:1]*dy(v[:,0:1]))+dy_p(p_new)-mu*laplace(v[:,0:1]))),dim=(1,2,3))#double-check this loss
			loss = params.loss_bound*loss_bound + params.loss_cont*loss_cont + params.loss_nav*loss_nav
			if params.loss == "log_square" or params.loss == "exp_square":
				loss = torch.mean(loss)
			else:
				loss = torch.mean(torch.log(loss))
			
			total_loss = total_loss + loss
			
			p_new = (p_new-torch.mean(p_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))#normalize pressure
			
			if params.flip:
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
			
			v_old,p_old = v_new,p_new
			
		dataset.tell(toCpu(v_new),toCpu(p_new))
		
		optimizer.zero_grad()
		total_loss = total_loss*params.loss_multiplier
		total_loss.backward()
		if params.clip_grad_norm is not None:
			torch.nn.utils.clip_grad_norm_(pde_cnn.parameters(),params.clip_grad_norm)
		optimizer.step()
	
		loss = toCpu(total_loss).numpy()
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
