import torch
from torch import nn

class PDE_CNN(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(6,10,kernel_size=[3,3],padding=[1,1])
		self.conv2 = nn.Conv2d(10,10,kernel_size=[7,7],padding=[3,3])
		self.conv3 = nn.Conv2d(10,3,kernel_size=[3,3],padding=[1,1])
	
	def forward(self,v_old,p_old,mask,v_cond):
		x = torch.cat([v_old,p_old,mask,v_cond],dim=1)#CODO: use gating mechanism for mask
		x = torch.sigmoid(self.conv1(x))
		x = torch.sigmoid(self.conv2(x))
		x = self.conv3(x)
		v_new, p_new = x[:,0:2], x[:,2:3]
		return v_new,p_new

class PDE_UNet(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.hidden_size = 10
		self.conv1 = nn.Conv2d( 6, self.hidden_size,kernel_size=[3,3],padding=[1,1])
		self.conv2 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv3 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv4 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.deconv1 = nn.ConvTranspose2d( self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv2 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv3 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.conv5 = nn.Conv2d( 2*self.hidden_size,3,kernel_size=[3,3],padding=[1,1])
		
	def forward(self,v_old,p_old,mask,v_cond):
		x = torch.cat([v_old,p_old,mask,v_cond],dim=1)#CODO: use gating mechanism for mask
		x1 = torch.sigmoid(self.conv1(x))
		x2 = torch.sigmoid(self.conv2(x1))
		x3 = torch.sigmoid(self.conv3(x2))
		x = torch.sigmoid(self.conv4(x3))
		x = torch.sigmoid(self.deconv1(x, output_size = [x3.shape[2],x3.shape[3]]))
		x = torch.cat([x,x3],dim=1)
		x = torch.sigmoid(self.deconv2(x, output_size = [x2.shape[2],x2.shape[3]]))
		x = torch.cat([x,x2],dim=1)
		x = torch.sigmoid(self.deconv3(x, output_size = [x1.shape[2],x1.shape[3]]))
		x = torch.cat([x,x1],dim=1)
		x = self.conv5(x)
		v_new, p_new = x[:,0:2], x[:,2:3]
		return v_new,p_new
	

class PDE_UNet2(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.hidden_size = 10
		self.conv1 = nn.Conv2d( 6, self.hidden_size,kernel_size=[3,3],padding=[1,1])
		self.conv2 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv3 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv4 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv5 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv6 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.deconv1 = nn.ConvTranspose2d( self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv2 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv3 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv4 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv5 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.conv7 = nn.Conv2d( 2*self.hidden_size,3,kernel_size=[3,3],padding=[1,1])
		
	def forward(self,v_old,p_old,mask,v_cond):
		x = torch.cat([v_old,p_old,mask,v_cond],dim=1)#CODO: use gating mechanism for mask
		x1 = torch.sigmoid(self.conv1(x))
		x2 = torch.sigmoid(self.conv2(x1))
		#x3 = torch.sigmoid(self.conv3(x2))
		x4 = torch.sigmoid(self.conv4(x2))
		x5 = torch.sigmoid(self.conv5(x4))
		x = torch.sigmoid(self.conv6(x5))
		x = torch.sigmoid(self.deconv1(x, output_size = [x5.shape[2],x5.shape[3]]))
		x = torch.cat([x,x5],dim=1)
		x = torch.sigmoid(self.deconv2(x, output_size = [x4.shape[2],x4.shape[3]]))
		x = torch.cat([x,x4],dim=1)
		#x = torch.sigmoid(self.deconv3(x, output_size = [x3.shape[2],x3.shape[3]]))
		#x = torch.cat([x,x3],dim=1)
		x = torch.sigmoid(self.deconv4(x, output_size = [x2.shape[2],x2.shape[3]]))
		x = torch.cat([x,x2],dim=1)
		x = torch.sigmoid(self.deconv5(x, output_size = [x1.shape[2],x1.shape[3]]))
		x = torch.cat([x,x1],dim=1)
		x = self.conv7(x)
		v_new, p_new = x[:,0:2], x[:,2:3]
		return v_new,p_new
	
	
	
