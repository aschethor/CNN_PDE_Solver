import torch
import torch.nn.functional as F

dx_kernel = torch.Tensor([-1,0,1]).unsqueeze(0).unsqueeze(1).unsqueeze(2).cuda()
#dx_kernel = torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2).cuda()
#dx_kernel = torch.Tensor([-0.5,-0.5,0,0.5,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(2).cuda()
#dx_kernel = torch.Tensor([-0.75,0,0.75]).unsqueeze(0).unsqueeze(1).unsqueeze(2).cuda()
def dx(v):
	#return F.conv2d(v,dx_kernel,padding=(0,2))
	return F.conv2d(v,dx_kernel,padding=(0,1))

dy_kernel = torch.Tensor([-1,0,1]).unsqueeze(0).unsqueeze(1).unsqueeze(3).cuda()
#dy_kernel = torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3).cuda()
#dy_kernel = torch.Tensor([-0.5,-0.5,0,0.5,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(3).cuda()
#dy_kernel = torch.Tensor([-0.75,0,0.75]).unsqueeze(0).unsqueeze(1).unsqueeze(3).cuda()
def dy(v):
	#return F.conv2d(v,dy_kernel,padding=(2,0))
	return F.conv2d(v,dy_kernel,padding=(1,0))

# idea here: treat pressure field as being displaced by 0.5 pixels in x/y direction
dx_p_kernel = torch.Tensor([[-0.5,0.5,0],[-0.5,0.5,0],[0,0,0]]).unsqueeze(0).unsqueeze(1).cuda()
def dx_p(v):
	return F.conv2d(v,dx_p_kernel,padding=(1,1))

dy_p_kernel = torch.Tensor([[-0.5,-0.5,0],[0.5,0.5,0],[0,0,0]]).unsqueeze(0).unsqueeze(1).cuda()
def dy_p(v):
	return F.conv2d(v,dy_p_kernel,padding=(1,1))



laplace_kernel = torch.Tensor([[0,0.5,0],[0.5,-2,0.5],[0,0.5,0]]).unsqueeze(0).unsqueeze(1).cuda()
def laplace(v):
	return F.conv2d(v,laplace_kernel,padding=(1,1))
 
"""
v = torch.Tensor([[0,0,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]]).unsqueeze(0).unsqueeze(0).cuda()

print(f"v:{v};")
print(f"dx(v):{dx(v)}")
print(f"dy(v):{dy(v)}")
print(f"laplace(v):{laplace(v)}")
"""

