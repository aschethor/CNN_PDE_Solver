import argparse

def str2bool(v):
	"""
	'type variable' for add_argument
	"""
	if v.lower() in ('yes','true','t','y','1'):
		return True
	elif v.lower() in ('no','false','f','n','0'):
		return False
	else:
		raise argparse.ArgumentTypeError('boolean value expected.')

def params():
	"""
	return parameters for training / testing / plotting of models
	:return: parameter-Namespace
	"""
	parser = argparse.ArgumentParser(description='train / test a pytorch model to predict frames')

	# Training parameters
	parser.add_argument('--net', default="UNet", type=str, help='network to train', choices=["UNet"])
	parser.add_argument('--n_epochs', default=100, type=int, help='number of epochs (after each epoch, the model gets saved)')
	parser.add_argument('--n_batches_per_epoch', default=5000, type=int, help='number of batches per epoch (default: 5000)')
	parser.add_argument('--batch_size', default=100, type=int, help='batch size (default: 100)')
	parser.add_argument('--n_time_steps', default=1, type=int, help='number of time steps to propagate gradients (default: 1)')
	parser.add_argument('--average_sequence_length', default=20000, type=int, help='average sequence length in dataset (default: 20000)')
	parser.add_argument('--dataset_size', default=1000, type=int, help='size of dataset (default: 1000)')
	parser.add_argument('--cuda', default=True, type=str2bool, help='use GPU')
	parser.add_argument('--loss_bound', default=1, type=float, help='loss factor for boundary conditions')
	parser.add_argument('--loss_cont', default=2000000, type=float, help='loss factor for continuity equation')
	parser.add_argument('--loss_nav', default=9000, type=float, help='loss factor for navier stokes equations')
	parser.add_argument('--lr', default=0.0001, type=float, help='learning rate of optimizer')
	parser.add_argument('--log', default=True, type=str2bool, help='log models / metrics during training (turn off for debugging)')
	parser.add_argument('--flip', default=False, type=str2bool, help='flip training samples randomly during training (default: False)')
	parser.add_argument('--integrator', default='explicit', type=str, help='integration scheme (explicit / implicit / imex) (default: explicit)',choices=['explicit','implicit','imex'])
	parser.add_argument('--loss', default='square', type=str, help='loss type to train network (default: square)',choices=['square','abs','log_square','exp_square'])
	parser.add_argument('--loss_multiplier', default=1, type=float, help='multiply loss / gradients (default: 1)')

	# Setup parameters
	parser.add_argument('--width', default=300, type=int, help='setup width')
	parser.add_argument('--height', default=100, type=int, help='setup height')
	
	# Fluid parameters
	parser.add_argument('--rho', default=1, type=float, help='fluid density rho')
	parser.add_argument('--mu', default=1, type=float, help='fluid viscosity mu')
	
	# Load parameters
	parser.add_argument('--load_date_time', default=None, type=str, help='date_time of run to load (default: None)')
	parser.add_argument('--load_index', default=None, type=int, help='index of run to load (default: None)')
	parser.add_argument('--load_optimizer', default=True, type=str2bool, help='load state of optimizer (default: True)')
	parser.add_argument('--load_latest', default=False, type=str2bool, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')
	
	# parse parameters
	params = parser.parse_args()
	
	return params

def get_hyperparam(params):
	return f"net {params.net}; mu {params.mu}; rho {params.rho};"
