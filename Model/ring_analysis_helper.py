import numpy as np
import torch
try:
    from .ring import get_input, get_time_input, get_fp
except ImportError:
    from ring import get_input, get_time_input, get_fp



# tuning for linear model
def get_tuning_curves(w,m):
	if 'torch' in str(type(w)):
		w = w.detach().cpu().numpy()
	if 'torch' in str(type(m)):
		m = m.detach().cpu().numpy()
	N = m.shape[0]
	directions = np.linspace(0,2*np.pi,8)
	contrast = 1
	tuning_curves = np.zeros((N,len(directions)))
	for i,d in enumerate(directions):
		inp = get_input(N, 'grating', {'direction':d,'contrast':contrast})
		fp_r = get_fp(w,m,inp)
		tuning_curves[:,i] = fp_r.flatten()
	return tuning_curves # N x n_directions

def get_tuning_linear(model):
	network_params = model.get_params()
	if 'npop' in network_params:
		npop = network_params['npop']
	else:
		npop = 1
	N = network_params['N']
	directions = np.linspace(0,2*np.pi,8)
	contrast = 1
	tuning_curves = np.zeros((npop*N,len(directions)))
	for i,d in enumerate(directions):
		grating_params = {
			'N':N,
			'npop':npop,
			'direction':d,
			'contrast':contrast
		}
		inp = get_input(N, 'grating', grating_params) # N
		fp_r = model.get_fp(inp)
		tuning_curves[:,i] = fp_r.flatten()
	return tuning_curves,directions*180/np.pi # N x n_directions

# tuning for non-linear model
def measure_tuning(model,T):
	network_params = model.get_params()
	if 'npop' in network_params:
		npop = network_params['npop']
	else:
		npop = 1
	N = network_params['N']
	directions = np.linspace(0,2*np.pi,8)
	contrast = 1
	tuning_curves = np.zeros((npop*N,len(directions)))
	for i,d in enumerate(directions):
		grating_params = {
			'N':N,
			'npop':npop,
			'direction':d,
			'contrast':1
		}
		inp = get_time_input(N, 'grating',T,'constant', grating_params) # T x N
		r_init = torch.zeros(npop*N)
		r_history = model.integrate(r_init, T,torch.tensor(inp,dtype=torch.float32)) # T x N
		tuning_curves[:,i] = r_history[-1,:] 
	return tuning_curves # N x n_directions
	

# under one visual direction, opto-stilulating multiple locations and get influence
def get_influence(model,T,locations,pop,grating_params,infl_type='absolute'):
	network_params = model.get_params()
	N = network_params['N']
	if 'npop' in network_params:
		npop = network_params['npop']
	else:
		npop = 1
	influences = np.zeros((len(locations),npop*N))
	
	for i,location in enumerate(locations):
		opto_params = {
		'N':N,
		'npop':npop,
		'pop':pop,
		'location':location,
		'opto_strength':1
		}
		opto = get_time_input(N, 'opto',T,'constant', opto_params)
		inp = get_time_input(N, 'grating',T,'constant', grating_params)
		r_init = torch.zeros(npop*N)
		r = model.integrate(r_init, T,torch.tensor(inp,dtype=torch.float32))
		r_opto = model.integrate(r_init, T,torch.tensor(inp,dtype=torch.float32),torch.tensor(opto,dtype=torch.float32)) # T x N
		influence = r_opto[0,-1,:] - r[0,-1,:] #(N)
		if infl_type == 'percentage':
			influence = influence/r[0,-1,:]
			# when r[-1,:] is 0, set influence to 0
			influence[np.isnan(influence)] = 0
			influence[np.isinf(influence)] = 0
		influences[i,:] = influence
	return influences # n_locations x N


def get_influence_spont_linear(model,T,locations,pop,inp_params):
	network_params = model.get_params()
	N = network_params['N']
	if 'npop' in network_params:
		npop = network_params['npop']
	else:
		npop = 1
	influences = np.zeros((len(locations),npop*N))

	for i,location in enumerate(locations):
		opto_params = {
		'N':N,
		'npop':npop,
		'pop':pop,
		'location':location,
		'opto_strength':1
		}
		opto = get_input(N, 'opto', opto_params)
		inp = get_input(N, 'spont',inp_params)
		r = model.get_fp(torch.tensor(inp,dtype=torch.float32))
		r_opto = model.integrate(torch.tensor(inp,dtype=torch.float32),torch.tensor(opto,dtype=torch.float32))
		influence = r_opto - r #(N)
		influences[i,:] = influence
	return influences # n_locations x N

def get_influence_spont(model,T,locations,pop,inp_params,percent=False):
	network_params = model.get_params()
	N = network_params['N']
	if 'npop' in network_params:
		npop = network_params['npop']
	else:
		npop = 1
	influences = np.zeros((len(locations),npop*N))

	for i,location in enumerate(locations):
		opto_params = {
		'N':N,
		'npop':npop,
		'pop':pop,
		'location':location,
		'opto_strength':1
		}
		opto = get_time_input(N, 'opto',T,'constant', opto_params)
		inp = get_time_input(N, 'spont',T,'constant', inp_params)
		r_init = torch.zeros(1,npop*N)
		r = model.integrate(r_init, T,torch.tensor(inp,dtype=torch.float32))
		r_opto = model.integrate(r_init, T,torch.tensor(inp,dtype=torch.float32),torch.tensor(opto,dtype=torch.float32))
		influence = r_opto[0,-1,:] - r[0,-1,:] #(N)
		if percent:
			influences[i,:] = influence/r[0,-1,:]
		else:
			influences[i,:] = influence
	return influences # n_locations x N

def influence_vs_distance_1loc(infl,location):
	"""
	infl: numpy array of shape (2N, ntrials)
	location: int, stimulated location (0 <= location < N)

	Returns:
		dict with keys 'toE' and 'toI', each mapping to
		1D arrays of averaged influence vs distance.
		The first entry corresponds to distance=0.
	"""
	if 'torch' in str(type(infl)):
		infl = infl.detach().cpu().numpy()
	N = infl.shape[0] // 2


	# Separate E and I
	infl_E = infl[:N]  # (N, ntrials)
	infl_I = infl[N:]  # (N, ntrials)

	# subtract input influence at stimulated location
	inp = np.zeros((N,))
	inp[location] = 1
	infl_E = infl_E - inp[:,None]

	# Average across trials
	mean_infl_E = np.mean(infl_E, axis=1)  # (N,)
	mean_infl_I = np.mean(infl_I, axis=1)  # (N,)

	# Compute distances
	neuron_positions = np.arange(N)
	distances = np.abs(neuron_positions - location)
	distances = np.minimum(distances, N - distances)  # periodic boundary

	max_dist = distances.max()

	# For each distance (0 to max_dist), average influences
	toE = np.zeros(max_dist )
	toI = np.zeros(max_dist )

	for d in range(max_dist ):
		mask = (distances == d)
		toE[d] = np.nanmean(mean_infl_E[mask])
		toI[d] = np.nanmean(mean_infl_I[mask])

	return {'toE': toE, 'toI': toI}



	
	