import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os, sys
from random import seed, gauss
from pandas import Series
import torch
try:
	from .ring import generate_model,get_input,get_time_input
	from .model_params import default_params
except ImportError:
	from ring import generate_model, get_input, get_time_input
	from model_params import default_params

def sim_grating_trials(model,nstims,ntrials,contrast=1,T=2000):
	network_params = model.get_params()
	if 'npop' in network_params:
		npop = network_params['npop']
	else:
		npop = 1
	N = network_params['N']
	directions = np.linspace(0,2*np.pi,nstims)
	resps = np.zeros((nstims,ntrials,npop*N))
	for i,d in enumerate(directions):
		grating_params = {
			'N':N,
			'npop':npop,
			'direction':d,
			'contrast':contrast,
			'ntrials':ntrials,
			'noise_level':5
		}
		inp = get_time_input(N, 'noisy_grating',T,'constant', grating_params) # ntrials x T x N
		# make input into shape ntrials x T x N
		r_init = torch.zeros(ntrials,npop*N)
		r_history = model.integrate(r_init, T,torch.tensor(inp,dtype=torch.float32)) # ntrials, T x N
		r_history = r_history.detach().numpy()
		resps[i,:,:] = r_history[:,-1,:]
	return resps # nstim x ntrials x N

def linear_grating_trials(model,nstims,ntrials,contrast=1,npile_flag=False,npile_structure=None,noise_level=1):
	network_params = model.get_params()
	if 'npop' in network_params:
		npop = network_params['npop']
	else:
		npop = 1
	N = network_params['N']
	directions = np.linspace(0,2*np.pi,nstims)
	resps = np.zeros((nstims,ntrials,npop*N))
	for i,d in enumerate(directions):
		grating_params = {
			'N':N,
			'npop':npop,
			'direction':d,
			'contrast':contrast,
			'ntrials':ntrials,
			'noise_level':noise_level
		}
		inp = get_input(N, 'noisy_grating', grating_params,ntrials) # N,ntrials 
		direct_noise = np.random.normal(0,0.1,(N*npop,ntrials)) # N x ntrials
		if npile_flag:
			# add pile noise
			if npile_structure is None:
				npile_noise = np.random.normal(0,1,(1,ntrials))
				direct_noise += np.tile(npile_noise, (N * npop, 1))
			else:
				x = np.linspace(0, 1, N) 
				center = 0.5  # center of the Gaussian (can be randomized)
				sigma = 0.1   # controls spread of the bump
				npile_structure = np.exp(-((x - center)**2) / (2 * sigma**2)) #(N,)
				structure = npile_structure / np.linalg.norm(npile_structure) # normalize
				cov = np.outer(structure, structure) # (N, N)
				structured_noise = np.random.multivariate_normal(
						mean=np.zeros(N*npop), cov=cov, size=ntrials).T # (N*npop, ntrials)
				direct_noise += structured_noise
				npile_noise = np.random.normal(0,1,(1,ntrials))
				direct_noise += np.tile(npile_noise, (N * npop, 1))

		r_history = model.get_fp(torch.tensor(inp,dtype=torch.float32),direct_noise) # N x ntrials
		r_history = r_history.detach().numpy()
		resps[i,:,:] = r_history.T
	return resps # nstim x ntrials x N

def linear_grating_opto(model,loc,nstims,ntrials,contrast=1,npile_flag=False,npile_structure=None,noise_level=1):
	network_params = model.get_params()
	if 'npop' in network_params:
		npop = network_params['npop']
	else:
		npop = 1
	N = network_params['N']
	directions = np.linspace(0,2*np.pi,nstims)
	resps = np.zeros((nstims,ntrials,npop*N))
	avg_infls = np.zeros((nstims,ntrials)) # average influence
	for i,d in enumerate(directions):
		grating_params = {
			'N':N,
			'npop':npop,
			'direction':d,
			'contrast':contrast,
			'ntrials':ntrials,
			'noise_level':noise_level
		}
		inp = get_input(N, 'noisy_grating', grating_params,ntrials) # N,ntrials 
		direct_noise = np.random.normal(0,0.1,(N*npop,ntrials)) # N x ntrials
		if npile_flag:
			# add pile noise
			if npile_structure is None:
				npile_noise = np.random.normal(0,1,(1,ntrials))
				direct_noise += np.tile(npile_noise, (N * npop, 1))
			else:
				x = np.linspace(0, 1, N) 
				center = 0.5  # center of the Gaussian (can be randomized)
				sigma = 0.1   # controls spread of the bump
				npile_structure = np.exp(-((x - center)**2) / (2 * sigma**2)) #(N,)
				structure = npile_structure / np.linalg.norm(npile_structure) # normalize
				cov = np.outer(structure, structure) # (N, N)
				structured_noise = np.random.multivariate_normal(
						mean=np.zeros(N*npop), cov=cov, size=ntrials).T # (N*npop, ntrials)
				direct_noise += structured_noise
				npile_noise = np.random.normal(0,1,(1,ntrials))
				direct_noise += np.tile(npile_noise, (N * npop, 1))

		opto_params = {
			'N':N,
			'npop':npop,
			'pop':'E',
			'location':loc,
			'opto_strength':2
		}
		opto = get_input(N, 'opto', opto_params)
		r =  model.get_fp(torch.tensor(inp,dtype=torch.float32),direct_noise) # N x ntrials 
		r = r.detach().numpy()
		r_history = model.get_fp(torch.tensor(inp,dtype=torch.float32),direct_noise+opto) # N x ntrials
		r_history = r_history.detach().numpy()
		infl = r_history - r
		avg_infl = np.nanmean(np.delete(infl, loc), axis=0) # shape (ntrials,)
		avg_infls[i,:] = avg_infl
		resps[i,:,:] = r_history.T

	return resps,np.nanmean(avg_infls) # nstim x ntrials x N



def linear_avg_infl_spont(model_params,nlocs,ntrials=1,return_max_eigval=False):
	params = default_params('custmized',custmized_params=model_params)
	if 'slope' in model_params.keys():
		params.update({'slope':model_params['slope']})
	if 'slope_e' in model_params.keys():
		params.update({'slope_e':model_params['slope_e']})
	if 'slope_i' in model_params.keys():
		params.update({'slope_i':model_params['slope_i']})
	model = generate_model(params,linear=True)
	N = params['N']
	npop = params['npop']
	# randomly sample nlocs locations
	locs = np.random.choice(N,nlocs,replace=False) 
	avg_infl_e = []
	avg_infl_i = []
	## loop through locations
	for i,loc in enumerate(locs):
		## spontaneous activity
		spont_params = {
			'npop':npop
		}
		inp = get_input(N,'spont',spont_params)
		r = model.get_fp(inp)
		## opto-perturbed activity
		opto_params = {
			'N':N,
			'npop':npop,
			'pop':'E',
			'location':loc,
			'opto_strength':1
		}
		opto = get_input(N, 'opto', opto_params)
		r_opto = model.get_fp(inp,opto)
		## influence
		infl = r_opto - r # (N,ntrials)
		## trial-average influence
		infl_e = np.nanmean(infl[0:N,:],axis=1)
		infl_i = np.nanmean(infl[N:npop*N,:],axis=1)
		## remove opto-targeted neuron
		infl_e = np.delete(infl_e,loc)
		infl_i = np.delete(infl_i,loc)
		## average influece
		avg_infl_e.append( np.nanmean(infl_e))
		avg_infl_i.append( np.nanmean(infl_i))

	if return_max_eigval:
		## get max eigenvalue
		max_eigval = model.get_max_eigenvalue()
		return np.nanmean(avg_infl_e), np.nanmean(avg_infl_i), max_eigval
	else:
		return np.nanmean(avg_infl_e), np.nanmean(avg_infl_i)
	
def linear_local_infl_spont(model_params,nlocs,ntrials=1,local_dist=2):
	if local_dist < 1:
		print('Warning: local_dist should be at least 1, setting to 1')
		local_dist = 1
	params = default_params('custmized',custmized_params=model_params)
	if 'slope' in model_params.keys():
		params.update({'slope':model_params['slope']})
	if 'slope_e' in model_params.keys():
		params.update({'slope_e':model_params['slope_e']})
	if 'slope_i' in model_params.keys():
		params.update({'slope_i':model_params['slope_i']})
	model = generate_model(params,linear=True)
	N = params['N']
	npop = params['npop']
	# randomly sample nlocs locations
	locs = np.random.choice(N,nlocs,replace=False) 
	local_infl_e = []
	local_infl_i = []
	## loop through locations
	for i,loc in enumerate(locs):
		## spontaneous activity
		spont_params = {
			'npop':npop
		}
		inp = get_input(N,'spont',spont_params)
		r = model.get_fp(inp)
		## opto-perturbed activity
		opto_params = {
			'N':N,
			'npop':npop,
			'pop':'E',
			'location':loc,
			'opto_strength':1
		}
		opto = get_input(N, 'opto', opto_params)
		r_opto = model.get_fp(inp,opto)
		## influence
		infl = r_opto - r # (N,ntrials)
		## trial-average influence
		infl_e = np.nanmean(infl[0:N,:],axis=1)
		infl_i = np.nanmean(infl[N:N*npop,:],axis=1)
		## remove opto-targeted neuron
		infl_e = np.delete(infl_e,loc)
		infl_i = np.delete(infl_i,loc)
		## local influence
		local_infl_e.append( np.nanmean(infl_e[np.abs(np.arange(N-1)-loc)<=local_dist]))
		local_infl_i.append( np.nanmean(infl_i[np.abs(np.arange(N-1)-loc)<=local_dist]))
	return np.nanmean(local_infl_e), np.nanmean(local_infl_i)

def linear_infl_ncorr(model_params,loc,return_corr=False,return_sbound=False, npile_flag=False):
	params = default_params('custmized',custmized_params=model_params)
	model = generate_model(params,linear=True)
	N = model.get_params()['N']
	npop = model.get_params()['npop']
	if 'slope' in params.keys():
		contrast = params['slope']
	else:
		contrast = 1
	resp = linear_grating_trials(model,nstims=8,ntrials=10,contrast=contrast,npile_flag=npile_flag) #nstim,ntrial,N
	grating_avg = np.mean(resp,axis=1)
	grating_noise = resp - grating_avg[:,None,:]
	scorr = np.corrcoef(grating_avg.T)
	ncorr = np.corrcoef(grating_noise.reshape((-1,200)).T)
	opto_params = {
		'N':N,
		'npop':npop,
		'pop':'E',
		'location':loc,
		'opto_strength':1
	}
	opto = get_input(N, 'opto', opto_params)
	infl,sbound_eff = model.get_fp(torch.tensor(np.zeros_like(opto),dtype=torch.float32),opto,return_r=True)
	# infl_e = np.delete(infl,loc)
	# ncorr_e = np.delete(ncorr[:N,loc],loc)
	_,kmax = linear_max_freq(params)
	# print('kmax:',kmax,flush=True)
	if return_sbound and return_corr:
		return infl[:N], ncorr[:N,loc], scorr, ncorr, sbound_eff
	elif return_corr:
		return infl[:N], ncorr[:N,loc], scorr,ncorr
	elif return_sbound:
		return infl[:N], ncorr[:N,loc], sbound_eff
	else:
		return infl[:N], ncorr[:N,loc]

	
def autoval_distr(G):
    Frequenze=[]
    Autoval=[]

    valsG,vecsG = np.linalg.eig(G)
    N=G.shape[0]
    autovettori= vecsG.T
    frequenze=[]
    autoval=[]

    seed(3000)

    series = [gauss(0.0, 1.0) for i in range(N)]
    series = Series(series)
    for i in range(N):    
            samplingFrequency   = N/10
            samplingInterval       = 1 / samplingFrequency

            beginTime = 0
            endTime = 100

            time = np.arange(beginTime, endTime, samplingInterval)

            amplitude =autovettori[i]

            fourierTransform = np.fft.fft(amplitude)/len(amplitude)           # Normalize amplitude

            fourierTransform = fourierTransform[range(int(len(amplitude)/2))] # Exclude sampling frequency

            tpCount     = len(amplitude)

            values      = np.arange(int(tpCount/2))

            timePeriod  = tpCount/samplingFrequency

            frequencies = values/timePeriod
            frequenze.append(frequencies[np.argmax(abs(fourierTransform ))])
            autoval.append(valsG[i])
    frequenze, autoval= zip(*sorted(zip(frequenze, autoval)))
    Frequenze.append(frequenze)
    Autoval.append(autoval)

    return Frequenze[0], Autoval[0]

def linear_max_freq(model_params):
	params = default_params('custmized',custmized_params=model_params)
	model = generate_model(params,linear=True)
	w = model.get_params()['W']
	freqs, eigvals = autoval_distr(w)
	max_eig = np.max(np.real(eigvals))
	max_k_index = np.argmax(np.real(eigvals))
	print('max_freq:', max_k_index,flush=True)
	return max_eig, freqs[max_k_index]


	

	