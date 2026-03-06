import torch
import numpy as np
from random import seed, gauss
from pandas import Series
try:
	from .ring import RingModel,LinearModel,ff_connections,recurrent_connections
except ImportError:
    # fallback for script or notebook usage
    from ring import RingModel, LinearModel, ff_connections, recurrent_connections


def freq_distribution(G):
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

def get_wavelength(w_rec):
	freqs, eigvals = freq_distribution(w_rec)
	max_eig = np.max(np.real(eigvals))
	max_k_index = np.argmax(np.real(eigvals))
	print('max_freq:', max_k_index,flush=True)
	return max_eig, freqs[max_k_index]


def generate_weights(params):
	N = 100
	default_params_ff = {
		'N': N,
		'npop': 2,
		'sigma': 5,
	}
	default_params_ff.update(params)
	w = 0.8*ff_connections(default_params_ff['N'],fftype='Gaussian',params=default_params_ff)
	default_params_rec = {
		'N': N,
		'npop': 2,
		'r': None,
		'sigma': 5,
		'wee': 1,
		'wie': 1,
		'wei': 1,
		'wii': 1,
		'sigma_ie': 5,
		'sigma_ii': 5*1.8,
		'sigma_ei': 5*1.8,
	}
	default_params_rec.update(params)
	m = recurrent_connections(default_params_rec['N'], rtype='MH', params=default_params_rec)
	return {
		'N': default_params_rec['N'],
		'W': torch.tensor(m, dtype=torch.float32),
		'W_ff': torch.tensor(w, dtype=torch.float32),
		'npop': default_params_rec['npop'],
		'sigma': default_params_rec.get('sigma', None),
	}

def default_params(kw = None,custmized_params=None,unit_ff=False):
	"""Load default parameters for the model."""
	# Default parameters
	if kw is None:
		N = 100
		
		inh_a = 1.8
		sigma_m = 5
		
		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			# 'r': 0.9,
			'r':0.5,
			'wee':1,
			'wie':1,
			'wei': 1,
			'wii': 1,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		
		sigma_w = sigma_m*2
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		if unit_ff:
			w = ff_connections(N,fftype='uniform',params=ffparams)
		else:
			w = 0.8*ff_connections(N,fftype='Gaussian',params=ffparams)

		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		return params
	
	if kw == 'custmized':
		if custmized_params is None:
			raise ValueError('custmized parameters must be provided')
		return generate_weights(custmized_params)


	## same as default, but no r normalization
	elif kw == 'no_norm':
		N = 100
		sigma_w = 5
		inh_a = 2
		sigma_m = 5
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		w = 0.8*ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None, # No normalization
			'wee':1,
			'wie':1,
			'wei': 1,
			'wii': 1,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,

		}

		m = recurrent_connections(N,rtype='MH',params=rec_params)
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		return params

	elif kw == 'weak_mh':
		N = 100
		sigma_w = 10
		inh_a = 2
		sigma_m = 5
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		w = 0.8*ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': 0.5,
			'wee':1,
			'wie':1,
			'wei': 1,
			'wii': 1,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}

		m = recurrent_connections(N,rtype='MH',params=rec_params)
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		return params
	
	elif kw == 'low_self_inhibition':
		N = 100
		sigma_w = 10
		inh_a = 2
		sigma_m = 5
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		w = 0.8*ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':0.01,
			'wie':0.32,
			'wei': 0.32,
			'wii': 0.1,
			'noise_level':0,
			'sigma_ie':1,
			'sigma_ii':2,
			'sigma_ei':2,
			'ii_ap': 0.11,
			'ii_an': 0.1,
			'ii_sp': sigma_m,
			'ii_sn':sigma_m*0.8,
		}

		m = recurrent_connections(N,rtype='noisy_MH_lile_inh',params=rec_params)
		
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		return params

	elif kw == 'strong_ei':
		N = 100
		sigma_w = 15
		inh_a = 1.8
		sigma_m = 5
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		w = ff_connections(N,fftype='Gaussian',params=ffparams)
		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':0.01,
			'wie':0.15,
			'wei': 0.4,
			#'wii': 0.1,
			'wii': 0.05,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		return params
	
	elif kw == 'strong_eie':
		N = 100
		sigma_w = 15
		inh_a = 1.8
		sigma_m = 5
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		w = ff_connections(N,fftype='Gaussian',params=ffparams)
		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':0.5,
			'wie': 2,
			'wei': 2,
			#'wii': 0.1,
			'wii': 0.5,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		return params
	
	elif kw == 'strong_eie_weak_rec':
		N = 100
		sigma_w = 10
		inh_a = 2.5
		sigma_m = 5
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		w = ff_connections(N,fftype='Gaussian',params=ffparams)
		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':0.1,
			'wie': 0.8,
			'wei': 0.8,
			#'wii': 0.1,
			'wii': 0.1,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		return params

	elif kw == 'analytic':
		N = 100
		sigma_w = 5
		inh_a = 2
		sigma_m = 5
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		w = ff_connections(N,fftype='Gaussian',params=ffparams)
		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':0.1,
			'wie':0.8,
			'wei': 0.8/inh_a,
			#'wii': 0.1,
			'wii': 0.1/inh_a,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}

		return params
	
	elif kw == 'eg_cross_pop':
		N = 100
		
		inh_a = 1.5
		sigma_m = 10
		sigma_w = sigma_m
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		if unit_ff:
			w = 2*ff_connections(N,fftype='uniform',params=ffparams)
		else:
			w = 2*ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':1.1,
			# 'wee':1.,
			'wie':2.89,
			'wei': 2.89,
			'wii': 2.5,
			# 'wii': 1.,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		
		return params
	
	elif kw == 'rand_cross_pop':
		N = 100
		
		inh_a = 1.9
		sigma_m = 18
		sigma_w = sigma_m
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		if unit_ff:
			w = 2*ff_connections(N,fftype='uniform',params=ffparams)
		else:
			w = 2*ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':1.1,
			'wie':3.3,
			'wei': 3.3,
			'wii': 2.5,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		print('debug: random MH')	
		m = recurrent_connections(N,rtype='rand_MH',params=rec_params)
		
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		
		return params
	
	elif kw == 'rand_mh':
		N = 100
		
		inh_a = 1.9
		sigma_m = 18
		sigma_w = sigma_m
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		if unit_ff:
			w = 2*ff_connections(N,fftype='uniform',params=ffparams)
		else:
			w = 2*ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':2.4,
			'wie':2.48,
			'wei': 2.48,
			'wii': 2.5,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		print('debug: random MH')	
		m = recurrent_connections(N,rtype='rand_MH',params=rec_params)
		
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		
		return params
	
	elif kw == 'border':
		N = 100
		sigma_w = 12
		inh_a = 1.8
		sigma_m = 10
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		w = ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':1.3,
			'wie':2.6,
			'wei': 2.6,
			'wii': 2.5,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
			'border_flag':True
		}
		
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		
		return params
	
	elif kw == 'eg_mh':
		N = 100
		inh_a = 1.5
		sigma_m = 10
		sigma_w = sigma_m
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		if unit_ff:
			w = ff_connections(N,fftype='uniform',params=ffparams)
		else:
			w = ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':2.49,
			'wie':2.39,
			'wei': 2.39,
			'wii': 2.5,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		return params
	
	elif kw == 'eg1_cd':
		N = 100
		
		inh_a = 1.5
		sigma_m = 2
		sigma_w = sigma_m
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		if unit_ff:
			w = 2*ff_connections(N,fftype='uniform',params=ffparams)
		else:
			w = 2*ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':1.1,
			'wie':2.3,
			'wei': 2.3,
			'wii': 1.12,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		
		return params
	
	elif kw == 'eg2_cd':
		N = 100
		
		inh_a = 1.5
		sigma_m = 2
		sigma_w = sigma_m
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		if unit_ff:
			w = 2*ff_connections(N,fftype='uniform',params=ffparams)
		else:
			w = 2*ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':0.9,
			'wie':2.5,
			'wei': 2.5,
			'wii': 1.15,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		
		return params
	
	elif kw == 'eg1_mh':
		N = 100
		
		inh_a = 1.5
		sigma_m = 2
		sigma_w = sigma_m
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		if unit_ff:
			w = 2*ff_connections(N,fftype='uniform',params=ffparams)
		else:
			w = 2*ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':1.1,
			'wie':0.9,
			'wei': 0.9,
			'wii': 1.12,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		
		return params
	
	elif kw == 'backup':
		N = 100
		inh_a = 1.5
		sigma_m = 7
		sigma_w = sigma_m
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		if unit_ff:
			w = ff_connections(N,fftype='uniform',params=ffparams)
		else:
			w = ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':1.01,
			'wie':3,
			'wei': 3,
			'wii': 1.5,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		return params

	elif kw == 'backup1':
		N = 100
		inh_a = 1.5
		sigma_m = 6.5
		sigma_w = sigma_m
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		if unit_ff:
			w = ff_connections(N,fftype='uniform',params=ffparams)
		else:
			w = ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':1.01,
			'wie':1.5,
			'wei': 1.5,
			'wii': 2,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		return params	

	elif kw == 'backup2':
		N = 100
		inh_a = 1.5
		sigma_m = 11
		sigma_w = sigma_m
		ffparams = {
			'N': N,
			'npop':2,
			'sigma': sigma_w,
		}
		if unit_ff:
			w = ff_connections(N,fftype='uniform',params=ffparams)
		else:
			w = ff_connections(N,fftype='Gaussian',params=ffparams)

		rec_params ={
			'N':N,
			'npop':2,
			'sigma':sigma_m,
			'r': None,
			'wee':1.01,
			'wie':2,
			'wei': 2,
			'wii': 1.02,
			'sigma_ie':sigma_m,
			'sigma_ii':sigma_m*inh_a,
			'sigma_ei':sigma_m*inh_a,
		}
		
		m = recurrent_connections(N,rtype='MH',params=rec_params)
		
		params = {
			'N':N,
			'W':torch.tensor(m,dtype=torch.float32),
			'W_ff':torch.tensor(w,dtype=torch.float32),
			'npop':2,
			'sigma':sigma_m,
		}
		return params	


def vanilla_ring():
	N = 100
	sigma_w = 10
	inh_a = 2
	sigma_m = 5
	ffparams = {
		'N': N,
		'npop':2,
		'sigma': sigma_w,
	}
	w = 0.8*ff_connections(N,fftype='Gaussian',params=ffparams)

	rec_params ={
		'N':N,
		'npop':2,
		'sigma':sigma_m,
		'r': 0.9,
		'wee':1,
		'wie':1,
		'wei': 1/inh_a,
		'wii': 1/inh_a,
		'sigma_ie':5,
		'sigma_ii':5*inh_a,
		'sigma_ei':5*inh_a,
	}

	m = recurrent_connections(N,rtype='MH',params=rec_params)
	f = torch.sigmoid
	T = 10000
	dt = 0.02
	tau= 0.01
	network_params = {
		'N':N,
		'npop':2,
		'tau':tau,
		'f':f,
		'W':torch.tensor(m,dtype=torch.float32),
		'W_ff':torch.tensor(w,dtype=torch.float32),
		# 'dt':dt
	}
	model = RingModel(network_params)
	return model


if __name__ == '__main__':
	pass