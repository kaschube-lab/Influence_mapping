import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
from pandas import Series
from scipy import signal
from random import seed, gauss
import os, sys
from scipy.spatial.distance import cdist
import statsmodels.api as sm
import torch
# from torchdiffeq import odeint

def relu(x):
	return x * (x > 0)

def powlaw(u):
	return  np.maximum(0,u)**2

def ff_connections(N,fftype='Gaussian',params=None):
	if fftype == 'Gaussian':
		sigma = params['sigma']
		if 'npop' in params:
			npop = params['npop']
		else:
			npop = 1
		xx, yy = np.meshgrid(np.arange(0,N,1),np.arange(0,1,1))
		dx = cdist(xx.T,xx.T)
		dx = np.minimum(dx , N*np.ones_like(dx)-dx)
		
		if npop == 1:
			w = np.exp(- dx**2/sigma**2)
		elif npop == 2:
			we = np.exp(- dx**2/sigma**2)
			w = np.block([[we,we],[we,we]])
		return w/N
	if fftype == 'uniform':
		if 'npop' in params:
			npop = params['npop']
		else:
			npop = 1
		# identity matrix
		w = np.eye(npop*N)
		return w
		
def random_field(N):
	sigma=4
	M=1
	xx, yy = np.meshgrid(np.arange(0,N,1),np.arange(0,M,1))
	dx = cdist(xx.T,xx.T)
	dx = np.minimum(dx , N*np.ones_like(dx)-dx)
	series1 = [gauss(0.0, 1.0) for k in range(N)]
	grf = Series(series1)

	w=(1/(np.sqrt(2*np.pi)*sigma))* np.exp(- dx**2/(2*sigma**2))
	D=int(N/2)
	correzione=np.zeros((N,N))
	for i in range(N):
			correzione.T[i]= (signal.convolve(w.T[D],grf, mode='same'))
	return correzione 


def gauss_func(x,sigma,a):
	return (a/(np.sqrt(2*np.pi)*sigma))*np.exp(-x**2/(2*sigma**2))


def mh_1d(x, sigma_e, sigma_i, w_e, w_i):
    return gauss_func(x, sigma_e,w_e) - gauss_func(x, sigma_i,w_i)

def recurrent_connections(N,rtype='Gaussian',params=None):

	if rtype == 'Gaussian':
		sigma = params['sigma']
		xx, yy = np.meshgrid(np.arange(0,N,1),np.arange(0,1,1))
		dx = cdist(xx.T,xx.T)
		dx = np.minimum(dx , N*np.ones_like(dx)-dx)
		if 'npop' in params:
			npop = params['npop']
		else:
			npop = 1
		if npop == 1:
			m = np.exp(- dx**2/sigma**2)
		elif npop == 2:
			we = np.exp(- dx**2/sigma**2)
			m = np.block([[we,we],[we,we]])
		
	if 'random' in rtype:
		np.random.seed(0)
		
		if 'npop' in params:
			npop = params['npop']
		else:
			npop = 1
		if npop == 1:
			if 'gaussian' in params:
				m = np.random.normal(0,1,(N,N))
			else:
				m = np.random.rand(N,N)
		elif npop == 2:
			aee = params['wee']
			aei = params['wei']
			aie = params['wie']
			aii = params['wii']
			if 'gaussian' in params:
				wee = aee * np.random.normal(0,1,(N,N))
				wei = - aei * np.random.normal(0,1,(N,N))
				wie = aie * np.random.normal(0,1,(N,N))
				wii = - aii * np.random.normal(0,1,(N,N))
			else:
				wee = aee * np.random.rand(N,N)
				wei = - aei * np.random.rand(N,N)
				wie = aie * np.random.rand(N,N)
				wii = - aii * np.random.rand(N,N)
			m = np.block([[wee,wei],[wie,wii]])
		r = params['r'] # spectral bound
		vals,vecs = np.linalg.eig(m)
		s_bound = np.max(np.real(vals))
		# print('r',s_bound)
		if r is not None:
			m = m/s_bound * r
		else:
			m = m
		return m

	# print('debug: rtype', rtype)
	if 'MH' in rtype:
		r = params['r'] # spectral bound
		sigma_m = params['sigma'] # spread of excitatory connections
		## 1 or 2 populations
		if 'npop' in params:
			npop = params['npop']
		else:
			npop = 1
		## parameters for 1 population
		if npop == 1:
			ei_ratio = params['ei_ratio']
			inh_a = params['inh_a']
		## parameters for 2 populations
		elif npop == 2:
			wee = params['wee']
			wei = params['wei']
			wie = params['wie']
			wii = params['wii']
			sigma_ii = params['sigma_ii']
			sigma_ie = params['sigma_ie']
			sigma_ei = params['sigma_ei']

		xx, yy = np.meshgrid(np.arange(0,N,1),np.arange(0,1,1))
		dx = cdist(xx.T,xx.T)
		dx = np.minimum(dx , N*np.ones_like(dx)-dx)
		if npop == 1:
			m = ei_ratio * np.exp(- dx**2/sigma_m**2) - 1/inh_a*np.exp(- dx**2/(inh_a*sigma_m)**2)
		elif npop == 2:
			# mee = wee * np.exp(- dx**2/sigma_m**2)
			mee = gauss_func(dx,sigma_m,wee)
			mie = gauss_func(dx,sigma_ie,wie)
			mei = - gauss_func(dx,sigma_ei,wei)
			# mie = wie * np.exp(- dx**2/sigma_ie**2)
			# mei = - wei * np.exp(- dx**2/sigma_ei**2)
			if 'crossinh' in rtype:
				# set local inhibition (set diagnoal and near diagonal entries of mei) to zero
				np.fill_diagonal(mei, 0)
				np.fill_diagonal(mei[1:], 0)  # Below the diagonal
				np.fill_diagonal(mei[:, 1:], 0)  # Above the diagonal		
			# mii = - wii * np.exp(- dx**2/sigma_ii**2)
			mii = - gauss_func(dx,sigma_ii,wii)
			if 'lile_inh' in rtype:
				# wii is a difference between two gaussians
				ii_ap = params['ii_ap']
				ii_an = params['ii_an']
				ii_sp = params['ii_sp']
				ii_sn = params['ii_sn']
				# iip = ii_ap * np.exp(- dx**2/ii_sp**2)
				# iin = ii_an * np.exp(- dx**2/ii_sn**2)
				iip = gauss_func(dx,ii_sp,ii_ap)
				iin = gauss_func(dx,ii_sn,ii_an)
				mii = - wii * (iip - iin)
			# print('debug: shpaes',mee.shape,mei.shape,mie.shape,mii.shape,flush=True)
			## check if tensor
			if type(mee) == np.ndarray and type(mei) == np.ndarray and type(mie) == np.ndarray and type(mii) == np.ndarray:
				m = np.block([[mee,mei],[mie,mii]])
			else:
				m = torch.cat([
					torch.cat([torch.tensor(mee),mei], dim=1),  # Concatenate along columns (dim=1)
					torch.cat([mie, mii], dim=1)  # Concatenate along columns (dim=1)
				], dim=0)  # Concatenate along rows (dim=0)

		## noise: Gaussian filtered with Gaussian random field
		if 'noisy_MH' in rtype:
			noise_level = params['noise_level']
			f1=random_field(npop*N)
			# repeat the random field for npop times
			# random_fields = np.tile(f1,(npop,1,1))
			m = (1+(noise_level*f1))*m


		# add a block random matrix to Gaussians
		if 'rand_MH' in rtype:
			rand_ee = np.random.normal(loc=1.5, scale=0.5, size=(N, N))
			rand_ei = np.random.normal(loc=0.5, scale=0.5, size=(N, N))
			rand_ie = - np.random.normal(loc=1, scale=0.5, size=(N, N))
			rand_ii = - np.random.normal(loc=1.3, scale=0.5, size=(N, N))
			rand_m = np.block([[rand_ee,rand_ei],[rand_ie,rand_ii]])

			m = m + 0.006*rand_m
			# print('debug: max random block', np.max(rand_m), flush=True)


		# print('debug:  m',m,flush=True)
		vals,vecs = np.linalg.eig(m)
		s_bound = np.max(np.real(vals))
		# print('r',s_bound)
		if r is not None:
			m = m/s_bound * r
		else:
			m = m
	return m


def find_kmax_block(G, N, tol=1e-6):
	"""
	Find dominant spatial mode of a 2N x 2N block recurrent matrix.

	Parameters
	----------
	G : ndarray (2N, 2N)
		Connectivity matrix.
	N : int
		Number of neurons per population.
	tol : float
		Tolerance below which eigenvalues are treated as zero.

	Returns
	-------
	result : tuple (m, k_rad, wavelength, max_eig)
	"""
	vals, vecs = np.linalg.eig(G)
	idx = np.argmax(np.real(vals))
	max_eig = np.real(vals[idx])
	v = vecs[:, idx]

	# take excitatory half
	v_exc = v[:N]
	V = np.fft.fft(v_exc)
	m = np.argmax(np.abs(V[:N//2]))  # best spatial frequency bin

	if m == 0 or max_eig <= tol:
		return  0, 0.0,np.nan
	else:
		k_rad = 2*np.pi*m / N
		wavelength = N / m
		return  m, k_rad,wavelength
	
def get_all_modes(G, N, tol=1e-6):
	"""
	Compute spatial frequencies and associated eigenvalues
	from a 2N x 2N block recurrent matrix.

	Parameters
	----------
	G : ndarray (2N, 2N)
		Connectivity matrix.
	N : int
		Number of neurons per population.
	tol : float
		Tolerance below which eigenvalues are treated as zero.

	Returns
	-------
	modes : dict
		Dictionary with keys:
			'eigval'     : list of eigenvalues
			'm'          : list of spatial frequency bins
			'k_rad'      : list of wave numbers in radians
			'wavelength' : list of spatial wavelengths
			'fft_amp'    : list of Fourier amplitudes
	"""
	vals, vecs = np.linalg.eig(G)

	modes = {
		'eigval': [],
		'm': [],
		'k_rad': [],
		'wavelength': [],
		'fft_amp': []
	}

	for i, eigval in enumerate(np.real(vals)):
		if eigval <= tol:
			continue

		v = vecs[:, i]
		v_exc = v[:N]
		V = np.fft.fft(v_exc)

		# Loop over positive frequencies only
		for m in range(1, N//2):
			fft_amp = np.abs(V[m])
			if fft_amp > tol:
				k_rad = 2*np.pi*m / N
				wavelength = N / m

				modes['eigval'].append(eigval)
				modes['m'].append(m)
				modes['k_rad'].append(k_rad)
				modes['wavelength'].append(wavelength)
				modes['fft_amp'].append(fft_amp)

	return modes
	
		
def freq_distri(G):
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

def get_input(N, stim_type, params=None,ntrials=1):
	if 'npop' in params:
		npop = params['npop']
	else:
		npop = 1
	
	if stim_type == 'spont':
		inp = np.random.random((npop*N,ntrials))
		if 'baseline' in params:
			baseline_e = params['baseline']
			baseline_i = params['baseline']
		elif 'baseline_e' in params:
			baseline_e = params['baseline_e']
			baseline_i = params['baseline_i']
		else:
			baseline_e = 1
			baseline_i = 1
		
		const = np.ones_like(inp)
		const[:N] = const[:N] * baseline_e
		const[N:] = const[N:] * baseline_i
		return inp + const
	
	if stim_type == 'grating':
		direction = params['direction']
		contrast = params['contrast']
		if 'baseline' in params:
			baseline = params['baseline']
		else:
			baseline = 0
		xx, yy = np.meshgrid(np.arange(0,N,1),np.arange(0,1,1))
		inp = np.sin(2*np.pi*xx/N + direction) * contrast # N x 1
		inp = inp + baseline
		if npop == 2:
			inp = np.concatenate((inp,inp),axis=0).reshape(npop*N,1) # 2N x 1
		if ntrials > 1:
			inp = np.tile(inp,(1,ntrials)) # 2N * ntrials
		return inp # ntrials x 2N
	
	if stim_type == 'noisy_grating':
		direction = params['direction']
		contrast = params['contrast']
		xx, yy = np.meshgrid(np.arange(0,N,1),np.arange(0,1,1))
		inp = np.sin(2*np.pi*xx/N + direction) * contrast # N x 1
		if npop == 2:
			inp = np.concatenate((inp,inp),axis=0).reshape(npop*N,1) # 2N x 1
		if ntrials > 1:	
			inp = np.tile(inp,(1,ntrials)) # 2N * ntrials
		if 'noise_level' in params:
			noise_level = params['noise_level']
		else:
			noise_level = 1
		inp = inp + noise_level*np.random.normal(0,0.1,(npop*N,ntrials))
		return inp
	
	if stim_type == 'opto':
		location = params['location']
		if 'opto_strength' in params:
			intensity = params['opto_strength']
		else:
			intensity = 1
		if npop == 1:
			inp = np.zeros((N,1))
			inp[location] = intensity
			if ntrials > 1:
				inp = np.tile(inp,(1,ntrials)) # N x ntrials
			return inp
		elif npop == 2:
			inp = np.zeros((2*N,1))
			if params['pop'] == 'E':
				inp[location] = intensity
			elif params['pop'] == 'I':
				inp[N+location] = intensity
			if ntrials > 1:
				inp = np.tile(inp,(1,ntrials))
			return inp
	
def get_time_input(N, stim_type,T,time_type='constant', params=None):
	if 'npop' in params:
		npop = params['npop']
	else:
		npop = 1
	if 'ntrials' in params:
		ntrials = params['ntrials']
	else:
		ntrials = 1
	if time_type == 'constant':
		single_tp = get_input(N, stim_type, params,ntrials).reshape(1,npop*N,ntrials) # 1x N x ntrials
		inp = np.tile(single_tp,(T,1,1)) #
		
	elif time_type == 'noisy':
		single_tp = get_input(N, stim_type, params,ntrials).reshape(1,npop*N)
		if 'noise_level' in params:
			noise_level = params['noise_level']
		else:
			noise_level = 0.1
		inp = np.tile(single_tp,(T,1,1)) + noise_level*np.random.normal(0,0.1,(T,npop*N,ntrials))
		
	elif time_type == 'ramp':
		single_tp = get_input(N, stim_type, params,ntrials).reshape(1,npop*N)
		if 'ramp_rate' in params:
			ramp_rate = params['ramp_rate']
		else:
			ramp_rate = 0.1
		inp = np.tile(single_tp,(T,1,1)) + np.linspace(0,1,T)[:,None,None]*ramp_rate
	## reshape into ntrials, T, N
	inp = np.transpose(inp,(2,0,1)) # ntrials x T x N
	return torch.tensor(inp) 
	
	

def get_fp( w, m,inp,opto=None):
	# inp: N x 1
	# w: N x N
	# m: N x N
	# (I-M)^-1 * w * inp
	N = m.shape[0]
	# fp_r: N x 1
	if opto is None:
		fp_r = np.linalg.inv(np.eye(N) - m) @ w @ inp 
	else:
		fp_r = np.linalg.inv(np.eye(N) - m) @ ( (w @ inp) + opto)

	#fp_r = np.linalg.inv(np.diag(np.diag(np.ones_like(m))) - m) @ w @ inp
	return fp_r



class RingModel:
	def __init__(self, params, device="cpu"):
		"""
		Initialize the model parameters.
		Args:
		- params (dict): Dictionary of model parameters.
			- N (int): Number of neurons.
			- tau (float): Time constant.
			- f (callable): Activation function.
			- W (torch.Tensor): Coupling matrix (NxN).
			- W_ff (torch.Tensor): Feedforward coupling matrix (NxN).
			- dt (float): Time step for integration.
		- device (str): Device to use ('cpu' or 'cuda').
		"""
		self.N = params['N']
		if 'npop' in params:
			self.npop = params['npop']
		else:
			self.npop = 1
		if 'fe' in params:
			self.fe = params['fe']
		if 'fi' in params:
			self.fi = params['fi']
		else:
			self.f = params['f']
		if 'power' in params:
			self.power = params['power']
		else:
			self.power = 2
		self.W = params['W']
		if 'W_ff' in params:
			self.W_ff = params['W_ff']
		else:
			self.W_ff = None
		if 'dt' in params:
			self.dt = params['dt']
		else:
			self.dt = 0.01
		if 'tau' in params:
			self.tau = params['tau']
		else:
			self.tau = 1.0
		self.device = device

	def get_params(self):
		return vars(self)
	
	def get_max_frequency(self):
		"""
		Compute the maximum frequency of the recurrent network.
		Returns:
		- float: Maximum frequency of the recurrent network.
		"""
		# freqs, eigvals = freq_distri(self.W)
		# max_eig = np.max(np.real(eigvals))
		# max_k_index = np.argmax(np.real(eigvals))
		# print('max_freq:', freqs[max_k_index])
		_,kmax,wavelength = find_kmax_block(self.W.numpy(), self.N)
		print('max_freq:', kmax,'wavelength:', wavelength)
		self.max_freq = kmax
		self.wavelength = wavelength
		return kmax
	
	def get_stability(self):
		"""
		Compute the stability of the recurrent network.
		Returns:
		- float: Maximum eigenvalue of the recurrent network.
		"""
		eigenvalues = torch.linalg.eigvals(self.W).detach().numpy()  # Get eigenvalues of W
		# get max real part of eigenvalues
		max_eigenvalue = np.max(np.real(eigenvalues))
		# print('Max eigenvalue:', max_eigenvalue, flush=True)
		self.sbound = max_eigenvalue
		return max_eigenvalue

	def integrate(self, r_init, T, h, opto=None):
		"""
		Simulate the model dynamics using the Runge-Kutta 4th order method with time-dependent input.
		Args:
		- r_init (torch.Tensor): Initial firing rates (N-dimensional).
		- T (int): Number of time steps to integrate.
		- h (torch.Tensor): Time-dependent input tensor of shape (ntrials, T, N).
		- opto (torch.Tensor): Optional time-dependent optogenetic input tensor of shape (ntrials,T, N).
		Returns:
		- torch.Tensor: Dynamics of firing rates (ntrials,T x N).
		"""
		# Check if the input tensor is 2D
		if len(h.shape) == 2:
			h = h.unsqueeze(0)
		# check if initial condition is 1D
		if len(r_init.shape) == 1:
			r_init = r_init.unsqueeze(0)
		
		
		if self.W_ff is not None:
			self.W_ff = self.W_ff.to(self.device)
			h =  h @ self.W_ff # (ntrials, T, N) @ (N, N) -> (ntrials, T, N)
			
		# print('debug: f', self.f, flush=True)
		# print('debug: power', self.power, flush=True)

		max_eig = self.get_stability()
		# if max_eig >= 1:
		# 	print('Warning: The system may be unstable (max eigenvalue >= 1).', flush=True)

		def dr_dt(self,r, t_idx):
			h_t = h[:,t_idx,:]  # Get external input for the current time step
			if opto is not None:
				h_t = h_t + opto[:,t_idx,:]

			tol_inp = torch.matmul(r,self.W) + h_t # (ntrials, N) 
			if self.f == torch.pow:
				if self.power != 1:
					tol_inp[tol_inp<0] = 0
				res = (-r + self.f(tol_inp,self.power)) / self.tau
			else:
				res = (-r + self.f(tol_inp)) / self.tau

			# ---- Jacobian stability check (trial 0) ----
			# with torch.no_grad():
			# 	Wr = (r[0] @ self.W).detach() # (N,)
			# 	D = torch.diag((Wr > 0).float())  # ReLU derivative
			# 	I = torch.eye(self.W.shape[0], device=self.device)
			# 	J = (-I + D @ self.W) / self.tau
			# 	eigs = torch.linalg.eigvals(J)
			# 	max_real = eigs.real.max().item()
			# 	if max_real > 0:
			# 		print(f"[t={t_idx}] Jacobian unstable: max Re(eig)={max_real:.3f}", flush=True)
			# --------------------------------------------

			return res
			
		
		# Initialize firing rates
		r = r_init.to(self.device) # (ntrials, N)
		r_history = [r]

		
		for t_idx in range(T):
			k1 = self.dt * dr_dt(self,r, t_idx)
			k2 = self.dt * dr_dt(self,r + 0.5 * k1, t_idx)
			k3 = self.dt * dr_dt(self,r + 0.5 * k2, t_idx)
			k4 = self.dt * dr_dt(self,r + k3, t_idx)
			r = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
			r_history.append(r.clone().detach())
		res = torch.stack(r_history, dim=0) # (T+1, ntrials, N)
		# reshape to (ntrials, T+1, N)
		res = res.permute(1,0,2)
		return   res.to("cpu")# Move results back to CPU for analysis
	
	def get_grating_input(self,ori,ntrials,T,contrast=1):
		"""
		Compute the grating input for the given orientation and number of trials.
		Args:
		- ori (float): Orientation of the grating.
		- ntrials (int): Number of trials.
		- T (int): Number of time steps.
		- contrast (float): Contrast of the grating.
		Returns:
		- torch.Tensor: Grating input tensor of shape (ntrials, T, N).
		"""
		## get max_frequency of the recurrent network
		max_freq = self.get_max_frequency()

		grating_input = torch.zeros((ntrials,T,self.npop*self.N))

		x = torch.linspace(0, self.N, self.N)

		# sine input using kmax
		grating_e = contrast * torch.sin(max_freq * x + ori) # (N, 1)

		# repeat for inhibitory neurons
		if self.npop == 2:
			grating = torch.cat((grating_e,grating_e),dim=0).reshape(self.npop*self.N,1) # (2N, 1)

		## repeat for ntrials and T
		grating_input = grating.T.unsqueeze(1).repeat(ntrials,T,1) # (ntrials, T, 2N)

		return grating_input # (ntrials, T, 2N)

	def get_noisy_input(self, ntrials, T, noise_level=0.1):
		"""
		Compute the noisy input for the given number of trials and time steps.
		Args:
		- ntrials (int): Number of trials.
		- T (int): Number of time steps.
		- noise_level (float): Noise level.
		Returns:
		- torch.Tensor: Noisy input tensor of shape (ntrials, T, N).
		"""
		noise = noise_level * torch.randn((ntrials,self.npop*self.N))
		## repeat for T time steps
		noisy_input = noise.unsqueeze(1).repeat(1,T,1) # (ntrials, T, N)
		return noisy_input

	def get_structured_noise_input(self, ntrials, T, noise_level=0.1, sigma=4, structure='flat', structure_params=None):
		"""
		Compute the structured noise input for the given number of trials and time steps.
		Args:
		- ntrials (int): Number of trials.
		- T (int): Number of time steps.
		- noise_level (float): Noise level.
		- sigma (float): Standard deviation of the Gaussian noise.
		- structure (str): Structure of the noise ('flat' or 'gaussian').
		- structure_params (dict): Additional parameters for the noise structure.
		Returns:
		- torch.Tensor: Structured noise input tensor of shape (ntrials, T, N).
		"""

		if structure == 'flat':
			noise = torch.randn((1,ntrials)) * noise_level #(1,ntrials)
			noise = noise.repeat(self.N*self.npop, 1) #(N x ntrials)
		elif structure == 'gaussian':
			base_mean = structure_params['base_mean']
			base_std = structure_params['base_std']
			# neuron axis (0 ... N)
			x = torch.arange(self.N).float().unsqueeze(1)  # (N,1)
			profiles = []
			for t in range(ntrials):
				# jitter mean and std around base values
				mu_t = base_mean +  torch.randn(1)
				sigma_t = base_std + torch.randn(1)
				amp_t = torch.abs(torch.randn(1))
				# Gaussian profile across neurons
				profile = amp_t * torch.exp(-0.5 * ((x - mu_t) / sigma_t)**2)
				profiles.append(profile)
			noise_e = torch.cat(profiles, dim=1)  # (N, ntrials)
			## repeat for I to get full noise
			noise = noise_e.repeat(2, 1) * noise_level  # (2N, ntrials)

		else:
			raise ValueError(f"Unknown structure: {structure}")
		
		## repeat for T time steps
		structured_noise_input = noise.T.unsqueeze(1).repeat(1,T,1) # (ntrials, T, N)
		return structured_noise_input

	def get_grating_resp(self,input_params,return_input=False):
		"""
		Compute the grating response for the given input parameters.
		Args:
		- input_params (dict): Dictionary of input parameters.
			- ori (float): Orientation of the grating.
			- ntrials (int): Number of trials.
			- T (int): Number of time steps.
			- contrast (float): Contrast of the grating.
			- noise_level (float): Noise level.
			- structured_noise (bool): Whether to add structured noise.
			- structure (str): Structure of the noise ('flat' or 'gaussian').
			- structure_params (dict): Additional parameters for the noise structure.
		- return_input (bool): Whether to return the input tensor along with the response.
		Returns:
		- torch.Tensor: Grating response tensor of shape (nstims, ntrials, T, N).
		- torch.Tensor (optional): Input tensor of shape (nstims, ntrials, T, N).
		"""
		nstims = input_params['nstims']
		ntrials = input_params['ntrials']
		T = input_params['T']
		if 'contrast' in input_params:
			contrast = input_params['contrast']
		else:
			contrast = 1
		if 'noise_level' in input_params:
			noise_level = input_params['noise_level']
		else:
			noise_level = 0.1
		if 'structured_noise' in input_params:
			structured_noise = input_params['structured_noise']
			structure = input_params['structure']
			structure_params = input_params['structure_params']
		else:
			structured_noise = False
			structure = 'flat'
			structure_params = {}
		
		stims = torch.linspace(0, np.pi, nstims)
		grating_responses = []
		all_inputs = []
		for ori in stims:
			grating_input = self.get_grating_input(ori,ntrials,T,contrast) # (ntrials, T, N)
			independent_noise = self.get_noisy_input(ntrials,T,noise_level) # (ntrials, T, N)
			shared_noise = self.get_structured_noise_input(ntrials,T,noise_level,sigma=4,structure='flat',structure_params=structure_params)
			total_input = grating_input + independent_noise + shared_noise
			all_inputs.append(total_input)
			# initial condition
			r_init = torch.zeros((ntrials,self.npop*self.N))
			resp = self.integrate(r_init,T,total_input) # (ntrials, T+1, N)
			grating_responses.append(resp)
		grating_responses = torch.stack(grating_responses, dim=0) # (nstims, ntrials, T+1, N)
		all_inputs = torch.stack(all_inputs, dim=0) # (nstims, ntrials, T, N)
		if return_input:
			return grating_responses, all_inputs
		else:
			return grating_responses
		
	def get_opto_input(self, ntrials, T, location, intensity=1, pop='E'):
		"""
		Compute the optogenetic input for the given location and intensity.
		Args:
		- ntrials (int): Number of trials.
		- T (int): Number of time steps.
		- location (int): Location of the optogenetic stimulation.
		- intensity (float): Intensity of the optogenetic stimulation.
		- pop (str): Population to stimulate ('E' or 'I').
		Returns:
		- torch.Tensor: Optogenetic input tensor of shape (ntrials, T, N).
		"""
		opto_input = torch.zeros((ntrials,T,self.npop*self.N))
		if self.npop == 1:
			opto_input[:,:,location] = intensity
		elif self.npop == 2:
			if pop == 'E':
				opto_input[:,:,location] = intensity
			elif pop == 'I':
				opto_input[:,:,self.N+location] = intensity
			else:
				raise ValueError(f"Unknown population: {pop}")
		return opto_input # (ntrials, T, N)
	
	def get_grating_and_opto_resp(self, input_params,return_input=False):
		"""
		Compute the grating and optogenetic response for the given parameters.
		Args:
		- nstim (int): Number of stimuli.
		- ntrials (int): Number of trials.
		- loc (int): Location of the stimulus.
		- contrast (float): Contrast of the stimulus.
		- base_line (float): Baseline firing rate.
		- noise_level (float): Level of noise in the stimulus.
		- neuron_noise (float): Level of noise in the neurons.
		Returns:
		- torch.Tensor: Grating and optogenetic response tensor of shape (nstim, ntrials, T, N).
		"""
		nstims = input_params['nstims']
		ntrials = input_params['ntrials']
		loc = input_params['loc']
		grating_resp, all_inputs = self.get_grating_resp(input_params=input_params, return_input=True) # (nstims, ntrials, T+1, N), (nstims, ntrials, T, N)
		opto_inp = self.get_opto_input(ntrials, input_params['T'], loc)
		all_opto_resps = []	
		for i in range(nstims):
			# initial condition
			r_init = torch.zeros((ntrials,self.npop*self.N))
			resp = self.integrate(r_init, input_params['T'], all_inputs[i], opto=opto_inp) # (ntrials, T+1, N)
			all_opto_resps.append(resp)
		all_opto_resps = torch.stack(all_opto_resps, dim=0) # (nstims, ntrials, T+1, N)
		if return_input:
			return grating_resp, all_opto_resps, all_inputs, opto_inp
		else:
			return grating_resp, all_opto_resps # (nstims, ntrials, T+1, N)
		


class LinearModel:
	def __init__(self, params):
		"""
		Initialize the model parameters.
		Args:
		- params (dict): Dictionary of model parameters.
			- N (int): Number of neurons.
			- tau (float): Time constant.
			- f (callable): Activation function.
			- W (torch.Tensor): Coupling matrix (NxN).
			- W_ff (torch.Tensor): Feedforward coupling matrix (NxN).
			- dt (float): Time step for integration.
		"""
		self.N = params['N']
		self.npop = params['npop']
		self.sigma = params['sigma']
		# check W is tensor
	
		if not torch.is_tensor(params['W']):
			self.W = torch.tensor(params['W'])
		else:
			self.W = params['W']
		if not torch.is_tensor(params['W_ff']):
			self.W_ff = torch.tensor(params['W_ff'])
		else:
			self.W_ff = params['W_ff']

		if 'slope' in params:
			self.slope = params['slope']
			self.slope_e = params['slope']
			self.slope_i = params['slope']
		else:
			self.slope = 1
			self.slope_e = 1
			self.slope_i = 1
		if 'slope_e' in params and 'slope_i' in params:
			self.slope_e = params['slope_e']
			self.slope_i = params['slope_i']
	
	def get_W_slope(self):
		"""
		Get the slope-modified coupling matrix W.
		Returns:
		- torch.Tensor: Slope-modified coupling matrix.
		"""
		## check if slope_e and slope_i are defined
		if not hasattr(self, 'slope'):
			# if slope is not defined, use 1
			self.slope = 1
			self.slope_e = 1
			self.slope_i = 1
			W_slope = self.W
		if hasattr(self, 'slope') and not hasattr(self, 'slope_e') and not hasattr(self, 'slope_i'):
			# if slope is defined, use it
			W_slope_e = self.slope * self.W
			W_slope_i = self.slope * self.W
			W_slope = torch.cat((W_slope_e[:self.N,:], W_slope_i[self.N:,:]), dim=0) # (2N x npop*N)
		if hasattr(self, 'slope_e') or hasattr(self, 'slope_i'):
			# if slope_e and slope_i are defined, use them
			W_slope_e = self.slope_e * self.W
			W_slope_i = self.slope_i * self.W
			W_slope = torch.cat((W_slope_e[:self.N,:], W_slope_i[self.N:,:]), dim=0)
		self.W_slope = W_slope
		return W_slope
	
	def get_W_ff_slope(self):
		"""
		Get the slope-modified feedforward coupling matrix W_ff.
		Returns:
		- torch.Tensor: Slope-modified feedforward coupling matrix.
		"""
		if not hasattr(self, 'slope'):
			self.slope = 1
			self.slope_e = 1
			self.slope_i = 1
			W_ff_slope = self.W_ff
		if hasattr(self, 'slope') and not hasattr(self, 'slope_e') and not hasattr(self, 'slope_i'):
			W_ff_slope_e = self.slope * self.W_ff
			W_ff_slope_i = self.slope * self.W_ff
			W_ff_slope = torch.cat((W_ff_slope_e[:self.N,:], W_ff_slope_i[self.N:,:]), dim=0)
		if hasattr(self, 'slope_e') or hasattr(self, 'slope_i'):
			W_ff_slope_e = self.slope_e * self.W_ff
			W_ff_slope_i = self.slope_i * self.W_ff
			W_ff_slope = torch.cat((W_ff_slope_e[:self.N,:], W_ff_slope_i[self.N:,:]), dim=0)
		self.W_ff_slope = W_ff_slope
		return W_ff_slope

	def get_max_frequency(self):
		"""
		Compute the maximum frequency of the recurrent network.
		Returns:
		- float: Maximum frequency of the recurrent network.
		"""
		# freqs, eigvals = freq_distri(self.W)
		# max_eig = np.max(np.real(eigvals))
		# max_k_index = np.argmax(np.real(eigvals))
		# print('max_freq:', freqs[max_k_index])
		_,kmax,wavelength = find_kmax_block(self.W.numpy(), self.N)
		# print('max_freq:', kmax,'wavelength:', wavelength)
		self.max_freq = kmax
		self.wavelength = wavelength
		return kmax
	def get_wavelength(self):
		_,kmax,wavelength = find_kmax_block(self.W.numpy(), self.N)
		self.wavelength = wavelength
		return wavelength

	def shuffle_W(self, shuffle_type='location',fraction=0.2,rng=None):
		"""
		Shuffle the coupling matrix W.
		Args:
		- type (str): Type of shuffling ('location' or 'weight').
		"""
		if rng is None:
			rng = np.random.default_rng()

		W = self.W.numpy()

		if shuffle_type == 'location':
			if fraction == 1.0:
				W_shuffled = W.copy()
				perm_pairs = rng.permutation(self.N)
				full_perm = np.concatenate([perm_pairs, perm_pairs + self.N])
				W_shuffled = W[full_perm][:, full_perm]
			else:
				N = self.N
				# --- Number of locations to shuffle ---
				k = int(np.round(fraction * N))     # how many E–I pairs to shuffle
				# --- Choose which locations to shuffle ---
				shuffle_idx = rng.choice(N, size=k, replace=False)
				# --- Create a permutation only for those chosen locations ---
				shuffled_subperm = shuffle_idx[rng.permutation(k)]
				# --- Initialize full identity permutation ---
				full_loc_perm = np.arange(N)
				# --- Replace only the chosen indices with the shuffled mapping ---
				full_loc_perm[shuffle_idx] = shuffled_subperm
				# --- Build excitatory + inhibitory permutation ---
				full_perm = np.concatenate([full_loc_perm, full_loc_perm + N])
				# --- Apply partial shuffle ---
				W_shuffled = W[full_perm][:, full_perm]
			
		elif shuffle_type == 'weight_swap':
			W = self.W.numpy().copy()
			idx = np.arange(self.N)
			# distance matrix on a ring: d_ij = min(|i-j|, N-|i-j|)
			diff = np.abs(idx[:, None] - idx[None, :])
			dist = np.minimum(diff, self.N - diff)
			# local vs non-local masks
			within_mask = dist <= self.sigma
			outside_mask = ~within_mask
			# usually we don't want to swap self-connections 
			np.fill_diagonal(within_mask, False)
			np.fill_diagonal(outside_mask, False)
			for i in range(self.N):
				local_targets = np.where(within_mask[i])[0]
				nonlocal_targets = np.where(outside_mask[i])[0]

				if local_targets.size == 0 or nonlocal_targets.size == 0:
					continue

				# number of local weights to swap
				n_swaps = int(np.floor(fraction * local_targets.size))
				if n_swaps == 0:
					continue

				# choose which local and non-local synapses to swap
				chosen_local = rng.choice(local_targets, size=n_swaps, replace=False)
				chosen_nonlocal = rng.choice(nonlocal_targets, size=n_swaps, replace=False)

				# swap weights in row i
				wee_row = W[i,:self.N]
				tmp = wee_row[chosen_local].copy()
				wee_row[chosen_local] = wee_row[chosen_nonlocal]
				wee_row[chosen_nonlocal] = tmp

				wii_row = W[self.N + i, self.N:]
				tmp = wii_row[chosen_local].copy()
				wii_row[chosen_local] = wii_row[chosen_nonlocal]
				wii_row[chosen_nonlocal] = tmp

				wei_row = W[i, self.N:]
				tmp = wei_row[chosen_local].copy()
				wei_row[chosen_local] = wei_row[chosen_nonlocal]
				wei_row[chosen_nonlocal] = tmp

				wie_row = W[self.N + i, :self.N]
				tmp = wie_row[chosen_local].copy()
				wie_row[chosen_local] = wie_row[chosen_nonlocal]
				wie_row[chosen_nonlocal] = tmp
			W_shuffled = W
		else:
			raise ValueError(f"Unknown shuffle type: {shuffle_type}")
		self.W = torch.tensor(W_shuffled, dtype=torch.float32)

	def get_max_eigenvalue(self):
		"""
		Compute the maximum eigenvalue of the coupling matrix W.
		Returns:
		- float: Maximum eigenvalue of W.
		"""
		eigenvalues = torch.linalg.eigvals(self.W).detach().numpy()  # Get eigenvalues of W
		W_slope = self.get_W_slope()
		eigenvalues = torch.linalg.eigvals(W_slope).detach().numpy()  # Get eigenvalues of W
		# get max real part of eigenvalues
		max_eigenvalue = np.max(np.real(eigenvalues))
		# print('Max eigenvalue (including slopes):', max_eigenvalue, flush=True)
		self.sbound_eff = max_eigenvalue
		return max_eigenvalue
	
	def get_opto_slope(self, opto):
		"""
		Get the slope-modified optogenetic input.
		Args:
		- opto (torch.Tensor): Optogenetic input tensor of shape (N, ntrials).
		Returns:
		- torch.Tensor: Slope-modified optogenetic input.
		"""
		if not torch.is_tensor(opto):
			opto = torch.tensor(opto, dtype=torch.float32)
		
		if hasattr(self, 'slope'):
			opto_slope = self.slope * opto
			return opto_slope
		elif hasattr(self, 'slope_e') and hasattr(self, 'slope_i'):
			opto_slope_e = self.slope_e * opto[:self.N,:]
			opto_slope_i = self.slope_i * opto[self.N:,:]
			opto_slope = torch.cat((opto_slope_e, opto_slope_i), dim=0)
			return opto_slope
		else:
			return opto
	
	def get_fp(self,inp,opto=None,return_r=False):
		"""
		Compute the fixed point of the model given the input.
		Args:
		- inp (torch.Tensor): Input tensor of shape (N, ntrials).
		Returns:
		- torch.Tensor: Fixed point of the model. (N x ntrials)
		"""
		if not torch.is_tensor(inp):
			inp = torch.tensor(inp,dtype=torch.float32) #(N x ntrials)
		## determinant of (I - W)
		sbound = self.get_max_eigenvalue()

		## Check stability, set to nan if not stable
		if sbound >= 1:
			# print('Warning:  max lambda(w)>=1, not stable')
			fp_r = torch.full_like(inp, float('nan'))
			if return_r:
				return fp_r, sbound
			else:
				return fp_r

		if opto is not None:
			opto = torch.tensor(opto,dtype=torch.float32)
			W_slope = self.get_W_slope()
			W_ff_slope = self.get_W_ff_slope()
			opto_slope = self.get_opto_slope(opto)  # (N x ntrials)
			fp_r = torch.linalg.inv(torch.eye(self.N*self.npop) - W_slope) @ (W_ff_slope @ inp + opto_slope)
			# fp_r = torch.linalg.inv(torch.eye(self.N*self.npop) - self.slope*self.W) @ ( self.slope*(self.W_ff @ inp) + opto)
		else:
			# fp_r = np.linalg.inv(torch.eye(self.N) - self.W) @ (self.W_ff @ inp)
			W_slope = self.get_W_slope()  # (2N x npop*N)
			W_ff_slope = self.get_W_ff_slope()
			fp_r = torch.linalg.inv(torch.eye(self.N*self.npop) - W_slope) @ (W_ff_slope @ inp)
			# fp_r = torch.linalg.inv(torch.eye(self.N*self.npop) - self.slope*self.W) @ (self.slope*self.W_ff @ inp) #(self.N x ntrials)
		if return_r:
			return fp_r, sbound
		else:
			return fp_r # (N x ntrials)

	def get_influence(self, loc,pop='E', intensity=1):
		"""
		Compute the influence of optogenetic stimulation at a given location.
		Args:
		- loc (int): Location of the optogenetic stimulation.
		- intensity (float): Intensity of the optogenetic stimulation.
		- pop (str): Population to stimulate ('E' or 'I').
		Returns:
		- torch.Tensor: Influence vector of shape (N,).
		"""
		opto = torch.zeros((self.npop*self.N,1))
		if self.npop == 1:
			opto[loc] = intensity
		elif self.npop == 2:
			if pop == 'E':
				opto[loc] = intensity
			elif pop == 'I':
				opto[self.N+loc] = intensity
			else:
				raise ValueError(f"Unknown population: {pop}")
		
		influence = self.get_fp(torch.zeros((self.npop*self.N,1)), opto=opto) #(N x 1)
		# subtract input 
		i = torch.zeros((self.npop*self.N,1))
		if self.npop == 1:
			i[loc] = intensity
		elif self.npop == 2:
			if pop == 'E':
				i[loc] = intensity
			elif pop == 'I':
				i[self.N+loc] = intensity
		influence = influence - i
		return influence.squeeze() #(N,)

	def get_influence_distance(self, loc, pop='E', intensity=1):
		"""
		Compute the influence of optogenetic stimulation as a function of distance from the stimulation site.
		Args:
		- loc (int): Location of the optogenetic stimulation.
		- intensity (float): Intensity of the optogenetic stimulation.
		- pop (str): Population to stimulate ('E' or 'I').
		Returns:
		- distances (torch.Tensor): Distances from the stimulation site.
		- influences (torch.Tensor): Influence values at each distance.
		"""
		influence = self.get_influence(loc, pop=pop, intensity=intensity) #(npop*N,)
		# compute distance from loc, with periodic boundary conditions
		distances = torch.arange(self.N)
		distances = torch.abs(distances - loc)
		distances = torch.minimum(distances, self.N - distances)  # Apply periodic boundary conditions
		# sort by distance
		sorted_distances, indices = torch.sort(distances)
		influence_e = influence[:self.N]
		influence_i = influence[self.N:]
		sorted_influences_e = influence_e[indices]
		sorted_influences_i = influence_i[indices] 
		return sorted_distances, sorted_influences_e, sorted_influences_i

	def assign_ncorr(self, ncorr_type='EE',sigma=None):
		"""
		Assign the type of noise correlation to compute.
		Args:
		- ncorr_type (str): how ncorr is assigned.
		"""
		## ncorr is the same as EE connection, but scaled from 0 to 1
		if ncorr_type == 'EE':
			ncorr_ee = self.W[:self.N, :self.N]/torch.max(torch.abs(self.W[:self.N, :self.N]))
			## repeat for other populations
			if self.npop == 2:
				ncorr = torch.block_diag(ncorr_ee, ncorr_ee)
				# off-diagonal blocks
				ncorr[:self.N, self.N:] = ncorr_ee
				ncorr[self.N:, :self.N] = ncorr_ee
			else:
				ncorr = ncorr_ee
		if ncorr_type == 'Gaussian':
			# Gaussian decay based on distance
			idx = torch.arange(self.N)
			diff = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
			dist = torch.minimum(diff, self.N - diff)  # (N x N)
			if sigma is None:
				decay = torch.exp(-0.5 * (dist / self.sigma)**2)
			else:
				decay = torch.exp(-0.5 * (dist / sigma)**2)
			ncorr_ee = decay / torch.max(decay)
			## repeat for other populations
			if self.npop == 2:
				ncorr = torch.block_diag(ncorr_ee, ncorr_ee)
				# off-diagonal blocks
				ncorr[:self.N, self.N:] = ncorr_ee
				ncorr[self.N:, :self.N] = ncorr_ee
			else:
				ncorr = ncorr_ee
		if ncorr_type == 'exp':
			# exponential decay based on distance
			idx = torch.arange(self.N)
			diff = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
			dist = torch.minimum(diff, self.N - diff)  # (N x N)
			if sigma is None:
				decay = torch.exp(-dist / self.sigma)
			else:
				decay = torch.exp(-dist / sigma)	
			ncorr_ee = decay / torch.max(decay)
			## repeat for other populations
			if self.npop == 2:
				ncorr = torch.block_diag(ncorr_ee, ncorr_ee)
				# off-diagonal blocks
				ncorr[:self.N, self.N:] = ncorr_ee
				ncorr[self.N:, :self.N] = ncorr_ee
			else:
				ncorr = ncorr_ee
		if ncorr_type == 'linear':
			# linear decay based on distance
			idx = torch.arange(self.N)
			diff = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
			dist = torch.minimum(diff, self.N - diff)  # (N x N)
			decay = torch.clamp(1 - dist / (self.N / 2), min=0)
			ncorr_ee = decay / torch.max(decay)
			## repeat for other populations
			if self.npop == 2:
				ncorr = torch.block_diag(ncorr_ee, ncorr_ee)
				# off-diagonal blocks
				ncorr[:self.N, self.N:] = ncorr_ee
				ncorr[self.N:, :self.N] = ncorr_ee
			else:
				ncorr = ncorr_ee
		if ncorr_type == 'analytic_ncorr':
			# (I-W)^-1 * (I-W.T)^-1
			ncorr_analytic = torch.linalg.inv(torch.eye(self.npop*self.N) - self.W) @ torch.linalg.inv(torch.eye(self.npop*self.N) - self.W.T)
			# remove diagonal
			ncorr_analytic = ncorr_analytic.masked_fill(torch.eye(ncorr_analytic.size(0), dtype=torch.bool), 0)
			# normalize to min 0 max 1
			ncorr_analytic = (ncorr_analytic - torch.min(ncorr_analytic)) / (torch.max(ncorr_analytic) - torch.min(ncorr_analytic))
			# add back diagonal
			ncorr = ncorr_analytic.masked_fill(torch.eye(ncorr_analytic.size(0), dtype=torch.bool), 1)
		if ncorr_type == 'EI_diff':
			ncorr_ee = self.W[:self.N, :self.N]+ self.W[self.N:, self.N:]
			if self.npop == 2:
				ncorr = torch.block_diag(ncorr_ee, ncorr_ee)
				# off-diagonal blocks
				ncorr[:self.N, self.N:] = ncorr_ee
				ncorr[self.N:, :self.N] = ncorr_ee
		if ncorr_type == 'MH':
			"""
			Build a (2N x 2N) translationally invariant ncorr matrix.
			Column j is the DoG kernel centered at j (and same for populations).
			"""

			# ---- 1. Compute 1D circular distances ----
			x = np.arange(self.N)
			d = np.abs(x[:, None] - x[None, :])
			d = np.minimum(d, self.N - d)  # circular distances

			# ---- 2. Compute the base 1D DoG kernel for one location ----
			kernel = mh_1d(d[0], sigma_e=10, sigma_i=15, w_e=1, w_i=1)  
			# shape (N,)
			# ---- 3. Build the N x N Toeplitz/circulant kernel (translational invariance) ----
			kernel_mat = np.zeros((self.N, self.N))
			for j in range(self.N):
				kernel_mat[:, j] = np.roll(kernel, j)
			EE = kernel_mat.copy()
			EI = kernel_mat.copy()
			IE = kernel_mat.copy()
			II = kernel_mat.copy()

			# Final 2N x 2N matrix
			ncorr = np.block([
				[EE, EI],
				[IE, II],
			])

			ncorr = torch.from_numpy(ncorr)
			ncorr_min = ncorr.min()
			ncorr_max = ncorr.max()
			ncorr = (ncorr - ncorr_min) / (ncorr_max - ncorr_min)
		
		if ncorr_type == 'sigmoid':
			# sigmoid decay based on distance
			idx = torch.arange(self.N)
			diff = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
			dist = torch.minimum(diff, self.N - diff)  # (N x N)
			turning_point = 20
			slope = 15
			decay = torch.sigmoid(slope*(turning_point - dist) / (turning_point))
			ncorr_ee = decay / torch.max(decay)
			## repeat for other populations
			if self.npop == 2:
				ncorr = torch.block_diag(ncorr_ee, ncorr_ee)
				# off-diagonal blocks
				ncorr[:self.N, self.N:] = ncorr_ee
				ncorr[self.N:, :self.N] = ncorr_ee
			else:
				ncorr = ncorr_ee
		self.ncorr_assigned = ncorr
		return ncorr
	
	def ncorr_distance(self):
		"""
		Compute the noise correlation as a function of distance.
		Returns:
		- distances (torch.Tensor): Distances from the reference neuron.
		- ncorrs (torch.Tensor): Noise correlation values at each distance.
		"""
		# ncorr = self.assign_ncorr(ncorr_type='EE') #(npop*N x npop*N)
		try:
			ncorr = self.ncorr_assigned
		except AttributeError:
			print('ncorr not assigned yet, assigning now using EE type')
			ncorr = self.assign_ncorr(ncorr_type='EE')
		# compute distance matrix on a ring
		idx = torch.arange(self.N)
		diff = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
		dist = torch.minimum(diff, self.N - diff)  # (N x N)
		distances = torch.arange(self.N)
		ncorr_e = ncorr[:self.N, :self.N]
		ncorr_i = ncorr[self.N:, self.N:]
		ncorrs_e = []
		ncorrs_i = []
		for d in distances:
			mask = dist == d
			ncorrs_e.append(torch.mean(ncorr_e[mask]).item())
			ncorrs_i.append(torch.mean(ncorr_i[mask]).item())
		ncorrs_e = torch.tensor(ncorrs_e)
		ncorrs_i = torch.tensor(ncorrs_i)
		return distances, ncorrs_e, ncorrs_i

	def get_params(self):
		return vars(self)

	def get_grating_input(self,ori,ntrials,contrast=1):
		"""
		Compute the grating input for the given orientation and number of trials.
		Args:
		- ori (float): Orientation of the grating.
		- ntrials (int): Number of trials.
		Returns:
		- torch.Tensor: Grating input tensor of shape (N, ntrials).
		"""
		## get max_frequency of the recurrent network
		max_freq = self.get_max_frequency()
		## generate grating inputs
		grating_input = torch.zeros((self.npop*self.N, ntrials))
		
		x = torch.linspace(0, self.N, self.N)

		# sine input using kmax
		grating = contrast * torch.sin(max_freq * x + ori) # (N, 1)

		## repeat if 2 populations
		if self.npop == 2:
			grating = torch.cat((grating, grating), dim=0)  # Repeat for second population
		# repeat ntrials
		grating_input = grating.unsqueeze(1).repeat(1, ntrials) # (N, ntrials)
		return grating_input

	def get_noisy_input(self,ntrials,noise_level=0.1):
		"""
		Generate 0 mean Gaussian noise input for the given number of trials.
		Args:
		- ntrials (int): Number of trials.
		- noise_level (float): Standard deviation of the Gaussian noise.
		Returns:
		- torch.Tensor: Noisy input tensor of shape (N, ntrials).
		"""
		noise = torch.randn((self.N*self.npop, ntrials)) * noise_level
		return noise

	def get_structured_noise_input(self,ntrials,structure='flat',noise_level=0.1,structure_params=None):
		"""
		Generate structured noise input for the given number of trials.
		Args:
		- ntrials (int): Number of trials.
		- noise_level (float): Standard deviation of the Gaussian noise.
		Returns:
		- torch.Tensor: Structured noise input tensor of shape (N, ntrials).
		"""
		## same noise for all neurons, different between trials
		if structure == 'flat':
			noise = torch.randn((1,ntrials)) * noise_level #(1,ntrials)
			noise = noise.repeat(self.N*self.npop, 1) #(N x ntrials)
		elif structure == 'gaussian':
			base_mean = structure_params['base_mean']
			base_std = structure_params['base_std']
			# neuron axis (0 ... N)
			x = torch.arange(self.N).float().unsqueeze(1)  # (N,1)
			profiles = []
			for t in range(ntrials):
				# jitter mean and std around base values
				mu_t = base_mean +  torch.randn(1)
				sigma_t = base_std + torch.randn(1)
				amp_t = torch.abs(torch.randn(1))
				# Gaussian profile across neurons
				profile = amp_t * torch.exp(-0.5 * ((x - mu_t) / sigma_t)**2)
				profiles.append(profile)
			noise_e = torch.cat(profiles, dim=1)  # (N, ntrials)
			## repeat for I to get full noise
			noise = noise_e.repeat(2, 1) * noise_level  # (2N, ntrials)

		else:
			raise ValueError(f"Unknown structure: {structure}")
		return noise

	def get_grating_resp(self,input_params,return_input=False):
		"""
		Compute the grating response for the given number of stimuli and trials.
		Args:
		- input_params (dict): Dictionary containing input parameters.
		Returns:
		- torch.Tensor: Grating response tensor of shape (nstims, ntrials, N).
		"""
		nstims = input_params['nstims']
		ntrials = input_params['ntrials']
		if 'contrast' not in input_params.keys():
			contrast = 1
		else:
			contrast = input_params['contrast']
		if 'base_line' not in input_params.keys():
			base_line = 0
		else:
			base_line = input_params['base_line']
		if 'structured_noise_level' not in input_params.keys():
			structured_noise_level = 0.0
		else:
			structured_noise_level = input_params['structured_noise_level']
		if 'independent_noise_level' not in input_params.keys():
			independent_noise_level = 0.0
		else:
			independent_noise_level = input_params['independent_noise_level']
		if 'shared_noise_level' not in input_params.keys():
			shared_noise_level = 0.0
		else:
			shared_noise_level = input_params['shared_noise_level']

		## equally spaced orientations
		stims = torch.linspace(0, np.pi, nstims)
		grating_responses = []
		all_inputs = []
		for ori in stims:
			grating_inp = self.get_grating_input(ori, ntrials, contrast=contrast) # (N, ntrials)
			independent_noise = self.get_noisy_input(ntrials, noise_level=independent_noise_level)
			shared_noise = self.get_structured_noise_input(ntrials, structure='flat', noise_level=shared_noise_level)
			base_mean = np.argmax(grating_inp.mean(dim=1).detach().numpy())
			structured_noise = self.get_structured_noise_input(ntrials, structure='gaussian', noise_level=structured_noise_level, structure_params={'base_mean': base_mean, 'base_std': 10})
			total_input = grating_inp + independent_noise + shared_noise + structured_noise # (N, ntrials)
			grating_resp = self.get_fp(total_input) # (N, ntrials)
			grating_responses.append(grating_resp.T) # (ntrials, N)
			all_inputs.append(total_input.T) # (ntrials, N)
		if return_input:
			return torch.stack(grating_responses), torch.stack(all_inputs)
		else:
			return torch.stack(grating_responses) # (nstims, ntrials, N)
		
	def get_preferred_oris(self, noris=8):
		"""
		Compute the preferred orientations of the neurons.
		Args:
		- noris (int): Number of orientations to test.
		Returns:
		- torch.Tensor: Preferred orientations of the neurons.
		"""
		# get evenly spaced orientations not include pi itself
		stims = torch.tensor(np.linspace(0, np.pi, noris, endpoint=False)) # (nstims,)
		grating_aprams = {'nstims': noris,
						'ntrials': 50,
						'contrast': 1,
						'base_line': 0,
						'structured_noise_level': 0.0,
						'independent_noise_level': 0.0,
						'shared_noise_level': 0.0,
						}
		grating_resp = self.get_grating_resp(grating_aprams) # (nstims, ntrials, N)
		# average over trials
		mean_resp = grating_resp.mean(dim=1) # (nstims, N)
		# get preferred orientation for each neuron
		preferred_oris = stims[mean_resp.argmax(dim=0)] # (N,)
		# convert to degrees
		preferred_oris = preferred_oris * 180 / np.pi
		return preferred_oris
	
	def get_cotuned_neurons(self, loc, thres=22.5, noris=8):
		"""
		Get the indices of neurons that are cotuned to the neuron at given location.
		Args:
		- loc (int): Location of the reference neuron.
		- thres (float): Threshold for cotuning in degrees.
		- noris (int): Number of orientations to test.
		Returns:
		- list: Indices of cotuned neurons.
		"""
		preferred_oris = self.get_preferred_oris(noris=noris) # (N,)
		ref_ori = preferred_oris[loc]
		# compute angle difference
		angle_diff = torch.abs(preferred_oris - ref_ori)
		# wrap around pi
		angle_diff = torch.minimum(angle_diff, 180 - angle_diff)
		# convert to degrees
		cotuned_neurons = torch.where(angle_diff < thres)[0].tolist()
		return np.array(cotuned_neurons)
		
	
	def get_opto_input(self,loc,ntrials,opto_strength=1,pop='E'):
		"""
		Compute the optogenetic input for the given location and number of trials.
		Args:
		- loc (int): Location of the stimulus.
		- ntrials (int): Number of trials.
		- opto_strength (float): Strength of the optogenetic stimulation.
		- pop (str): Population to apply the stimulation ('E' or 'I').
		Returns:
		- torch.Tensor: Optogenetic input tensor of shape (N, ntrials).
		"""
		opto_input = torch.zeros((self.N*self.npop, ntrials))
		if pop == 'E':
			opto_input[loc, :] = opto_strength
		elif pop == 'I':
			opto_input[loc + self.N, :] = opto_strength
		return opto_input 

	def get_grating_and_opto_resp(self,input_params,return_input=False):
		"""
		Compute the grating and optogenetic response for the given parameters.
		Args:
		- nstim (int): Number of stimuli.
		- ntrials (int): Number of trials.
		- loc (int): Location of the stimulus.
		- contrast (float): Contrast of the stimulus.
		- base_line (float): Baseline firing rate.
		- noise_level (float): Level of noise in the stimulus.
		- neuron_noise (float): Level of noise in the neurons.
		Returns:
		- torch.Tensor: Grating and optogenetic response tensor of shape (nstim, ntrials, N).
		"""
		nstims = input_params['nstims']
		ntrials = input_params['ntrials']
		loc = input_params['loc']
		grating_resp, all_inputs = self.get_grating_resp(input_params=input_params, return_input=True)
		opto_inp = self.get_opto_input(loc, ntrials)
		all_opto_resps = []
		for i in range(nstims):
			opto_resp = self.get_fp(all_inputs[i].T, opto=opto_inp) # (N x ntrials)
			all_opto_resps.append(opto_resp.T)
		if return_input:
			return grating_resp, torch.stack(all_opto_resps), all_inputs
		else:
			return grating_resp, torch.stack(all_opto_resps)

	def get_determinant(self):
		"""
		Compute the determinant of (I - W).
		Returns:
		- float: Determinant of (I - W).
		"""
		det = torch.det(torch.eye(self.N*self.npop) - self.W)
		return det

	

def generate_model(params,linear=True):
	"""
	Generate a model with the given parameters.
	Args:
	- params (dict): Dictionary of model parameters.
	Returns:
	RingModel: An instance of the RingModel class.
	or
	LinearModel: An instance of the LinearModel class.
	"""
	
	if linear:
		model = LinearModel(params)
	else:
		model = RingModel(params)
	return model


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


def test_nonlin():
	# Parameters
	N = 5  # Number of neurons
	tau = 0.01  # Time constant
	f = torch.sigmoid  # Activation function
	W = torch.randn(N, N) * 0.1  # Random coupling matrix

	dt = 0.01  # Integration time step
	T = 1000  # Number of time steps
	h = torch.tensor(np.tile(torch.randn(N),(T,1)))  # External input
	print(h.shape)

	# Initial firing rates
	r_init = torch.zeros(N)

	network_params = {
		'N':N,
		'tau':tau,
		'f':f,
		'W':W,
		'dt':dt
	}

	# Create and simulate the model
	model = RingModel(network_params)
	r_history = model.integrate(r_init, T,h)

	# Visualization
	time = torch.arange(0, T + 1) * dt
	for i in range(N):
		plt.plot(time, r_history[:, i], label=f"Neuron {i+1}")
	plt.xlabel("Time")
	plt.ylabel("Firing rate (r)")
	plt.title("Dynamics of Coupled Neurons")
	plt.legend()
	plt.show()

	


if __name__ == '__main__':
	save_path  = '/Users/fionakong/Downloads/kaschube-lab/Influence_mapping/sim_out/'
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	N = 100
	sigma_m = 5
	sigma_w = 10

	w = ff_connections(N,fftype='Gaussian',params={'sigma':sigma_w})
	m = recurrent_connections(N,rtype='Gaussian',params={'sigma': sigma_m,})
	
	directions = np.linspace(0,2*np.pi-0.1,4)
	contrast = 1
	ntrials = 10
	opto_locs = [10,20,30,40,50,60,70,80,90]
	

	# no opto stimulation
	stim_resp = {
		'stim':np.zeros((len(directions),2*ntrials,1)),
		'resp':np.zeros((len(directions),2*ntrials,N))
	}
	for i,d in enumerate(directions):
		for j in range(2*ntrials):
			inp = get_input(N, 'grating', {'direction':d,'contrast':contrast})
			fp_r = get_fp(w,m,inp)
			stim_resp['resp'][i,j,:] = fp_r.flatten().T
			stim_resp['stim'][i,j,:] = d


	np.save(save_path + 'stim_responses_visual.npy',stim_resp)
	del stim_resp

	

	# opto stimulation
	opto_resp = {
		'stim':np.zeros((len(opto_locs),len(directions),ntrials,1)),
		'opto':np.zeros((len(opto_locs),len(directions),ntrials,1)),
		'resp':np.zeros((len(opto_locs),len(directions),ntrials,N))
	}
	# responses_opto = np.zeros((len(directions),ntrials,len(opto_locs),N))
	for i,loc in enumerate(opto_locs):
		for j,d in enumerate(directions):
			for k in range(ntrials):
				inp = get_input(N, 'grating', {'direction':d,'contrast':contrast})
				opto = get_input(N, 'opto', {'location':loc})
				fp_r = get_fp(w,m,inp,opto)
				opto_resp['resp'][i,j,k,:] = fp_r.flatten().T
				opto_resp['stim'][i,j,k,:] = d
				opto_resp['opto'][i,j,k,:] = loc

	np.save(save_path + 'stim_responses_opto.npy',opto_resp)
	del opto_resp

				

		
	

	








