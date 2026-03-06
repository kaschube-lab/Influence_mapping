import math
from math import dist
import numpy as np
import torch
import os,sys
import h5py
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import linregress, wilcoxon
import pickle

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from Model.ring import generate_model,get_input
from Model.model_params import default_params
from Model.ring_analysis_helper import influence_vs_distance_1loc
from Model.simulate_helpers import autoval_distr, linear_grating_trials

def infl_vs_ncorr(infl,ncorr,loc,nbins=6,distance=None):
	"""
	Calculates the average influence for noise correlation bin.
	
	Parameters:
	- infl: Influence matrix (N)
	- ncorr: Noise correlation matrix (N x 1)
	- loc: Location of the target neuron (int)
	Returns:
	- avg_infl: Average influence excluding the target neuron
	- avg_ncorr: Average noise correlation excluding the target neuron
	"""
	## if infl and ncorr are tensors, convert to numpy arrays
	if isinstance(infl, torch.Tensor):
		infl = infl.detach().numpy()
	if isinstance(ncorr, torch.Tensor):
		ncorr = ncorr.detach().numpy()

	# Exclude the target neuron
	infl = np.delete(infl, loc)
	ncorr = np.delete(ncorr, loc)

	# if distance is provided, only consider neurons within the distance
	if distance is not None:
		# Calculate the distance from the target neuron
		distances = np.abs(np.arange(len(ncorr)) - loc)
		# Exclude neurons that are farther than the specified distance
		mask = distances <= distance
		infl = infl[mask]
		ncorr = ncorr[mask]

	# bin noise correlation using min and max values
	min_ncorr = np.min(ncorr)
	max_ncorr = np.max(ncorr)
	if min_ncorr == max_ncorr:
		# If all values are the same, return zeros
		return np.zeros(1), np.zeros(1)
	# Create bins for noise correlation
	# Using 6 bins for better resolution
	# Adjust the edges to avoid division by zero
	# and to ensure the edges are within the range of noise correlation values
	edges = np.linspace(min_ncorr, max_ncorr, nbins+1)
	binned_ncorr = np.digitize(ncorr, edges) - 1  # -1 to make it zero-indexed
	
	
	
	avg_infl = np.zeros(len(edges)-1)
	avg_ncorr = np.zeros(len(edges)-1)
	for i in range(len(edges)-1):
		# Get indices for the current bin
		indices = np.where(binned_ncorr == i)[0]
		# Calculate average influence and noise correlation for the bin
		# If there are no indices in the bin, set to NaN
		# to avoid division by zero
		# and later replace NaN with 0
		# If there are indices in the bin, calculate the mean
		if len(indices) > 0:
			avg_infl[i] = np.mean(infl[indices])
			avg_ncorr[i] = np.mean(ncorr[indices])
		else:
			avg_infl[i] = np.nan
			# use the center of the bin as the average noise correlation
			avg_ncorr[i] = (edges[i] + edges[i+1]) / 2
	# Return the average influence and noise correlation
	avg_infl = np.nan_to_num(avg_infl, nan=0.0)
	avg_ncorr = np.nan_to_num(avg_ncorr, nan=0.0)
	avg_ncorr = np.clip(avg_ncorr, -1, 1)  # Ensure values are within [-1, 1]

	return avg_infl, avg_ncorr

def get_rec_connectivity(keyword,other_kw=None):
	"""
	Returns the recurrent connectivity matrix W from the default model parameters.
	"""
	# Load the default parameters
	N=100
	if other_kw == 'cross-pop':
		customized_params = {
			'N': N,
			'npop': 2,
			'sigma': 8,
			'r': None,
			'wee': 1.001,
			'wie': 3.5,
			'wei': 3.5,
			'wii': 2.5,
			'sigma_ee': 8,
			'sigma_ie': 8,
			'sigma_ii': 8*1.5,
			'sigma_ei': 8*1.5,
		}
	elif other_kw == 'mh':
		customized_params = {
			'N': N,
			'npop': 2,
			'sigma': 8,
			'r': 0.9,
			'wee': 2.5,
			'wie': 2.4,
			'wei': 2.4,
			'wii': 2.5,
			'sigma_ie': 8,
			'sigma_ii': 8*1.5,
			'sigma_ei': 8*1.5,
		}
	else:
		customized_params = None
	params = default_params(keyword, customized_params,unit_ff=True)
	
	
	# Return the recurrent connectivity matrix W
	w = params['W'].cpu().detach().numpy()
	# valsM,vecsM = np.linalg.eig(w)
	# Freq,Autoval=autoval_distr(w)
	# R=np.real(np.max(Autoval) )
	# kM=Freq[np.argmax(Autoval) ]
	# if kM>0:
	# 	Lambda=round((N/(2*np.pi*(kM))))
	# else :
	# 	Lambda=N//2
	model = generate_model(params,linear=True)
	Lambda = model.get_wavelength() 
	return w,Lambda

def get_influence_dist(keyword,contrast=1,population='E'):
	"""
	Returns the influence matrix for the specified keyword and input type.
	"""
	# Load the default parameters
	params = default_params(keyword,unit_ff=True)
	params.update({'slope':contrast})
	freqs, eigvals = autoval_distr(params['W'])
	max_eig = np.max(np.real(eigvals))
	max_k_index = np.argmax(np.real(eigvals))
	k_max = freqs[max_k_index]
	print(f"Maximum frequency ': {k_max:.2f} Hz")
	
	# Generate the model
	model = generate_model(params,linear=True)

	# Get the input for the model
	N = params['N']
	npop = params['npop']
	spont_params = {
		'npop':npop
	}
	inp = get_input(N,'spont',spont_params)

	# get spont response
	r = model.get_fp(inp)


	opto_params = {
	'N':N,
	'npop':npop,
	'pop':population,
	'location':20,
	'opto_strength':1
	}
	opto = get_input(N, 'opto', opto_params)

	# get opto response
	r_opto = model.get_fp(inp,opto)

	# get influence
	infl = r_opto - r

	infl_dict = influence_vs_distance_1loc(infl,20)

	return infl_dict


def compute_params_scan():
	Wee=np.arange(0.1,2.7,0.1)
	Wei=np.arange(0.1,3.2,0.1)
	wii = 2.5
	sigma = 10
	inh_a = 1.5
	avg_range = int(1.6*sigma*inh_a)

	# avg_infl_all= np.empty((len(Wei),len(Wee)),dtype=object)
	# stability_all = np.empty((len(Wei),len(Wee)),dtype=object)
	avg_infl_all= np.empty((len(Wei),len(Wee)),dtype=np.float32)
	stability_all = np.empty((len(Wei),len(Wee)),dtype=np.float32) 
	for i,wei in enumerate(Wei):
		for j,wee in enumerate(Wee):
			customized_params = {
				'wee' : wee,
				'wei' : wei,
				'wie' : wei,
				'wii' : wii,
				'sigma' : sigma,
				'sigma_ie' : sigma,
				'sigma_ei' : sigma*inh_a,
				'sigma_ii' : sigma*inh_a
			}
			params = default_params('custmized',customized_params,unit_ff=True)
			model = generate_model(params,linear=True)
			max_eig = model.get_max_eigenvalue()
			dist,infl_e,infl_i = model.get_influence_distance(loc=20,pop='E')
			dist = dist.cpu().detach().numpy()
			infl_e = infl_e.cpu().detach().numpy()
			avg_infl = infl_e[dist < avg_range].mean()
			avg_infl_all[i,j] = avg_infl
			stability_all[i,j] = max_eig
	
	return avg_infl_all, stability_all

def compute_plausible():
	Wee=np.arange(0.1,2.7,0.1)
	Wei=np.arange(0.1,3.2,0.1)
	wii = 2.5
	sigma = 10
	inh_a = 1.5
	avg_range = int(1.6*sigma*inh_a)
	local_range = math.ceil(int(1.6 * sigma * inh_a) / 4)

	avg_infl_all= np.empty((len(Wei),len(Wee)),dtype=np.float32)
	stability_all = np.empty((len(Wei),len(Wee)),dtype=np.float32) 
	local_infl_all = np.empty((len(Wei),len(Wee)),dtype=np.float32) 
	max_supress_all = np.empty((len(Wei),len(Wee)),dtype=np.float32)  
	for i,wei in enumerate(Wei):
		for j,wee in enumerate(Wee):
			customized_params = {
				'wee' : wee,
				'wei' : wei,
				'wie' : wei,
				'wii' : wii,
				'sigma' : sigma,
				'sigma_ie' : sigma,
				'sigma_ei' : sigma*inh_a,
				'sigma_ii' : sigma*inh_a
			}
			params = default_params('custmized',customized_params,unit_ff=True)
			model = generate_model(params,linear=True)
			max_eig = model.get_max_eigenvalue()
			dist, infl_e, infl_i = model.get_influence_distance(loc=20,pop='E')
			dist = dist.cpu().detach().numpy()
			infl_e = infl_e.cpu().detach().numpy()
			avg_infl = infl_e[dist < avg_range].mean()
			avg_infl_all[i,j] = avg_infl
			stability_all[i,j] = max_eig
			local_infl = np.mean(infl_e[dist < local_range])
			local_infl_all[i,j] = local_infl
			idx = np.argmin(infl_e)      # index of the minimum influence
			dist_at_min = dist[idx] 
			max_supress_all[i,j] = dist_at_min

	return avg_infl_all, stability_all, local_infl_all, max_supress_all

def get_plausible_bool(return_all=False):
	avg_infl, stability, local_infl, max_supress = compute_plausible()
	bool_avg = avg_infl < 0
	bool_local = local_infl < 0
	bool_stable = stability < 1
	bool_max_supress = max_supress > 1
	all_plausible = bool_avg & bool_local & bool_stable & bool_max_supress
	if return_all:
		return all_plausible,bool_avg, bool_local, bool_stable, bool_max_supress
	else:
		return all_plausible,stability

def load_plausible_supfig():
	if 'dldevel' in os.path.expanduser("~"):
		save_path = '/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/analysis_data/figs/final/'
	else:
		save_path = '/Volumes/DEYUE/Downloads/kaschube-lab/Influence_mapping/analysis_data/figs/final/'
	plausible = np.load(save_path + 'all_plausible.npy',allow_pickle=True)
	stable = np.load(save_path + 'stability.npy',allow_pickle=True)
	local = np.load(save_path + 'local_infl.npy',allow_pickle=True)
	max_supress = np.load(save_path + 'max_supress.npy',allow_pickle=True)
	avg_infl = np.load(save_path + 'avg_infl.npy',allow_pickle=True)
	return plausible, stable, local, max_supress, avg_infl

def get_params_scan():
	if 'dldevel' in os.path.expanduser("~"):
		save_path = '/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/analysis_data/model/'
	else:
		save_path = '/Users/fionakong/Downloads/kaschube-lab/Influence_mapping/analysis_data/model/'
	with open(save_path + 'linear_influence_results.pickle', 'rb') as f:
		results = pickle.load(f)
		avg_infl_all = results['avg_infl']
		local_infl_all = results['local_infl']
		sbound_all = results['sbound']
		k_all = results['k']
		wee_ls = results['wee_ls']
		s_ls = results['s_ls']
		r_ls = results['r_ls']
	return avg_infl_all,s_ls,r_ls 

def get_valid_params():
	"""
	Returns the valid parameters for the model.
	"""
	if 'dldevel' in os.path.expanduser("~"):
		save_path = '/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/analysis_data/model/'
	else:
		save_path = '/Users/fionakong/Downloads/kaschube-lab/Influence_mapping/analysis_data/model/'
	with open(save_path + 'linear_influence_results.pickle', 'rb') as f:
		results = pickle.load(f)
		wee_ls = results['wee_ls']
		s_ls = results['s_ls']
		r_ls = results['r_ls']
		avg_infl_all = results['avg_infl']
		local_infl_all = results['local_infl']
		sbound_all = results['sbound']
		k_all = results['k']

	mh_mask = (sbound_all < 1) & (sbound_all>0.6) & (k_all >0.01)
	mask = (avg_infl_all < 0) & (local_infl_all < 0) & (sbound_all < 1) & (k_all >0.01)
	return mh_mask,mask
		
def get_avg_infl():
	"""
	Returns the average influence for the model.
	"""
	if 'dldevel' in os.path.expanduser("~"):
		save_path = '/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/analysis_data/figs/'
	else:
		save_path = '/Users/fionakong/Downloads/kaschube-lab/Influence_mapping/analysis_data/model/'
	avg_infl = np.load(save_path + 'AverageEE_Infl.npy')
	stable = np.load(save_path + 'StableEE.npy')
	# avg_infl = np.load(save_path + 'S1.npy')
	# stable = np.load(save_path + 'stability.npy')
	return avg_infl, stable

def get_plausible():
	"""
	Returns the plausible parameters for the model.
	"""
	if 'dldevel' in os.path.expanduser("~"):
		save_path = '/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/analysis_data/figs/'
	else:
		save_path = '/Users/fionakong/Downloads/kaschube-lab/Influence_mapping/analysis_data/model/'
	plausible = np.load(save_path + 'BiologicallyPlausibleEE_Infl.npy')
	stable = np.load(save_path + 'StableEE.npy')
	# plausible = np.load(save_path + 'S0.npy')
	# stable = np.load(save_path + 'stability.npy')
	return plausible, stable

def compute_infl_gain():
	gain_ls = [0.3,0.5,0.7,0.9,1.1,1.3,1.5]
	infl_gain_ls = []
	for gain in gain_ls:
		params = default_params('eg_cross_pop',unit_ff=True)
		params.update({'slope':gain})
		model = generate_model(params,linear=True)
		# dist,infl,_ = model.get_influence_distance(loc=20,pop='E')#.cpu().detach().numpy()
		infl = model.get_influence(loc=0,pop='E').cpu().detach().numpy()
		infl_gain_ls.append(infl[1:])
	return infl_gain_ls


def get_ncorr_schem():
	params = default_params('eg_cross_pop')
	model = generate_model(params,linear=True)
	ncorr = model.assign_ncorr('EE')
	distn, ncorr_e, _ = model.ncorr_distance()
	return distn, ncorr_e

def get_infl_vs_ncorr():
	# low_slopes = {'slope_e':0.61, 'slope_i':0.32}
	low_slopes = {'slope_e':0.23, 'slope_i':0.32}
	# high_slopes = {'slope_e':1.04, 'slope_i':1.77}
	high_slopes = {'slope_e':0.43, 'slope_i':1.88}
	locs = np.arange(0,100,20)

	params = default_params('eg_cross_pop')
	params.update(low_slopes)
	model = generate_model(params,linear=True)

	params1 = default_params('eg_cross_pop')
	params1.update(high_slopes)
	model1 = generate_model(params1,linear=True)

	avg_infl_all  = []
	avg_infl1_all = []

	for loc in locs:
		# influence vs distance around this location
		dist,  infl_e,  infl_i  = model.get_influence_distance(loc=loc)
		infl = model.get_influence(loc=loc)
		ncorr = model.assign_ncorr('EE')
		dist1, infl_e1, infl_i1 = model1.get_influence_distance(loc=loc)
		infl1 = model1.get_influence(loc=loc)
		ncorr1 = model1.assign_ncorr('EE')

		# influence vs ncorr for this location
		# avg_infl,  avg_ncorr  = infl_vs_ncorr(infl[:100],  ncorr[loc, :100],
		# 									loc, nbins=5, distance=10*1.5*1.6)
		avg_infl,  avg_ncorr  = infl_vs_ncorr(infl[:100],  ncorr[loc, :100],
											loc, nbins=5, distance=28)
		# avg_infl1, avg_ncorr1 = infl_vs_ncorr(infl1[:100], ncorr1[loc, :100],
											# loc, nbins=5, distance=10*1.5*1.6)
		avg_infl1, avg_ncorr1 = infl_vs_ncorr(infl1[:100], ncorr1[loc, :100],
											loc, nbins=5, distance=28)

		avg_infl_all.append(avg_infl)
		avg_infl1_all.append(avg_infl1)

	# stack to arrays: shape (n_locs, n_dist) / (n_locs, n_bins)
	avg_infl_all  = np.vstack(avg_infl_all)
	avg_infl1_all = np.vstack(avg_infl1_all)

	avg_infl_mean   = avg_infl_all.mean(axis=0)
	avg_infl_std    = avg_infl_all.std(axis=0)
	avg_infl1_mean  = avg_infl1_all.mean(axis=0)
	avg_infl1_std   = avg_infl1_all.std(axis=0)

	return avg_ncorr, avg_infl_mean, avg_infl_std,avg_infl1_mean, avg_infl1_std 

def sim_infl_ncorr_slopes():
	kw = 'eg_cross_pop'
	loc_ls = np.arange(0,100,10)  # locations to sample influence and noise correlation
	slope_e_ls = np.linspace(0.01, 2, 20)
	slope_i_ls = np.linspace(0.01, 2.5, 25)
	avg_infl_all = []
	avg_ncorr_all = []
	params = default_params(kw)
	for slope_e in slope_e_ls:
		for slope_i in slope_i_ls:
			params.update({'slope_e':slope_e, 'slope_i':slope_i})
			model = generate_model(params)
			ncorr = model.assign_ncorr('EE')
			avg_infl_locs = []
			avg_ncorr_locs = []
			for loc in loc_ls:
				infl = model.get_influence(loc=loc)
				avg_infl, avg_ncorr = infl_vs_ncorr(infl[:100], ncorr[loc, :100],
													loc, nbins=5, distance=10*1.5*1.6)
				avg_infl_locs.append(avg_infl)
				avg_ncorr_locs.append(avg_ncorr)
			avg_infl_all.append(avg_infl_locs)
			avg_ncorr_all.append(avg_ncorr_locs)
	res = {
		'avg_infl': np.array(avg_infl_all),
		'avg_ncorr': np.array(avg_ncorr_all),
		'slope_e_ls': slope_e_ls,
		'slope_i_ls': slope_i_ls,
		'loc_ls': loc_ls,
		'kw': kw
	}
	print('Done')
	return res

def fit_infl_ncorr_slopes():
	res = sim_infl_ncorr_slopes()

	y_vals = res['avg_infl'].reshape(500, -1)   # y: infl
	x_vals = res['avg_ncorr'].reshape(500, -1)  # x: ncorr

	n_boot = 50  # number of bootstrap repetitions
	slopes = np.full(500, np.nan)      # mean bootstrap slope
	pvals_boot = np.full(500, np.nan)  # bootstrap p-values

	for i in range(500):
		x = x_vals[i]
		y = y_vals[i]
		
		valid = np.isfinite(x) & np.isfinite(y)
		x = x[valid]
		y = y[valid]
		
		if len(x) >= 2:
			boot_slopes = []
			for _ in range(n_boot):
				idx = np.random.randint(0, len(x), len(x))  # sample with replacement
				xb = x[idx]
				yb = y[idx]
				try:
					slope = linregress(xb, yb).slope
					boot_slopes.append(slope)
				except Exception:
					continue
			
			boot_slopes = np.array(boot_slopes)
			stat, p_value = wilcoxon(boot_slopes, alternative='two-sided')
			if p_value < 0.005:
				eg_slopes = boot_slopes
			
			if len(boot_slopes) > 0:
				slopes[i] = np.mean(boot_slopes)
				# Two-tailed bootstrap p-value for slope ≠ 0
				pvals_boot[i] = p_value
	return slopes.reshape(20, 25), res['slope_e_ls'], res['slope_i_ls'], pvals_boot.reshape(20, 25)


def influence_4n(gammaI,gammaE):
	w_EE = 1.1
	w_II = 2.5
	w_EI = 2.89
	w_IE = 2.89
	sigmaE=10
	

	delta = 1
	rI = np.linspace(0.07,1 , 30)
	rE= np.linspace(0, 1, 30)

	R_I, R_E = np.meshgrid(rI, rE)
	R_E, R_I = np.meshgrid(rE, rI)
	a1=1/np.sqrt(2*np.pi*(sigmaE)**2)
	b1=1/np.sqrt(2*np.pi*(sigmaE*1.5)**2)
	r=b1/a1
	b=1/np.sqrt(2*np.pi)
	a=b/r

	numerator1 = delta * (1+(1+gammaI)*R_I*b*w_II )
	denominator1 = 2*((1 - (1+gammaE)*R_E*a*w_EE)*(1+(1+gammaI)*R_I*b*w_II) +(1+gammaI)*(1+gammaE)* R_E * R_I * a*b*w_EI * w_IE)
	numerator2 = delta * (1+(1-gammaI)*R_I*b*w_II )
	denominator2 = 2*((1 - (1-gammaE)*R_E*a*w_EE)*(1+(1-gammaI)*R_I*b*w_II) +(1-gammaI)*(1-gammaE)* R_E * R_I *a*b*w_EI * w_IE)
	primo_termine=numerator1/denominator1
	secondo_termine=numerator2/denominator2
	somma=primo_termine-secondo_termine

	return somma


def load_4n_eg():
	if 'dldevel' in os.path.expanduser("~"):
		save_path = '/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/analysis_data/figs/'
	else:
		save_path = '/Volumes/DEYUE/Downloads/kaschube-lab/Influence_mapping/analysis_data/figs/final/'
	
	eg_low = np.load(save_path + 'NC3.npy')
	eg_high = np.load(save_path + 'NC4.npy')

	return eg_low, eg_high