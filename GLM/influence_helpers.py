"""
This file contains the implementation of the influence function.

Created on: 2024-04-20
Created by: Deyue

"""

import numpy as np
import scipy
import h5py 
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
from tqdm import tqdm
import psutil
import traceback
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ProcessPoolExecutor

from dataset import Dataset
from helper import sort_together,find_significant_thresholds,get_p_values

def get_dir_X(X,dir_id, n_vis):
    ntrial,nstim = X.shape
    x1 = x1 = X[:,:n_vis]
    trial_idx = np.where(x1[:,dir_id] ==1)[0]
	# detele all visual stim terms, (ok because there is only one visual stim, and we have a bias term in the end)
    stim_rest = np.delete(np.arange(nstim),np.arange(n_vis)) 
    x_dir = X[trial_idx,:][:,stim_rest]
    return x_dir

def construct_X_y(filename, bias_flag=True,interaction='none',dir=None,neurophil=None,drift=None,verbose=True,timepoint=None):
	with h5py.File(filename, 'r') as f:
		# print some dataset information
		if verbose:
			print(f.keys())
			print(f['ShareData']['VisStim'])
			print(f['ShareData']['Response_dFF'])
			print(f['ShareData']['Response_inferredSpikes'])
			print(f['ShareData']['photoStim'])
			print(f['ShareData']['UseableNonTargetCells'])
			print(f['ShareData']['ShamLocations'])
			target_cells = f['ShareData']['TargetCells'][()][:,0] # (n_real_stims,)
			print('target cells',target_cells)


		# get the response matrix
		## remove last trial for 31

		if timepoint is not None:
			if timepoint == 'early':
				y = f['ShareData']['peak_ispk_early'][()] 
			elif timepoint == 'mid':
				y = f['ShareData']['peak_ispk_mid'][()] 
			elif timepoint == 'late':
				y = f['ShareData']['peak_ispk_late'][()] 
				
		else:
			y = f['ShareData']['Response_inferredSpikes'][()] # (ntrial,ncell)
		ntrial,ncell = y.shape
		# print('shape of y',y.shape)

		# get the design matrix
		x1 = f['ShareData']['VisStim'][()] # (ntrial,n_v_stim)
		_,n_vstim = x1.shape
		num_trial_per_vstim = np.sum(x1,axis=0)
		print('visual num_trial_per_stim',num_trial_per_vstim)
		x2 = f['ShareData']['photoStim'][()] # (ntrial,n_o_stim)
		num_trial_per_ostim = np.sum(x2,axis=0)
		print('opto num_trial_per_stim',num_trial_per_ostim)
		x3 = f['ShareData']['Sham_photoStim'][()] # (ntrial,n_sham_stim)
		num_trial_per_shamstim = np.sum(x3,axis=0)
		print('sham num_trial_per_stim',num_trial_per_shamstim)
		bias = np.ones((ntrial,1))
		print('debug: interaction',interaction)
		
		if interaction == 'none' or interaction is None:
			X0 = np.concatenate((x1,x2,x3),axis=1)
			if bias_flag is not None:
				X0 = np.concatenate((X0,bias),axis=1)
				
			if drift is not None:
				drift = np.reshape(np.arange(1,ntrial+1)/ntrial,(ntrial,1))
				X0 = np.concatenate((X0,drift),axis=1)
				
			if neurophil is not None:
				npile = np.reshape(np.nanmean(y,axis=1),(ntrial,1))
				X0 = np.concatenate((X0,npile),axis=1)
			X = X0

			# if drift is None and neurophil is None:
			# 	X = np.concatenate((x1,x2,x3,bias),axis=1)

			# elif drift is not None and neurophil is None:
			# 	drift = np.reshape(np.arange(1,ntrial+1)/ntrial,(ntrial,1)) # slow drift (ntrial,1) first order
			# 	# second order drift
			# 	# drift1 = np.reshape(np.arange(ntrial)**2/ntrial**2,(ntrial,1))
			# 	# drift2 = np.reshape(np.arange(ntrial)**3/ntrial**3,(ntrial,1))
			# 	X = np.concatenate((x1,x2,x3,bias,drift),axis=1) # (ntrial,n_v_stim+n_o_stim+n_sham)
			# elif drift is None and neurophil is not None:
			# 	npile = np.reshape(np.nanmean(y,axis=1),(ntrial,1))
			# 	# normalize npile to 0-1
			# 	# npile = (npile - np.min(npile))/(np.max(npile)-np.min(npile))
			# 	X = np.concatenate((x1,x2,x3,bias,npile),axis=1)
			# else:
			# 	drift = np.reshape(np.arange(1,ntrial+1)/ntrial,(ntrial,1))
			# 	npile = np.reshape(np.nanmean(y,axis=1),(ntrial,1))
			# 	# npile = (npile - np.min(npile))/(np.max(npile)-np.min(npile))
			# 	X = np.concatenate((x1,x2,x3,bias,drift,npile),axis=1)
				
		elif interaction == 'contrast':
			x_hcont = f['ShareData']['highCont'][()] #(ntrial, n_ostim)
			x_lcont = f['ShareData']['lowCont'][()] #(ntrial, n_ostim)
			x4 = np.concatenate((x_hcont,x_lcont),axis=1) #(ntrial, 2*n_ostim)
			print('x4: ',x4.shape)
			X = np.concatenate((x1,x4,x3,bias),axis=1)
		elif interaction == 'direction_single':
			# get the direction of the visual stim
			X0 = np.concatenate((x1,x2,x3),axis=1)
			if dir is None:
				dir = 0
				print('No direction specified, using the first direction')
			if bias_flag is not None:
				X0 = np.concatenate((X0,bias),axis=1)
			if drift is not None:
				drift_ = np.reshape(np.arange(1,ntrial+1)/ntrial,(ntrial,1))
				X0 = np.concatenate((X0,drift_),axis=1)
			if neurophil is not None:
				if np.isnan(y).any():
					print('NaN in y')
					y = np.nan_to_num(y)
				npile = np.reshape(np.nanmean(y,axis=1),(ntrial,1))
				npile = (npile - np.min(npile))/(np.max(npile)-np.min(npile)) # (ntrial,1)
				if neurophil == 'fix':
					if 'dldevel' in os.path.expanduser("~"):
						fov_idx_start = filename.find('ShareData_')
						fov_idx_end = filename.find('.mat')
						fov = filename[fov_idx_start+10:fov_idx_end]
						w_npile = np.load('/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/analysis_data/2024-10-17_noreg/shared_data{}/glm_weights_final.npy'.format(fov))[:,-1]
					else:
						w_npile = np.load('/Users/fionakong/Downloads/kaschube-lab/Influence_mapping/shareData/all_w.npy')[:,:,-1].mean(axis=0)#(ncell,)
					weight_npil = np.outer(npile,w_npile) #(ntrial,ncell)
					y = y / np.exp(weight_npil)
				else:
					X0 = np.concatenate((X0,npile),axis=1)
			X = get_dir_X(X0,dir,n_vstim)
			y = y[np.where(X0[:,dir] ==1)[0],:]
			print('debug (in direction_single): shape of X',X.shape)

		print('shape of x',X.shape)
		print('shape of y',y.shape)
		## remove last trial for dataset 31
		if 'ShareData_31' in filename:
			X = X[:-1,:]
			y = y[:-1,:]
			print('Remove last trial for dataset 31')
	return X,y*30



def construct_X_y_for_ve(filename,params):
	# construct the design matrix and response matrix 
	# for exploring the variance explained comtributuion of factor in design matrix
	# params: dict with keys 'visual' 'target' 'bias', 'drift', 'neurophil'
	with h5py.File(filename, 'r') as f:
		# print some dataset information
		if 'verbose' in params:
			print(f.keys())
			print(f['ShareData']['VisStim'])
			print(f['ShareData']['Response_dFF'])
			print(f['ShareData']['Response_inferredSpikes'])
			print(f['ShareData']['photoStim'])
			print(f['ShareData']['UseableNonTargetCells'])
			print(f['ShareData']['ShamLocations'])
			target_cells = f['ShareData']['TargetCells'][()][:,0]

		# get the response matrix
		## remove last trial for 31
		y = f['ShareData']['Response_inferredSpikes'][()] # (ntrial,ncell)
		ntrial,ncell = y.shape
		# print('shape of y',y.shape)
		# get the design matrix
		# visual
		x1 = f['ShareData']['VisStim'][()] # (ntrial,n_v_stim)
		_,n_vstim = x1.shape
		num_trial_per_vstim = np.sum(x1,axis=0)
		print('visual num_trial_per_stim',num_trial_per_vstim)
		# target
		x2 = f['ShareData']['photoStim'][()]
		num_trial_per_ostim = np.sum(x2,axis=0)
		print('opto num_trial_per_stim',num_trial_per_ostim)
		# sham
		x3 = f['ShareData']['Sham_photoStim'][()] # (ntrial,n_sham_stim)
		num_trial_per_shamstim = np.sum(x3,axis=0)
		print('sham num_trial_per_stim',num_trial_per_shamstim)
		# bias
		bias = np.ones((ntrial,1))
		# drift
		drift = np.arange(1,ntrial+1)/ntrial
		drift = np.reshape(drift,(ntrial,1))
		# neurophil
		if np.isnan(y).any():
			print('NaN in y')
			y = np.nan_to_num(y)
		npile = np.reshape(np.nanmean(y,axis=1),(ntrial,1))
		npile = (npile - np.min(npile))/(np.max(npile)-np.min(npile))
		#
		
		# Mapping from param keys to data arrays
		data_map = {
			'visual': x1,
			'target': x2,
			'bias': bias,
			'drift': drift,
			'npil': npile,
			'sham': x3
		}
		# Collect the arrays where params[key] is True
		arrays_to_concat = [data_map[key] for key in params if params[key]]

		# Concatenate along the second axis (features axis)
		if arrays_to_concat:
			combined = np.concatenate(arrays_to_concat, axis=1)
		else:
			combined = None  

		return combined, y*30  # Return the combined array and response matrix
		
		



	

def get_resource_availability(min_memory_gb=4, min_cores=2):
    import psutil
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # Available memory in GB
    available_cores = psutil.cpu_count(logical=True)  # Logical cores
    print(f"Available Memory: {available_memory:.2f} GB")
    print(f"Available Cores: {available_cores}")

    # Determine the number of workers based on memory and CPU
    if available_memory >= min_memory_gb and available_cores >= min_cores:
        # Adjust number of workers based on memory and CPU availability
        max_workers = min(available_cores, available_memory // min_memory_gb)
        return True, max_workers

    return False, 1  # Fallback to sequential execution


def split_sham_target_trials(opto_stims_bool,target_cells,sham_stims,resp):
	"""
	Get the sham trials
	(in the old version, first visial_stims, then mixed target and sham stims)
	opto_stims_bool: (ntrial,n_o_stim) #one-hot encoding
	target_cells: (n_real_stims,)
	sham_stims: (n_shams,)
	resp: (ntrial,ncell)
	
	returns:
	sham_trials: (n_shams,ncell)
	"""
	
	# get all targets (conbination of target and sham)
	all_targets = np.sort(np.concatenate((target_cells,sham_stims)))
	# print('target cells',target_cells)
	# print('sham locations',sham_stims)
	# print('all targets',all_targets)

	# get the indices of the sham stims
	sham_stimid = np.where(np.isin(all_targets,sham_stims))[0]
	# print('sham indices',sham_stimid)

	# in opto_stims_bool, get the where sham stims are true
	sham_stims_bool = opto_stims_bool[:,sham_stimid]

	# get the indices of the trials where sham stims are true
	sham_indices = np.where(np.sum(sham_stims_bool,axis=1)>0)[0]


	# get sham trials
	sham_trials = resp[sham_indices,:]

	target_trials = resp[np.setdiff1d(np.arange(resp.shape[0]),sham_indices),:]

	return sham_trials,sham_indices, target_trials

def get_sham_trial_inx(opto_stims_bool,target_stims, sham_stims):
	# get the indices of the sham stims
	# in the new ShareData, first visual_stims, then target_stims, then sham_stims
	# opto_stims_bool: (ntrial,nstim) one hot encoding
	# sham_stims: (nsham_stims,)
	# target_stims: (ntarget_stims,)
	# return: sham_indices: (nsham_trials,) target_indices: (ntarget_trials,)
	n_target_stims = len(target_stims)
	n_sham_stims = len(sham_stims)
	nstims = opto_stims_bool.shape[1]
	n_vstims = nstims - n_target_stims - n_sham_stims
	# get all trials where sham stims colloums are true
	## only one sham stim
	sham_indices = np.where(opto_stims_bool[:,-2]>0)[0]
	# get all trials where target stims colloums are true
	target_indices = np.where(np.sum(opto_stims_bool[:,n_vstims:-2],axis=1)>0)[0]
	return sham_indices, target_indices

def sham_cv_split(X,y,sham_indices,K):
	# split data into K folds, while make sure there are sham trials in each fold
	# X: (ntrials,nstims)
	# y: (ntrials,ncells)
	# sham_indices: (nsham_trials,)
	# K: number of folds

	ntrials = X.shape[0]
	nstims = X.shape[1]
	ncells = y.shape[1]
	ntrials_per_fold = ntrials//K
	nsham_per_fold = len(sham_indices)//K
	nrest_per_fold = ntrials_per_fold - nsham_per_fold
	
	# randomly select nsham_per_fold sham trials for each fold
	np.random.shuffle(sham_indices)
	sham_indices = sham_indices[:nsham_per_fold*K]
	sham_indices = np.reshape(sham_indices,(K,nsham_per_fold))

	# get the indices of the non-sham trials
	non_sham_indices = np.setdiff1d(np.arange(ntrials),sham_indices.flatten())

	# randomly select nrest_per_fold non-sham trials for each fold
	np.random.shuffle(non_sham_indices)
	non_sham_indices = non_sham_indices[:nrest_per_fold*K]
	non_sham_indices = np.reshape(non_sham_indices,(K,nrest_per_fold))

	# combine the sham and non-sham indices
	indices = np.concatenate((sham_indices,non_sham_indices),axis=1)

	# split the data
	X_train = np.zeros((K,ntrials_per_fold,nstims))
	y_train = np.zeros((K,ntrials_per_fold,ncells))


	for i in range(K):
		train_indices = indices[i]
		X_train[i] = X[train_indices.flatten(),:]
		y_train[i] = y[train_indices.flatten(),:]

	return X_train,y_train


def sham_bootstrapping(X,y,sham_indices,K):
	# create K repeats of data, while make sure there are sham trials in each repeat
	# X: (ntrials,nstims)
	# y: (ntrials,ncells)
	# sham_indices: (nsham_trials,)
	# K: number of folds

	ntrials = X.shape[0]
	nstims = X.shape[1]
	ncells = y.shape[1]
	ntrials_per_fold = ntrials
	nsham_per_fold = len(sham_indices)
	nrest_per_fold = ntrials_per_fold - nsham_per_fold

	target_indecies = np.setdiff1d(np.arange(ntrials),sham_indices) # (ntrials-nsham_trials,)

	X_train = np.zeros((K,ntrials_per_fold,nstims))
	y_train = np.zeros((K,ntrials_per_fold,ncells))
	
	for i in range(K):
		# randomly select sham trials with replacement
		sham_inx_i = np.random.choice(sham_indices,nsham_per_fold,replace=True) # (nsham_per_fold,)
		# get the indices of the non-sham trials
		target_inx_i = np.random.choice(target_indecies,nrest_per_fold,replace=True) # (nrest_per_fold,)
		# combine the sham and non-sham indices
		indices = np.concatenate((sham_inx_i,target_inx_i),axis=0).astype(int) # (ntrials_per_fold,)
		# split the data		
		
		X_train[i] = X[indices.flatten(),:]
		y_train[i] = y[indices.flatten(),:]

	return X_train,y_train


def train_test_split(X,y,i,K):
	# take the full design matrix (ntrails, nstims) and response matrix (ntrials,ncells)
	# and split them into training and testing sets
	# i is the index of the test set
	# K is the number of folds
	ntrials = X.shape[0]
	nstims = X.shape[1]
	ntrials_per_fold = ntrials//K
	train_indices = np.setxor1d(np.arange(ntrials),np.arange(i*ntrials_per_fold,(i+1)*ntrials_per_fold))
	test_indices = np.arange(i*ntrials_per_fold,(i+1)*ntrials_per_fold)
	X_train = X[train_indices,:]
	y_train = y[train_indices]
	X_test = X[test_indices,:]
	y_test = y[test_indices]
	
	return (X_train,y_train,X_test,y_test)


def Kfold_CV(K,lambdas,l1_wts,X,y):
	#K-fold CV: train on K-1 sessions, test on held-out session.
	num_folds = K
	validation_performance = np.zeros((len(lambdas),len(l1_wts),num_folds))
	r2 = np.zeros((len(lambdas),len(l1_wts),num_folds))
	ed = np.zeros((len(lambdas),len(l1_wts),num_folds))

	all_ytest = []
	all_ypredicted = []
	all_ytrain = []
	all_yh_train = []

	get_resource_availability()

	# def calculate_performance(X,y,i_fold, i_lambda, i_l1, lamb, l1_wt, K):
	# 	X_train,y_train,X_test,y_test = train_test_split(X,y,i_fold,K)
	# 	glm = sm.GLM(y_train,X_train,family=sm.families.Poisson())
	# 	# -loglike/n + alpha(1-l1_wt)*||params||_2^2 + l1_wt*||params||_1
	# 	res = glm.fit_regularized(method='elastic_net',
	# 								alpha=lamb,
	# 								maxiter=2000,
	# 								L1_wt=l1_wt # 1.0:all L1, 0.0: all L2
	# 								)
	# 	yh_train = glm.predict(params=res.params,exog=X_train)
	# 	y_predicted = glm.predict(params=res.params,exog=X_test)
	# 	res = {}
	# 	res['val_mse'] = np.mean((y_test-y_predicted)**2)
	# 	res['val_r2'] = 1 - np.sum((y_test-y_predicted)**2)/np.sum((y_test-np.mean(y_test))**2)
	# 	dev = 2*np.nansum(y_test*np.ma.masked_invalid(np.log(y_test/y_predicted)) - y_test +y_predicted)
	# 	mu = np.nanmean(y_test)
	# 	td = 2*np.nansum(y_test*np.ma.masked_invalid(np.log(y_test/mu)) - y_test + mu)
	# 	res['val_ed'] =  1 - dev/td
	# 	return i_fold, i_lambda, i_l1, res
	
	# tasks = [( i_lambda, i_l1, lamb, l1_wt)
			
	# 		for i_lambda, lamb in enumerate(lambdas)
	# 		for i_l1, l1_wt in enumerate(l1_wts)]
		
	# for i_fold in range(K):
	# 	with ProcessPoolExecutor() as executor:
	# 		results = list(tqdm(executor.map(lambda x: calculate_performance(*x), tasks), total=len(tasks)))

	
	# # Store the results in the appropriate index
	# for i_fold, i_lambda, i_l1, res in results:
	# 	validation_performance[i_lambda, i_l1, i_fold] = res['val_mse']
	# 	r2[i_lambda, i_l1, i_fold] = res['val_r2']
	# 	ed[i_lambda, i_l1, i_fold] = res['val_ed']


	for i_fold in tqdm(range(K),position=0,colour='green'):
		X_train,y_train,X_test,y_test = train_test_split(X,y,i_fold,K)
		for i_lambda, lamb in enumerate(lambdas):
			for i_l1, l1_wt in enumerate(l1_wts):
				glm = sm.GLM(y_train,X_train,family=sm.families.Poisson())
				# -loglike/n + alpha(1-l1_wt)*||params||_2^2 + l1_wt*||params||_1
				res = glm.fit_regularized(method='elastic_net',
											alpha=lamb,
											maxiter=2000,
											L1_wt=l1_wt # 1.0:all L1, 0.0: all L2
											)
				yh_train = glm.predict(params=res.params,exog=X_train)
				y_predicted = glm.predict(params=res.params,exog=X_test)
				validation_performance[i_lambda,i_l1,i_fold] = np.mean((y_test-y_predicted)**2)
				r2[i_lambda,i_l1,i_fold] = 1 - np.sum((y_test-y_predicted)**2)/np.sum((y_test-np.mean(y_test))**2)
				# ed[i_lambda,i_l1,i_fold] = 2*np.sum(y_test* np.ma.masked_invalid(np.log(y_test/y_predicted)) - y_test + y_predicted)
				dev = 2*np.nansum(y_test*np.ma.masked_invalid(np.log(y_test/y_predicted)) - y_test +y_predicted)
				mu = np.nanmean(y_test)
				td = 2*np.nansum(y_test*np.ma.masked_invalid(np.log(y_test/mu)) - y_test + mu)
				ed[i_lambda,i_l1,i_fold] =  1 - dev/td
				# print('lambda: ' + str(lamb) + ' perf: ' + str(validation_performance[i_lambda,i_fold]))  
				# all_ytest.append(y_test)
				# all_ypredicted.append(y_predicted)   
				# all_ytrain.append(y_train)
				# all_yh_train.append(yh_train)
	return validation_performance, r2, ed #, all_ytest, all_ypredicted, all_ytrain, all_yh_train

def K_CV_coarse_pass(K,savepath,plotpath,X,y,LAMBDA_SAMPLES=10,l1_samples=5):
	#cross-validation: first pass with wide range of lambdas
	if os.path.isfile(savepath+'_validation_performance.npy'):
		validation_performance = np.load(savepath+'_validation_performance.npy')
		r2 = np.load(savepath+'_r2.npy')
		ed = np.load(savepath+'_ed.npy')
		lambdas = np.load(savepath+'_validation_lambdas.npy')
		l1_wts = np.load(savepath+'_validation_l1_wts.npy')
		# ytest = np.load(savepath+'_ytest.npy')
		# yhat = np.load(savepath+'_yhat.npy')
		# ytrain = np.load(savepath+'_ytrain.npy')
		# yhtrain = np.load(savepath+'_yhtrain.npy')

	else:
		lambdas = 5* 10.0 ** np.linspace(-10.0,1.0,num=LAMBDA_SAMPLES)
		l1_wts = np.linspace(0.0,1.0,num=l1_samples)
		validation_performance,r2,ed,ytest,yhat,ytrain,yhtrain = Kfold_CV(K,lambdas,l1_wts,X,y)
		np.save(savepath+'_validation_lambdas.npy',lambdas)
		np.save(savepath+'_validation_l1_wts.npy',l1_wts)
		np.save(savepath+'_validation_performance.npy',validation_performance)
		np.save(savepath+'_r2.npy',r2)
		np.save(savepath+'_ed.npy',ed)
		# np.save(savepath+'_ytest.npy',ytest)
		# np.save(savepath+'_yhat.npy',yhat)
		# np.save(savepath+'_ytrain.npy',ytrain)
		# np.save(savepath+'_yhtrain.npy',yhtrain)

	Kfold_performance = np.nanmean(validation_performance,axis=2)    # (nlambda,nl1)
	lambda_min_idx,l1_wt_min_idx = np.unravel_index(np.argmin(Kfold_performance, axis=None), Kfold_performance.shape)
	# print('lambdas',lambdas)
	# print('Kfold_performance',Kfold_performance)  

	### plotting
	## plot MSE
	fig = plt.figure(figsize=(6,6))
	im = plt.imshow(Kfold_performance,aspect='auto',interpolation='none',origin='lower')
	plt.yticks(np.arange(len(lambdas)),[str(x) for x in lambdas] )
	plt.ylabel('lambda')
	plt.xticks(np.arange(len(l1_wts)),['{:.2f}'.format(x) for x in l1_wts])
	plt.xlabel('L1_wt')
	plt.title('Coarse pass MSE')
	cbar = plt.colorbar(im)
	cbar.set_label('MSE')
	plt.scatter(l1_wt_min_idx,lambda_min_idx,marker='x',color='r')
	plt.savefig(plotpath+'validation_performance_coarse.png',bbox_inches='tight')
	plt.close(fig)

	## plot r^2
	Kfold_r2 = np.nanmean(r2,axis=2) 
	fig = plt.figure(figsize=(6,6))
	im = plt.imshow(Kfold_r2,aspect='auto',interpolation='none',origin='lower')
	plt.yticks(np.arange(len(lambdas)),[str(x) for x in lambdas] )
	plt.ylabel('lambda')
	plt.xticks(np.arange(len(l1_wts)),['{:.2f}'.format(x) for x in l1_wts])
	plt.xlabel('L1_wt')
	plt.title('Coarse pass R^2')
	cbar = plt.colorbar(im)
	cbar.set_label('R^2')
	plt.scatter(l1_wt_min_idx,lambda_min_idx,marker='x',color='r')
	plt.savefig(plotpath+'r2_coarse.png',bbox_inches='tight')
	plt.close(fig)

	## plot explained deviance
	Kfold_ed = np.nanmean(ed,axis=2)
	fig = plt.figure(figsize=(6,6))
	im = plt.imshow(Kfold_ed,aspect='auto',interpolation='none',origin='lower')
	plt.yticks(np.arange(len(lambdas)),[str(x) for x in lambdas] )
	plt.ylabel('lambda')
	plt.xticks(np.arange(len(l1_wts)),['{:.2f}'.format(x) for x in l1_wts])
	plt.xlabel('L1_wt')
	plt.title('Coarse pass Explained deviance')
	cbar = plt.colorbar(im)
	cbar.set_label('Deviance')
	plt.scatter(l1_wt_min_idx,lambda_min_idx,marker='x',color='r')
	plt.savefig(plotpath+'ed_coarse.png',bbox_inches='tight')
	plt.close(fig)

	## plot ytest and yhat
	# y_test = []
	# y_predicted = []
	# cnt = 0
	# for i_fold in range(K):
	# 	pp = PdfPages(plotpath+'ytest_yhat_fold{}.pdf'.format(i_fold))
	# 	pp1 = PdfPages(plotpath+'ytest_yhat_sorted_fold{}.pdf'.format(i_fold))
	# 	pp_train = PdfPages(plotpath+'ytrain_yhtrain_fold{}.pdf'.format(i_fold))
	# 	pp1_train = PdfPages(plotpath+'ytrain_yhtrain_sorted_fold{}.pdf'.format(i_fold))
	# 	for i_lambda, lamb in enumerate(lambdas):
	# 		fig = plt.figure(figsize=(5*len(l1_wts),6))
	# 		fig.tight_layout()
	# 		fig1 = plt.figure(figsize=(5*len(l1_wts),6))
	# 		fig1.tight_layout()
	# 		fig_train = plt.figure(figsize=(5*len(l1_wts),6))
	# 		fig_train.tight_layout()
	# 		fig1_train = plt.figure(figsize=(5*len(l1_wts),6))
	# 		fig1_train.tight_layout()

	# 		for i_l1, l1_wt in enumerate(l1_wts):
	# 			ax = fig.add_subplot(1,len(l1_wts),i_l1+1)
	# 			ax.scatter(ytest[cnt],yhat[cnt])
	# 			ax.set_xlabel('Activity')
	# 			ax.set_ylabel('Predicted activity')
	# 			ax.set_title('l1:{:.2f} lam:{:.2f}'.format(l1_wt,lamb))

	# 			ax1 = fig1.add_subplot(1,len(l1_wts),i_l1+1)
	# 			y_sorted,yhat_sorted,_ = sort_together(ytest[cnt],yhat[cnt])
	# 			ax1.scatter(np.arange(len(y_sorted)),y_sorted,label='ytest')
	# 			ax1.scatter(np.arange(len(yhat_sorted)),yhat_sorted,label='yhat')
	# 			ax1.legend()
	# 			ax1.set_xlabel('Trials')
	# 			ax1.set_ylabel('Predicted activity')
	# 			ax1.set_title('l1:{:.2f} lam:{:.2f}'.format(l1_wt,lamb))

	# 			ax_train = fig_train.add_subplot(1,len(l1_wts),i_l1+1)
	# 			ax_train.scatter(ytrain[cnt],yhtrain[cnt])
	# 			ax_train.set_xlabel('Activity')
	# 			ax_train.set_ylabel('Predicted activity')
	# 			ax_train.set_title('l1:{:.2f} lam:{:.2f}'.format(l1_wt,lamb))

	# 			ax1_train = fig1_train.add_subplot(1,len(l1_wts),i_l1+1)
	# 			ytrain_sorted,yhtrain_sorted,_ = sort_together(ytrain[cnt],yhtrain[cnt])
	# 			ax1_train.scatter(np.arange(len(ytrain_sorted)),ytrain_sorted,label='ytrain')
	# 			ax1_train.scatter(np.arange(len(yhtrain_sorted)),yhtrain_sorted,label='yhtrain')
	# 			ax1_train.legend()
	# 			ax1_train.set_xlabel('Trials')
	# 			ax1_train.set_ylabel('Predicted activity')
	# 			ax1_train.set_title('l1:{:.2f} lam:{:.2f}'.format(l1_wt,lamb))

	# 			if i_lambda == lambda_min_idx and i_l1 == l1_wt_min_idx:
	# 				y_test.append(ytest[cnt])
	# 				y_predicted.append(yhat[cnt])
	# 			cnt += 1
	# 		pp.savefig(fig)
	# 		pp1.savefig(fig1)
	# 		pp_train.savefig(fig_train)
	# 		pp1_train.savefig(fig1_train)

	# 		plt.close(fig)
	# 		plt.close(fig1)
	# 		plt.close(fig_train)
	# 		plt.close(fig1_train)

	# 	pp.close()
	# 	pp1.close()
	# 	pp_train.close()
	# 	pp1_train.close()


						
	# flatten ytest and yhat
	# y_test = np.concatenate(y_test).flatten()
	# y_predicted = np.concatenate(y_predicted).flatten()
	# fig = plt.figure(figsize=(6,6))
	# plt.scatter(y_test,y_predicted)
	# plt.xlabel('Activity')
	# plt.ylabel('Predicted activity')
	# plt.title('ytest vs yhat')
	# plt.savefig(plotpath+'ytest_yhat.png',bbox_inches='tight')
	# plt.close(fig)

	


	# sort ytest and keep the same order for yhat
	# y_test,y_predicted,_ = sort_together(y_test,y_predicted)
	# fig = plt.figure(figsize=(6,6))
	# plt.scatter(np.arange(len(y_test)),y_test,label='ytest')
	# plt.scatter(np.arange(len(y_predicted)),y_predicted,label='yhat')
	# plt.legend()
	# plt.xlabel('')
	# plt.ylabel('Predicted activity')
	# plt.savefig(plotpath+'ytest_yhat_sorted.png',bbox_inches='tight')
	# plt.close(fig)

	# fig = plt.figure(figsize=(6,6))
	# plt.scatter(np.arange(len(y_predicted)),np.sort(y_predicted),label='ytest')
	# plt.xlabel('Activity')
	# plt.ylabel('Predicted activity')
	# plt.title('yhat')
	# plt.savefig(plotpath+'yhat_sorted.png',bbox_inches='tight')
	# plt.close(fig)

	# ## plot y distribution
	# fig = plt.figure(figsize=(6,6))
	# plt.hist(y_test,bins=50,alpha=0.5,label='ytest')
	# plt.hist(y_predicted,bins=50,alpha=0.5,label='yhat')
	# plt.legend()
	# plt.xlabel('Activity')
	# plt.ylabel('Number of trials')
	# plt.title('Activity distribution')
	# plt.savefig(plotpath+'hist_y.png',bbox_inches='tight')
	# plt.close(fig)
	return None

def K_CV_fine_pass(K,savepath,plotpath,X,y,lambdas_fine,l1s_fine):
	# do a second pass near the minimum:
	if os.path.isfile(savepath+'_validation_performance_fine.npy'):
		validation_performance_fine = np.load(savepath+'_validation_performance_fine.npy')
		r2 = np.load(savepath+'_r2_fine.npy')
		ed = np.load(savepath+'_ed_fine.npy')
		lambdas_fine = np.load(savepath+'_validation_lambdas_fine.npy')
		l1s_fine = np.load(savepath+'_validation_l1s_fine.npy')
		# ytest = np.load(savepath+'_ytest_fine.npy')
		# yhat = np.load(savepath+'_yhat_fine.npy')
		# ytrain = np.load(savepath+'_ytrain_fine.npy')
		# yhtrain = np.load(savepath+'_yhtrain_fine.npy')

	else:
		validation_performance_fine,r2,ed,ytest,yhat,ytrain,yhtrain = Kfold_CV(K,lambdas_fine,l1s_fine,X,y)
		np.save(savepath+'_validation_lambdas_fine.npy',lambdas_fine)
		np.save(savepath+'_validation_performance_fine.npy',validation_performance_fine)
		np.save(savepath+'_r2_fine.npy',r2)
		np.save(savepath+'_ed_fine.npy',ed)
		np.save(savepath+'_validation_l1s_fine.npy',l1s_fine)
		# np.save(savepath+'_ytest_fine.npy',ytest)
		# np.save(savepath+'_yhat_fine.npy',yhat)
		# np.save(savepath+'_ytrain_fine.npy',ytrain)
		# np.save(savepath+'_yhtrain_fine.npy',yhtrain)


	Kfold_performance_fine = np.nanmean(validation_performance_fine,axis=2)    # (nlambda,nl1)

	#find the optimal lambda and l1_wt
	lambda_opt_idx,l1_wt_opt_idx= np.unravel_index(np.argmin(Kfold_performance_fine, axis=None), Kfold_performance_fine.shape)
	print('fine: lambda_opt_idx:',lambda_opt_idx, ', l1_wt_opt_idx: ',l1_wt_opt_idx)

	#plot results mse
	fig = plt.figure(figsize=(6,6))
	im = plt.imshow(Kfold_performance_fine,aspect='auto',interpolation='none',origin='lower')
	plt.yticks(np.arange(len(lambdas_fine)),[str(x) for x in lambdas_fine] )
	plt.ylabel('lambda')
	plt.xticks(np.arange(len(l1s_fine)),['{:.2f}'.format(x) for x in l1s_fine])
	plt.xlabel('L1_wt')
	cbar = plt.colorbar(im)
	cbar.set_label('MSE')
	plt.scatter(l1_wt_opt_idx,lambda_opt_idx,marker='x',color='r')
	plt.savefig(plotpath+'validation_performance_fine.png',bbox_inches='tight')
	plt.close(fig)

	#plot r^2
	Kfold_r2 = np.nanmean(r2,axis=2)
	fig = plt.figure(figsize=(6,6))
	im = plt.imshow(Kfold_r2,aspect='auto',interpolation='none',origin='lower')
	plt.yticks(np.arange(len(lambdas_fine)),[str(x) for x in lambdas_fine] )
	plt.ylabel('lambda')
	plt.xticks(np.arange(len(l1s_fine)),['{:.2f}'.format(x) for x in l1s_fine])
	plt.xlabel('L1_wt')
	cbar = plt.colorbar(im)
	cbar.set_label('R^2')
	plt.scatter(l1_wt_opt_idx,lambda_opt_idx,marker='x',color='r')
	plt.savefig(plotpath+'r2_fine.png',bbox_inches='tight')
	plt.close(fig)

	#plot explained deviance
	Kfold_ed = np.nanmean(ed,axis=2)
	fig = plt.figure(figsize=(6,6))
	im = plt.imshow(Kfold_ed,aspect='auto',interpolation='none',origin='lower')
	plt.yticks(np.arange(len(lambdas_fine)),[str(x) for x in lambdas_fine] )
	plt.ylabel('lambda')
	plt.xticks(np.arange(len(l1s_fine)),['{:.2f}'.format(x) for x in l1s_fine])
	plt.xlabel('L1_wt')
	cbar = plt.colorbar(im)
	cbar.set_label('Deviance')
	plt.scatter(l1_wt_opt_idx,lambda_opt_idx,marker='x',color='r')
	plt.savefig(plotpath+'ed_fine.png',bbox_inches='tight')
	plt.close(fig)

	

	## plot ytest and yhat
	# y_test = []
	# y_predicted = []
	# cnt = 0
	# for i in range(K):
	# 	pp = PdfPages(plotpath+'fine_ytest_yhat_fold{}.pdf'.format(i))
	# 	pp1 = PdfPages(plotpath+'fine_ytest_yhat_sorted_fold{}.pdf'.format(i))
	# 	pp_train = PdfPages(plotpath+'fine_ytrain_yhtrain_fold{}.pdf'.format(i))
	# 	pp1_train = PdfPages(plotpath+'fine_ytrain_yhtrain_sorted_fold{}.pdf'.format(i))
	# 	for i_lambda, lamb in enumerate(lambdas_fine):
	# 		fig = plt.figure(figsize=(5*len(l1s_fine),6))
	# 		fig.tight_layout()
	# 		fig1 = plt.figure(figsize=(5*len(l1s_fine),6))
	# 		fig1.tight_layout()
	# 		fig_train = plt.figure(figsize=(5*len(l1s_fine),6))
	# 		fig_train.tight_layout()
	# 		fig1_train = plt.figure(figsize=(5*len(l1s_fine),6))
	# 		fig1_train.tight_layout()
	# 		for i_l1, l1_wt in enumerate(l1s_fine):
	# 			ax = fig.add_subplot(1,len(l1s_fine),i_l1+1)
	# 			ax.scatter(ytest[cnt],yhat[cnt])
	# 			ax.set_xlabel('Activity')
	# 			ax.set_ylabel('Predicted activity')
	# 			ax.set_title('l1: {:.2f} lam: {:.2f}'.format(l1_wt,lamb))

	# 			ax1 = fig1.add_subplot(1,len(l1s_fine),i_l1+1)
	# 			y_sorted,yhat_sorted,_ = sort_together(ytest[cnt],yhat[cnt])
	# 			ax1.scatter(np.arange(len(y_sorted)),y_sorted,label='ytest')
	# 			ax1.scatter(np.arange(len(yhat_sorted)),yhat_sorted,label='yhat')
	# 			ax1.legend()
	# 			ax1.set_xlabel('Trials')
	# 			ax1.set_ylabel('Predicted activity')
	# 			ax1.set_title('l1: {:.2f} lam: {:.2f}'.format(l1_wt,lamb))

	# 			ax_train = fig_train.add_subplot(1,len(l1s_fine),i_l1+1)
	# 			ax_train.scatter(ytrain[cnt],yhtrain[cnt])
	# 			ax_train.set_xlabel('Activity')
	# 			ax_train.set_ylabel('Predicted activity')
	# 			ax_train.set_title('l1: {:.2f} lam: {:.2f}'.format(l1_wt,lamb))

	# 			ax1_train = fig1_train.add_subplot(1,len(l1s_fine),i_l1+1)
	# 			ytrain_sorted,yhtrain_sorted,_ = sort_together(ytrain[cnt],yhtrain[cnt])
	# 			ax1_train.scatter(np.arange(len(ytrain_sorted)),ytrain_sorted,label='ytrain')
	# 			ax1_train.scatter(np.arange(len(yhtrain_sorted)),yhtrain_sorted,label='yhtrain')
	# 			ax1_train.legend()
	# 			ax1_train.set_xlabel('Trials')
	# 			ax1_train.set_ylabel('Predicted activity')
	# 			ax1_train.set_title('l1: {:.2f} lam: {:.2f}'.format(l1_wt,lamb))

	# 			if i_lambda == lambda_opt_idx and i_l1 == l1_wt_opt_idx:

	# 				y_test.append(np.asarray(ytest[cnt]))
	# 				y_predicted.append(yhat[cnt])
	# 			cnt += 1
	# 		pp.savefig(fig)
	# 		pp1.savefig(fig1)
	# 		pp_train.savefig(fig_train)
	# 		pp1_train.savefig(fig1_train)

	# 		plt.close(fig)
	# 		plt.close(fig1)
	# 		plt.close(fig_train)
	# 		plt.close(fig1_train)
	# 	pp.close()
	# 	pp1.close()
	# 	pp_train.close()
	# 	pp1_train.close()

	# # flatten ytest and yhat
	
	# y_test = np.concatenate(y_test).flatten()
	# y_predicted = np.concatenate(y_predicted).flatten()
	# fig = plt.figure(figsize=(6,6))
	# plt.scatter(y_test,y_predicted)
	# plt.xlabel('Activity')
	# plt.ylabel('Predicted activity')
	# plt.title('ytest vs yhat')
	# plt.savefig(plotpath+'ytest_yhat_fine.png',bbox_inches='tight')
	# plt.close(fig)
	
	

	# # sort ytest and keep the same order for yhat
	# y_test,y_predicted,_ = sort_together(y_test,y_predicted)
	# fig = plt.figure(figsize=(6,6))
	# plt.scatter(np.arange(len(y_test)),y_test,label='ytest')
	# plt.scatter(np.arange(len(y_predicted)),y_predicted,label='yhat')
	# plt.legend()
	# plt.xlabel('')
	# plt.ylabel('Predicted activity')
	# plt.savefig(plotpath+'ytest_yhat_fine_sorted.png',bbox_inches='tight')
	# plt.close(fig)

	# ## plot distribution of ytest and yhat
	# fig = plt.figure(figsize=(6,6))
	# plt.hist(y_test,bins=50,alpha=0.5,label='ytest')
	# plt.hist(y_predicted,bins=50,alpha=0.5,label='yhat')
	# plt.xlabel('Activity')
	# plt.ylabel('Count')
	# plt.legend()
	# plt.savefig(plotpath+'hist_y_fine.png',bbox_inches='tight')



	return None
	


### old function that finds optimal hyperparams for each cell
def K_session_cross_validation(K,
								savepath,
								plotpath,
								X,y,LAMBDA_SAMPLES=10,l1_samples=5):

	#cross-validation is to determine which parameters make a significant
	#     contribution to model performance.

	#first pass with wide range of lambdas
	if os.path.isfile(savepath+'_validation_performance.npy'):
		validation_performance = np.load(savepath+'_validation_performance.npy')
		lambdas = np.load(savepath+'_validation_lambdas.npy')
		l1_wts = np.load(savepath+'_validation_l1_wts.npy')
	else:
		lambdas = 10.0 ** np.linspace(-5.0,3.0,num=LAMBDA_SAMPLES)
		l1_wts = np.linspace(0.0,1.0,num=LAMBDA_SAMPLES)
		validation_performance = Kfold_CV(K,lambdas,l1_wts,X,y)
		np.save(savepath+'_validation_lambdas.npy',lambdas)
		np.save(savepath+'_validation_l1_wts.npy',l1_wts)
		np.save(savepath+'_validation_performance.npy',validation_performance)

	Kfold_performance = np.nanmean(validation_performance,axis=2)    # (nlambda,nl1)
	lambda_min_idx,l1_wt_min_idx = np.unravel_index(np.argmin(Kfold_performance, axis=None), Kfold_performance.shape)
	# print('lambdas',lambdas)
	# print('Kfold_performance',Kfold_performance)  

	# do a second pass near the minimum:
	if os.path.isfile(savepath+'_validation_performance_fine.npy'):
		validation_performance_fine = np.load(savepath+'_validation_performance_fine.npy')
		lambdas_fine = np.load(savepath+'_validation_lambdas_fine.npy')
		l1s_fine = np.load(savepath+'_validation_l1s_fine.npy')
	else:
		
		lambda_lb = lambdas[lambda_min_idx-1]
		lambda_ub = lambdas[lambda_min_idx+2]

		if l1_wt_min_idx == 0:
			l1_lb = 0.0
			l1_ub = l1_wts[l1_wt_min_idx+2]
		elif l1_wt_min_idx == len(l1_wts)-1 or l1_wt_min_idx == len(l1_wts)-2:
			l1_lb = l1_wts[l1_wt_min_idx-1]
			l1_ub = 1.0
		else:
			l1_lb = l1_wts[l1_wt_min_idx-1]
			l1_ub = l1_wts[l1_wt_min_idx+2]
		
		lambdas_fine = np.linspace(lambda_lb,lambda_ub,num=LAMBDA_SAMPLES)
		l1s_fine = np.linspace(l1_lb,l1_ub,num=l1_samples)
		validation_performance_fine = Kfold_CV(K,lambdas_fine,l1s_fine,X,y)# (nlambda,nl1,nfold)
		np.save(savepath+'_validation_lambdas_fine.npy',lambdas_fine)
		np.save(savepath+'_validation_performance_fine.npy',validation_performance_fine)
		np.save(savepath+'_validation_l1s_fine.npy',l1s_fine)
	
	Kfold_performance_fine = np.nanmean(validation_performance_fine,axis=2)    # (nlambda,nl1)

	#find the optimal lambda and l1_wt
	lambda_opt_idx,l1_wt_opt_idx= np.unravel_index(np.argmin(Kfold_performance_fine, axis=None), Kfold_performance_fine.shape)

	#plot results


	fig = plt.figure(figsize=(10,9))
	ax = fig.add_subplot(121)
	im = ax.imshow(Kfold_performance,aspect='auto',interpolation='none',origin='lower')
	ax.scatter(l1_wt_min_idx,lambda_min_idx,marker='x',color='r')
	ax.set_yticks(np.log(lambdas))
	ax.set_yticklabels([str(x) for x in lambdas])
	ax.set_ylabel('lambda')
	ax.set_xticks(np.arange(len(l1_wts)))
	ax.set_xticklabels(['{:.2f}'.format(x) for x in l1_wts])
	ax.set_xlabel('L1_wt')

	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im, cax=cax, orientation='vertical')
	cbar.set_label('MSE')

	ax1 = plt.subplot(122)
	im1 = ax1.imshow(Kfold_performance_fine,aspect='auto',interpolation='none',origin='lower')
	ax1.scatter(l1_wt_opt_idx,lambda_opt_idx,marker='x',color='r')
	ax1.set_yticks(np.arange(len(lambdas_fine)))
	ax1.set_yticklabels([str(x) for x in lambdas_fine])
	ax1.set_ylabel('lambda')
	ax1.set_xticks(np.arange(len(l1s_fine)))
	ax1.set_xticklabels(['{:.2f}'.format(x) for x in l1s_fine])
	ax1.set_xlabel('L1_wt')

	divider1 = make_axes_locatable(ax1)
	cax1 = divider1.append_axes('right', size='5%', pad=0.05)
	cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
	cbar1.set_label('MSE')

	plt.savefig(plotpath+'validation_performance.png',bbox_inches='tight',dpi=300)
	plt.close(fig)

	if Kfold_performance_fine[lambda_opt_idx,l1_wt_opt_idx]:
		return lambdas_fine[lambda_opt_idx],l1s_fine[l1_wt_opt_idx]
	else:
		return lambdas[lambda_min_idx],l1_wts[l1_wt_min_idx]


### old function that finds optimal hyperparams for each cell
def load_best_hyperparameters(savepath,ncells):
	# load the optimal hyperparameters for each cell
	# if not found, nan
	# return: (ncells,2) (lambda,l1_wt) if avg is true, return (2,) (lambda,l1_wt)

	# optomal parameter for each cell
	
	opt_params = np.zeros((ncells,2))
	for i in range(ncells):
		try:
			cv_loss_fine = np.nanmean(np.load(savepath + 'y{}_validation_performance_fine.npy'.format(i)),axis=2) # (nlambda,nl1)
			lambdas_fine = np.load(savepath + 'y{}_validation_lambdas_fine.npy'.format(i))
			l1s_fine = np.load(savepath + 'y{}_validation_l1s_fine.npy'.format(i))
			cv_loss_coarse = np.nanmean(np.load(savepath + 'y{}_validation_performance.npy'.format(i)),axis=2) # (nlambda,nl1)
			lambdas = np.load(savepath + 'y{}_validation_lambdas.npy'.format(i))
			l1s = np.load(savepath + 'y{}_validation_l1_wts.npy'.format(i))
		except Exception:
			# print(traceback.format_exc())
			print('No hyperparameter found for cell {}, set to nan'.format(i))
			opt_params[i,0] = np.nan
			opt_params[i,1] = np.nan
			continue

		if cv_loss_fine.min() < cv_loss_coarse.min():
			lambda_opt_idx,l1_wt_opt_idx = np.unravel_index(np.argmin(cv_loss_fine, axis=None), cv_loss_fine.shape)
			opt_params[i,0] = lambdas_fine[lambda_opt_idx]
			opt_params[i,1] = l1s_fine[l1_wt_opt_idx]
	
		else:
			lambda_opt_idx,l1_wt_opt_idx = np.unravel_index(np.argmin(cv_loss_coarse, axis=None), cv_loss_coarse.shape)
			opt_params[i,0] = lambdas[lambda_opt_idx]
			opt_params[i,1] = l1s[l1_wt_opt_idx]

	return opt_params

def load_best_hyperparameters_avg(savepath):
	# load the optimal hyperparameters
	# if not found, nan
	# return: (2,) (lambda,l1_wt) if avg is true, return (2,) (lambda,l1_wt)
	opt_params = np.zeros((2,))

	# load params and avg loss for 2 passes
	lambdas_fine = np.load(savepath + 'CV_fine/y0_validation_lambdas_fine.npy')
	l1s_fine = np.load(savepath + 'CV_fine/y0_validation_l1s_fine.npy')
	lambdas = np.load(savepath + 'CV/y0_validation_lambdas.npy')
	l1s = np.load(savepath + 'CV/y0_validation_l1_wts.npy')
	try:
		cv_loss_fine = np.load(savepath + 'CV_fine/avg_validation_performance_fine.npy')# (nlambda,nl1)
		cv_loss_coarse = np.load(savepath + 'CV/avg_validation_performance_coarse.npy')# (nlambda,nl1)
	except Exception:
		# print(traceback.format_exc())
		print('No hyperparameter found, set to nan')
		opt_params[0] = np.nan
		opt_params[1] = np.nan
		return opt_params
		

	if cv_loss_fine.min() < cv_loss_coarse.min():
		lambda_opt_idx,l1_wt_opt_idx = np.unravel_index(np.argmin(cv_loss_fine, axis=None), cv_loss_fine.shape)
		opt_params[0] = lambdas_fine[lambda_opt_idx]
		opt_params[1] = l1s_fine[l1_wt_opt_idx]

	else:
		lambda_opt_idx,l1_wt_opt_idx = np.unravel_index(np.argmin(cv_loss_coarse, axis=None), cv_loss_coarse.shape)
		opt_params[0] = lambdas[lambda_opt_idx]
		opt_params[1] = l1s[l1_wt_opt_idx]

	return opt_params

def get_cv_loss(savepath,ncells):
	# if not found, nan
	# return: (ncells,) cv_loss
	all_cv_loss_coarse = []
	all_cv_loss_fine = []
	for i in range(ncells):
		try:
			cv_loss_fine = np.nanmean(np.load(savepath + 'y{}_validation_performance_fine.npy'.format(i)),axis=2) # (nlambda,nl1)
			cv_loss_coarse = np.nanmean(np.load(savepath + 'y{}_validation_performance.npy'.format(i)),axis=2) # (nlambda,nl1)
			all_cv_loss_coarse.append(cv_loss_coarse)
			all_cv_loss_fine.append(cv_loss_fine)
		except Exception:
			# print(traceback.format_exc())
			print('no loss found'.format(i))
			continue
	return np.array(all_cv_loss_coarse),np.array(all_cv_loss_fine)

def get_significant_influence(savepath,plotpath,sham_stims,p_value=0.05,mean_adjust=True):
	# load sham weights
	glm_weights = np.load(savepath + 'glm_weights_sham_cv_all.npy') # (K,nstim,ncell)
	K,nstim,ncell = glm_weights.shape
	sham_stimid = sham_stims.astype(int)-np.ones_like(sham_stims).astype(int) + 16*np.ones_like(sham_stims).astype(int)
	print('sham_stimid',sham_stimid)
	sham_weights = glm_weights[:,sham_stimid.astype(int),:] # (K,nshams,ncell)

	# load target weights
	# get target distribution (after 16 gratings, everything except for sham)
	target_stimid = np.delete(np.arange(nstim),sham_stimid)[16:-1]
	print('target_stimid',target_stimid)
	glm_weights = np.load(savepath + 'glm_weights_final.npy') # (ncell,nstim)
	target_weights = glm_weights[:,target_stimid.astype(int)] # (ncell,ntargets)

	# adjust mean
	if mean_adjust:
		_, ntarget = target_weights.shape
		_, nshams, _ = sham_weights.shape
		avg_sham = np.nanmean(sham_weights,axis=(0,1)) # (ncell,)
		target_weights = target_weights - np.repeat(avg_sham,ntarget).reshape(ncell,ntarget)
		sham_weights = sham_weights  - np.tile(avg_sham,(K,nshams,1)).reshape(K,nshams,ncell)

	
	#plot sham distribution
	plotpath = plotpath + 'debug/'
	fig = plt.figure(figsize=(6,6))
	for i in range(len(sham_stims)):
		plt.hist(sham_weights[:,i,:].flatten(),label='sham {}'.format(i))
	plt.legend()
	plt.xlabel('Sham weights')
	plt.ylabel('Number of weights')
	plt.title('Sham weights distribution')
	plt.savefig(plotpath + 'sham_weights_distribution.png',bbox_inches='tight')
	plt.close(fig)

	"""# show sham weights
	fig = plt.figure(figsize=(6,6))
	plt.imshow(np.nanmean(sham_weights,axis=0), aspect='auto',cmap='bwr',interpolation='none')
	plt.colorbar()
	plt.savefig(plotpath + 'sham_weights.png',bbox_inches='tight')
	plt.close(fig)

	# show target weights
	fig = plt.figure(figsize=(6,6))
	plt.imshow(target_weights, aspect='auto',cmap='bwr',interpolation='none')
	plt.colorbar()
	plt.savefig(plotpath + 'target_weights.png',bbox_inches='tight')
	plt.close(fig)

	# plot target distribution
	fig = plt.figure(figsize=(6,6))
	for i in range(len(target_stimid)):
		plt.hist(target_weights[:,i].flatten(),label='target {}'.format(i))
	plt.legend()
	plt.xlabel('Target weights')
	plt.ylabel('Number of weights')
	plt.title('Target weights distribution')
	plt.savefig(plotpath + 'target_weights_distribution.png',bbox_inches='tight')
	plt.close(fig)

	# plot sham and target distribution
	fig = plt.figure(figsize=(6,6))
	plt.hist(sham_weights.flatten(),label='sham',alpha=0.5)
	plt.hist(target_weights.flatten(),label='target',alpha=0.5)
	plt.legend()
	plt.xlabel('Weights')
	plt.ylabel('Number of weights')
	plt.title('Sham and target weights distribution')
	plt.savefig(plotpath + 'sham_target_weights_distribution.png',bbox_inches='tight')
	plt.close(fig)"""

	# get significant weights
	thresholds = find_significant_thresholds(sham_weights.flatten(), p_value=p_value) # (lower,upper)
	significant_target_weights = target_weights.copy()
	significant_target_weights[(target_weights > thresholds[0]) & (target_weights < thresholds[1])] = np.nan
	return significant_target_weights

def get_pvalue_of_influence(savepath,sham_stims,mean_adjust=True):
	# load sham weights
	glm_weights = np.load(savepath + 'glm_weights_sham_cv_all.npy') # (K,nstim,ncell)
	K,nstim,ncell = glm_weights.shape
	sham_stimid = sham_stims.astype(int)-np.ones_like(sham_stims).astype(int) + 16*np.ones_like(sham_stims).astype(int)
	print('sham_stimid',sham_stimid)
	sham_weights = glm_weights[:,sham_stimid.astype(int),:] # (K,nshams,ncell)

	# load target weights
	# get target distribution (after 16 gratings, everything except for sham)
	target_stimid = np.delete(np.arange(nstim),sham_stimid)[16:-1]
	print('target_stimid',target_stimid)
	glm_weights = np.load(savepath + 'glm_weights_final.npy') # (ncell,nstim)
	target_weights = glm_weights[:,target_stimid.astype(int)] # (ncell,ntargets)
	# target_weights[target_weights == 0] = np.nan
	
	# adjust mean
	_, ntarget = target_weights.shape
	_, nshams, _ = sham_weights.shape
	if mean_adjust:
		
		avg_sham = np.nanmean(sham_weights,axis=(0,1)) # (ncell,)
		target_weights = target_weights - np.repeat(avg_sham,ntarget).reshape(ncell,ntarget)
		sham_weights = sham_weights  - np.tile(avg_sham,(K,nshams,1)).reshape(K,nshams,ncell)

	# get p-value
	p_values = np.zeros((ncell,ntarget))
	for i in range(ntarget):
		ps = get_p_values(sham_weights.flatten(),target_weights[:,i])
		p_values[:,i] = ps
		
	return p_values # (ncell,ntarget)






if __name__ == '__main__':
	pass 
	# data_id = 22
	# # Set the path
	# # run_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
	# run_id = '2024-05-31_14-09-08'
	# datapath = os.path.expanduser("~") + "/Downloads/kaschube-lab/Influence_mapping/"
	# savepath = os.path.expanduser("~") + "/Downloads/kaschube-lab/Influence_mapping/analysis_data/{}/shared_data{}/".format(run_id,data_id)
	# plotpath = os.path.expanduser("~") + "/Downloads/kaschube-lab/Influence_mapping/plots/{}/shared_data{}/".format(run_id,data_id)

	# if not os.path.exists(savepath):
	# 	os.makedirs(savepath)
	# if not os.path.exists(plotpath):
	# 	os.makedirs(plotpath)

	
	# # Load the data using h5py
	# with h5py.File(datapath + 'ShareData_{}.mat'.format(data_id), 'r') as f:
	# 	# Check the keys
	# 	print(f.keys())

	# 	print(f['ShareData']['VisStim'])
	# 	print(f['ShareData']['Response_dFF'])
	# 	print(f['ShareData']['Response_inferredSpikes'])
	# 	print(f['ShareData']['photoStim'])
	# 	print(f['ShareData']['UseableNonTargetCells'])
	# 	print(f['ShareData']['ShamLocations'])

	# 	sham_stims = f['ShareData']['ShamLocations'][()][:,0] # (n_shams,)
	# 	target_cells = f['ShareData']['TargetCells'][()][:,0] # (n_real_stims,)


	# 	y = f['ShareData']['Response_inferredSpikes'][()] # (ntrial,ncell)
	# 	ntrial,ncell = y.shape
	# 	print('shape of y',y.shape)

	# 	x1 = f['ShareData']['VisStim'][()] # (ntrial,n_v_stim)
	# 	num_trial_per_vstim = np.sum(x1,axis=0)
	# 	print('visual num_trial_per_stim',num_trial_per_vstim)
	# 	x2 = f['ShareData']['photoStim'][()] # (ntrial,n_o_stim)
	# 	num_trial_per_ostim = np.sum(x2,axis=0)
	# 	print('opto num_trial_per_stim',num_trial_per_ostim)
	# 	bias = np.ones((ntrial,1))
	# 	X = np.concatenate((x1,x2,bias),axis=1) # (ntrial,n_v_stim+n_o_stim)
	# 	print('shape of x',X.shape)
	
	
	




		
	


	

