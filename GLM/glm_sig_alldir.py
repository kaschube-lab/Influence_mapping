"""
Step 2: Use saved optimal hyperparameters, and calculate sham weights distribution

Created on: 2024-05-20
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
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dataset import Dataset
from helper import sort_together,find_significant_thresholds
from influence_helpers import construct_X_y, get_sham_trial_inx,load_best_hyperparameters_avg,sham_bootstrapping #sham_cv_split,

plt.rcParams.update({'font.size': 22})


if __name__ == '__main__':
	parser = argparse.ArgumentParser('Parse input parameters')
	parser.add_argument("--dataset", action='store',dest="data_id",type=int,help="FOV")
	parser.add_argument("--run_id", action='store',dest="run_id",type=str,help="which job batch, where to save data")
	parser.add_argument("--interaction", action='store',dest="interaction",type=str,help="Interaction between visual and opto stim: none, contrast, etc.")
	# parser.add_argument("--dir", action='store',dest="dir",type=int,help="only one direction")
	parser.add_argument("--baseline_correct", action='store',dest="baseline_correct",type=str,help="Baseline correction: cell, global")
	parser.add_argument("--drift", action='store',dest="drift",type=str,help="slow drift")
	parser.add_argument("--neurophil", action='store',dest="neurophil",type=str,help="global fluctuation")
	parser.add_argument("--timepoint", action='store',dest="timepoint",type=str,help="which time point")

	args = parser.parse_args()
	data_id = args.data_id
	run_id = args.run_id
	interaction = args.interaction

	# dir = args.dir
	baseline_correct = args.baseline_correct
	if baseline_correct is None:
		baseline_correct = 'global'
	neurophil = args.neurophil
	drift = args.drift
	timepoint = args.timepoint

	# dir_ls = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	# dir_ls = [4,5,6,7,8,9,10,11,12,13,14,15]
	# dir_ls = [0,1,2,3]
	dir_ls = [None]
	

	for dir in dir_ls:

		# data_id = 22 #9
		# Set the path
		# run_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
		# run_id = '2024-05-31_14-09-08'
		# run_id = '2024-06-16'
		if 'dldevel' in os.path.expanduser("~"):
			datapath = "/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/shareData/"
			savepath = "/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/analysis_data/{}_noreg/shared_data{}/".format(run_id,data_id)
			sig_savepath = savepath + 'sig/{}/'.format(baseline_correct) 
			plotpath = "/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/plots/{}_noreg/shared_data{}/sham_sig/".format(run_id,data_id)
		else:
			datapath = os.path.expanduser("~") + "/Downloads/kaschube-lab/Influence_mapping/shareData/"
			savepath = os.path.expanduser("~") + "/Downloads/kaschube-lab/Influence_mapping/analysis_data/{}_noreg/shared_data{}/".format(run_id,data_id)
			sig_savepath = savepath + 'sig/{}/'.format(baseline_correct) 
			plotpath = os.path.expanduser("~") + "/Downloads/kaschube-lab/Influence_mapping/plots/{}/shared_data{}/sham_sig/".format(run_id,data_id)

		if timepoint is not None:
			savepath = savepath + 'time{}/'.format(timepoint)
			sig_savepath = sig_savepath + 'time{}/'.format(timepoint)
			plotpath = plotpath + 'time{}/'.format(timepoint)
		if dir is not None:
			sig_savepath = sig_savepath + 'dir{}/'.format(dir)
			plotpath = plotpath + 'dir{}/'.format(dir)
		else:
			sig_savepath = sig_savepath + 'all/'
			plotpath = plotpath + 'all/'
		if not os.path.exists(savepath):
			os.makedirs(savepath)
		if not os.path.exists(sig_savepath):
			os.makedirs(sig_savepath)
		if not os.path.exists(plotpath):
			os.makedirs(plotpath)

		
		# Load the data using h5py
		with h5py.File(datapath + 'ShareData_{}.mat'.format(data_id), 'r') as f:
			# Check the keys
			print(f['ShareData'].keys())

			print(f['ShareData']['VisStim'])
			_, n_vstim = f['ShareData']['VisStim'][()].shape
			print('Number of visual stimuli',n_vstim)
			print(f['ShareData']['Response_dFF'])
			print(f['ShareData']['Response_inferredSpikes'])
			print(f['ShareData']['photoStim'])
			print(f['ShareData']['Sham_photoStim'])
			print(f['ShareData']['UseableNonTargetCells'])
			print(f['ShareData']['ShamLocations'])

			sham_stims = f['ShareData']['ShamLocations'][()][:,0] # (n_shams,)
			target_cells = f['ShareData']['TargetCells'][()][:,0] # (n_real_stims,)
			n_targets = len(target_cells)
			print('sham_stims',sham_stims)
			print('target_cells',target_cells)


			y = f['ShareData']['Response_inferredSpikes'][()] # (ntrial,ncell)
			ntrial,ncell = y.shape
			print('shape of y',y.shape)

		X,y = construct_X_y(datapath + 'ShareData_{}.mat'.format(data_id),\
						True,interaction,dir,neurophil=neurophil,drift=drift,verbose=True,timepoint=timepoint) # (ntrial,nstim), (ntrial,ncell)
		ntrial,ncell = y.shape
		print('shape of y',y.shape)
		


		## test significant influence
		# sham_trials,sham_indx,target_trials = split_sham_target_trials(x2,target_cells, sham_stims,y)
		# print('sham_trials',sham_trials.shape)
		# print('sham_indx',sham_indx.shape)
		# print('target_trials',target_trials.shape)

		# split the data into K folds
		K = 20
		# X_k, y_k = sham_cv_split(X, y,sham_indx, K) # (K,ntrials_per_fold,nstims), (K,ntrials_per_fold,ncells)
		# merge into only one sham stim
		
		sham_indx,_ = get_sham_trial_inx(X,target_cells ,sham_stims)
		X_k, y_k = sham_bootstrapping(X, y,sham_indx, K) # (K,ntrials_per_fold,nstims), (K,ntrials_per_fold,ncells)
		_,ntrials_per_fold,nstims = X_k.shape

		# load optimal GLM hyperparameters
		opt_params = load_best_hyperparameters_avg(savepath) # (2,) (lambda,l1_wt)
		# for testing
		# opt_params = [0.01,0.1]
		print('optimal hyperparameters',opt_params)
		if np.isnan(opt_params[0]) or np.isnan(opt_params[1]):
			print('No optimal hyperparameters found. Exiting...')
			sys.exit(0)

		# check saved weights
		
		# try:
		# 	glm_weights = np.load(sig_savepath + 'glm_weights_sham_cv_all.npy')
		# except:
		print('No saved sham_CV weights found. Calculating...')
		glm_weights = np.nan*np.ones((K,ncell,X_k.shape[2])) # (K,ncell,nstims)

		
		for jcell in tqdm(range(ncell)):
			# check if optimal parameters are loaded correctly
			if np.isnan(opt_params[0]) or np.isnan(opt_params[1]):
				print('cell {} has nan optimal GLM parameters'.format(jcell))
				glm_weights[:,jcell,:] = np.nan*np.ones((K,X_k.shape[2]))
				continue

			for i in range(K):
				# check if weights are already calculated (not all nan)
				if not np.isnan(glm_weights[i,jcell,:]).all():
					print('cell {}, repitition{} already has weights calculated'.format(jcell,i))
					continue
				
				nstims = X_k.shape[2]
				# X_i = X_k[i,:,:].reshape((-1,nstims)) # (ntrials_per_fold,nstims)
				X_i = X_k[i,:,:] # (ntrials_per_fold,nstims)
				# y_i = y_k[i,:,jcell].reshape((-1,)) # (ntrials_per_fold,)
				y_i = y_k[i,:,jcell] # (ntrials_per_fold,)
				# print('debug',X_i.shape,y_i.shape)
				
				model = sm.GLM(y_i, X_i, family=sm.families.Poisson())
				# u, s, vt = np.linalg.svd(model.exog, 0)
				# print("check if design matrix is singular:(all should be positive)",s)
				try:
					results = model.fit_regularized(method='elastic_net',
						alpha=opt_params[0],
						maxiter=1000,
						L1_wt=opt_params[1], # 1.0:all L1, 0.0: all L2
						refit=True
						)
					glm_weights[i,jcell,:] = results.params
				except Exception as e:
					print(e)
					print('Error in fitting GLM without regularization')
					glm_weights[i,jcell,:] = np.zeros(X_i.shape[1])
					continue

		glm_weights = np.array(glm_weights)# (K,nstims,ncell)
		# Save the weights
		np.save(sig_savepath + 'glm_weights_sham_cv_all.npy',glm_weights)



		# get sham distribution (after the 16 gratings)
		print('------------plotting------------')
		# if interaction == 'none':
		# 	sham_stimid = np.arange(n_vstim+n_targets,X.shape[1]-1) # (nshams,)
		# elif interaction == 'direction_single':
		# 	sham_stimid = np.arange(X.shape[1]-3)
		# print('sham_stimid',sham_stimid)
		sham_weights = glm_weights[:,:,-4] # (K,nsham=1,ncell)
		print('shape sham_weights',sham_weights.shape)

		# plot sham distribution
		fig = plt.figure(figsize=(6,6))
		
		plt.hist(sham_weights.flatten(),label='sham')
		plt.legend()
		plt.xlabel('Sham weights')
		plt.ylabel('Number of weights')
		plt.title('Sham weights distribution')
		plt.savefig(plotpath + 'sham_weights_distribution.png',bbox_inches='tight')
		plt.close(fig)

		# get target distribution (after gratings, everything except for sham)
		if interaction == 'none':
			target_stimid = np.arange(n_vstim,n_vstim+n_targets) # (ntargets,)
		elif interaction == 'contrast':
			target_stimid = np.arange(n_vstim,n_vstim+2*n_targets) # (ntargets,)
		elif interaction == 'direction_single':
			target_stimid = np.arange(0,n_targets)
		print('target_stimid',target_stimid)

		target_weights = glm_weights[:,:,target_stimid.astype(int)] # (K,ncell,ntargets)
		print('shape target_weights',target_weights.shape)
		

		# plot target distribution
		fig = plt.figure(figsize=(6,6))
		for i in range(len(target_stimid)):
			plt.hist(target_weights[:,:,i].flatten(),label='target {}'.format(i))
		plt.legend()
		plt.xlabel('Target weights')
		plt.ylabel('Number of weights')
		plt.title('Target weights distribution')
		plt.savefig(plotpath + 'target_weights_distribution.png',bbox_inches='tight')
		plt.close(fig)

		## plot avg gml weights
		avg_glm_weights = np.nanmean(glm_weights,axis=0) # (nstims,ncell)
		fig = plt.figure(figsize=(6,6))
		plt.imshow(avg_glm_weights, aspect='auto',cmap='bwr',interpolation='none',vmin=-1,vmax=1)
		plt.colorbar()
		plt.savefig(plotpath + 'avg_glm_weights.png',bbox_inches='tight')
		plt.close(fig)

		

		
		# plot significant weights
		# adjust target mean
		if baseline_correct == 'cell':
			nstim = target_weights.shape[-1]
			ncells = target_weights.shape[1]
			# target_weights = target_weights - np.repeat(sham_weights,nstim).reshape(ncells,nstim)
			# adjust sham to zero mean
			target_weights = target_weights - np.repeat(np.nanmean(sham_weights,axis=0),nstim).reshape(ncells,nstim)
			# adjust sham to zero mean
			sham_weights = sham_weights - np.nanmean(sham_weights,axis=0)
		elif baseline_correct == 'global':
			target_weights = target_weights - np.nanmean(sham_weights)
			# adjust sham to zero mean
			sham_weights = sham_weights - np.nanmean(sham_weights)
		

		## save sham weights
		np.save(sig_savepath + 'sham.npy',sham_weights) # (K,ncell,nsham)
		
		# plot sham and target weights
		fig = plt.figure(figsize=(6,6))
		plt.hist(sham_weights.flatten(),label='sham',alpha=0.5)
		plt.hist(target_weights.flatten(),label='target',alpha=0.5)
		plt.legend()
		plt.xlabel('Weights')
		plt.ylabel('Number of weights')
		plt.title('Sham and target weights distribution')
		plt.savefig(plotpath + 'sham_target_weights_distribution.png',bbox_inches='tight')
		plt.close(fig)
		
		# get significant target weights, and set the rest to nan
		thresholds = find_significant_thresholds(sham_weights.flatten(), p_value=0.05) # (lower,upper)
		print('target_weights',target_weights)
		print('thresholds',thresholds)
		significant_target_weights = target_weights.copy()
		significant_target_weights[(target_weights > thresholds[0]) & (target_weights < thresholds[1])] = np.nan

		


		
		
		# bar plot of number of significant vs in significant weights
		n_significant = np.sum(~np.isnan(significant_target_weights))
		n_insignificant = np.sum(np.isnan(significant_target_weights))
		print('# significant target weights',n_significant)
		ntotal = n_significant + n_insignificant
		fig = plt.figure(figsize=(6,6))
		plt.bar(['significant','insignificant'],[n_significant/ntotal,n_insignificant/ntotal])
		plt.xlabel('Weights')
		plt.ylabel('Number of weights')
		plt.title('Number of significant vs insignificant weights')
		plt.savefig(plotpath + 'significant_vs_insignificant_influence.png',bbox_inches='tight')

		

		# drop all nan target weights
		significant_target_weights = significant_target_weights[~np.isnan(significant_target_weights)] 

		# save the significant weights
		np.save(sig_savepath + 'significant_influence.npy',significant_target_weights) # (K,ntargets,ncell)
		# plot significant weights
		fig = plt.figure(figsize=(6,6))
		plt.hist(significant_target_weights.flatten(),label='significant target weights')
		plt.hist(sham_weights.flatten(),label='sham',alpha=0.5)
		# plot threshold as vertical lines
		plt.axvline(thresholds[0],color='r',linestyle='--',label='lower threshold')
		plt.axvline(thresholds[1],color='r',linestyle='--',label='upper threshold')
		plt.xlabel('Weights')
		plt.ylabel('Number of weights')
		plt.title('Significant target weights distribution')
		plt.legend()
		plt.savefig(plotpath + 'significant_target_weights_distribution.png',bbox_inches='tight')
		plt.close(fig)

		print('------------done with dir {}------------'.format(dir))

	













	




		
	


	

