"""
Fit the GLM model to the data using cross-validation to find best hyperparameters

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
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dataset import Dataset
from helper import sort_together
from influence_helpers import construct_X_y, K_CV_coarse_pass,K_CV_fine_pass #, K_session_cross_validation,

plt.rcParams.update({'font.size': 22})

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Parse input parameters')
	parser.add_argument("--dataset", action='store',dest="data_id",type=int,help="FOV")
	parser.add_argument("--run_id", action='store',dest="run_id",type=str,help="which job batch, where to save data")
	parser.add_argument("--interaction", action='store',dest="interaction",type=str,help="Interaction between visual and opto stim: none, contrast, etc.")
	parser.add_argument("--baseline_correct", action='store',dest="baseline_correct",type=str,help="Baseline correction: cell, global") 
	args = parser.parse_args()
	data_id = args.data_id # 9#22 #9
	run_id = args.run_id
	interaction = args.interaction
	baseline_correct = args.baseline_correct
	if baseline_correct is None:
		baseline_correct = 'global'
	# Set the path
	# run_id = datetime.today().strftime("%Y-%m-%d")
	# run_id = '2024-05-31_14-09-08'
	# run_id = '2024-06-16'
	# run_id = '2024-06-19'
	datapath = os.path.expanduser("~") + "/Downloads/kaschube-lab/Influence_mapping/shareData/"
	savepath = os.path.expanduser("~") + "/Downloads/kaschube-lab/Influence_mapping/analysis_data/{}/shared_data{}/".format(run_id,data_id)
	plotpath = os.path.expanduser("~") + "/Downloads/kaschube-lab/Influence_mapping/plots/{}/shared_data{}/".format(run_id,data_id)

	if not os.path.exists(savepath):
		os.makedirs(savepath)
	if not os.path.exists(plotpath):
		os.makedirs(plotpath)

	
	# Load the data using h5py
	with h5py.File(datapath + 'ShareData_{}.mat'.format(data_id), 'r') as f:
		# Check the keys
		print(f['ShareData'].keys())

		print(f['ShareData']['VisStim'])
		print(f['ShareData']['Response_dFF'])
		print(f['ShareData']['Response_inferredSpikes'])
		print(f['ShareData']['photoStim'])
		print(f['ShareData']['UseableNonTargetCells'])
		print(f['ShareData']['ShamLocations'])

		sham_stims = f['ShareData']['ShamLocations'][()][:,0] # (n_shams,)
		target_cells = f['ShareData']['TargetCells'][()][:,0] # (n_real_stims,)


	X,y = construct_X_y(datapath + 'ShareData_{}.mat'.format(data_id),interaction,verbose=True) # (ntrial,nstim), (ntrial,ncell)
	ntrial,ncell = y.shape
	print('shape of y',y.shape)

	
	n_lambda = 10
	l1_samples = 5

	# GLM CV
	if os.path.isfile(savepath + 'CV/avg_validation_performance_coarse.npy'):
		avg_loss_coarse = np.load(savepath + 'CV/avg_validation_performance_coarse.npy')
		lambdas = np.load(savepath + 'CV/y0_validation_lambdas.npy')
		l1_wts = np.load(savepath + 'CV/y0_validation_l1_wts.npy')
		print('Coarse pass avg loss loaded')
	else:
		print('-----------Calculating Coarse pass-----------')
		glm_weights = np.zeros((ncell,X.shape[1]))
		for i in tqdm(range(ncell)):
			y_i = y[:,i] # (ntrial,)
			cv_savepath = savepath + 'CV/'
			if not os.path.exists(cv_savepath):
				os.makedirs(cv_savepath)
			cv_plotpath = plotpath + 'CV/'
			if not os.path.exists(cv_plotpath):
				os.makedirs(cv_plotpath)
			K_CV_coarse_pass(5,cv_savepath+'y{}'.format(i),cv_plotpath+'y{}'.format(i),X,y_i,LAMBDA_SAMPLES=n_lambda,l1_samples=l1_samples)
			

		## load all cv performance from the coarse pass
		all_loss_coarse = []
		for i in range(ncell):
			validation_performance = np.load(cv_savepath+'y{}'.format(i)+'_validation_performance.npy')
			Kfold_performance = np.nanmean(validation_performance,axis=2)    # (nlambda,nl1)
			all_loss_coarse.append(Kfold_performance)
		lambdas = np.load(cv_savepath+'y{}'.format(i)+'_validation_lambdas.npy')
		l1_wts = np.load(cv_savepath+'y{}'.format(i)+'_validation_l1_wts.npy')
		all_loss_coarse = np.array(all_loss_coarse) # (ncell,nlambda,nl1)
		avg_loss_coarse = np.nanmean(all_loss_coarse,axis=0) # (nlambda,nl1)
		np.save(cv_savepath+'avg_validation_performance_coarse.npy',avg_loss_coarse)
	
	## find the minimum in avg performance in coase pass 
	lambda_min_idx,l1_wt_min_idx = np.unravel_index(np.nanargmin(avg_loss_coarse, axis=None), avg_loss_coarse.shape)
	
	## plotting coarse pass
	fig = plt.figure(figsize=(6,6))
	im = plt.imshow(avg_loss_coarse,aspect='auto',interpolation='none',origin='lower')
	plt.yticks(np.arange(len(lambdas)),[str(x) for x in lambdas] )
	plt.ylabel('lambda')
	plt.xticks(np.arange(len(l1_wts)),['{:.2f}'.format(x) for x in l1_wts])
	plt.xlabel('L1_wt')
	cbar = plt.colorbar(im)
	cbar.set_label('MSE')
	plt.scatter(l1_wt_min_idx,lambda_min_idx,marker='x',color='r')
	plt.savefig(plotpath+'avg_validation_performance_coarse.png',bbox_inches='tight')
	plt.close(fig)

	## find new params range for fine pass
	if lambda_min_idx == 0:
		lambda_lb = lambdas[0]
		lambda_ub = lambdas[lambda_min_idx+2]
	elif lambda_min_idx == len(lambdas)-1 or lambda_min_idx == len(lambdas)-2:
		lambda_lb = lambdas[lambda_min_idx-1]
		lambda_ub = lambdas[-1]
	else:
		lambda_lb = lambdas[lambda_min_idx-1]
		lambda_ub = lambdas[lambda_min_idx+2]
	## find the l1_lb and l1_ub
	if l1_wt_min_idx == 0:
		l1_lb = 0.0
		l1_ub = l1_wts[l1_wt_min_idx+2]
	elif l1_wt_min_idx == len(l1_wts)-1 or l1_wt_min_idx == len(l1_wts)-2:
		l1_lb = l1_wts[l1_wt_min_idx-1]
		l1_ub = 1.0
	else:
		l1_lb = l1_wts[l1_wt_min_idx-1]
		l1_ub = l1_wts[l1_wt_min_idx+2]

	## fine pass
	lambdas_fine = np.linspace(lambda_lb,lambda_ub,num=n_lambda)
	l1s_fine = np.linspace(l1_lb,l1_ub,num=l1_samples)
	# GLM CV
	glm_weights = np.zeros((ncell,X.shape[1]))

	print('-----------Calculating fine pass-----------')
	cv_savepath = savepath + 'CV_fine/'
	if not os.path.exists(cv_savepath):
		os.makedirs(cv_savepath)
	cv_plotpath = plotpath + 'CV_fine/'
	if not os.path.exists(cv_plotpath):
		os.makedirs(cv_plotpath)
	for i in tqdm(range(ncell)):
		y_i = y[:,i]
		K_CV_fine_pass(5,cv_savepath+'y{}'.format(i),cv_plotpath+'y{}'.format(i),X,y_i,lambdas_fine,l1s_fine)

	### load all cv performance from the fine pass
	all_loss_fine = []
	for i in range(ncell):
		validation_performance = np.load(cv_savepath+'y{}'.format(i)+'_validation_performance_fine.npy')
		Kfold_performance = np.nanmean(validation_performance,axis=2)    # (nlambda,nl1)
		all_loss_fine.append(Kfold_performance)

	all_loss_fine = np.array(all_loss_fine) # (ncell,nlambda,nl1)
	avg_loss_fine = np.nanmean(all_loss_fine,axis=0) # (nlambda,nl1)
	np.save(cv_savepath+'avg_validation_performance_fine.npy',avg_loss_fine)
	## find the optimal in avg performance in fine pass
	lambda_min_idx,l1_wt_min_idx = np.unravel_index(np.nanargmin(avg_loss_fine, axis=None), avg_loss_fine.shape)
	lambda_opt = lambdas_fine[lambda_min_idx]
	l1_wt_opt = l1s_fine[l1_wt_min_idx]
	print('optimal lambda',lambda_opt)
	print('optimal l1_wt',l1_wt_opt)

	## plotting fine pass
	fig = plt.figure(figsize=(6,6))
	im = plt.imshow(avg_loss_fine,aspect='auto',interpolation='none',origin='lower')
	plt.yticks(np.arange(len(lambdas_fine)),[str(x) for x in lambdas_fine] )
	plt.ylabel('lambda')
	plt.xticks(np.arange(len(l1s_fine)),['{:.2f}'.format(x) for x in l1s_fine])
	plt.xlabel('L1_wt')
	cbar = plt.colorbar(im)
	cbar.set_label('MSE')
	plt.scatter(l1_wt_min_idx,lambda_min_idx,marker='x',color='r')
	plt.savefig(plotpath+'avg_validation_performance_fine.png',bbox_inches='tight')
	plt.close(fig)

	# Save the weights
	# np.save(savepath + 'glm_weights.npy',glm_weights)

	
	




		
	


	

