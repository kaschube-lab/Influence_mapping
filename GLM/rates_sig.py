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


plt.rcParams.update({'font.size': 22})


def d_prime(x1, x2):
	# Calculate the means and standard deviations of the two distributions
	mean_x1 = np.mean(x1)
	mean_x2 = np.mean(x2)
	std_x1 = np.std(x1)
	std_x2 = np.std(x2)

	# Calculate d-prime using the formula
	d_prime = (mean_x1 - mean_x2) / np.sqrt((std_x1**2 + std_x2**2) / 2)
	return d_prime


if __name__ == '__main__':
	# time = ['early','mid','late']
	run_id = '2024-11-02_noreg'
	data_id_ls = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
	# data_id_ls = [31]
	dir_ls = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	# dir_ls = [15]
	# for tp in time:
	for data_id in data_id_ls:
		for dir in dir_ls:
			try:
				if 'dldevel' in os.path.expanduser("~"):
					datapath = '/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/analysis_data/{}/shared_data{}/sig/global/dir{}/'.format(run_id,data_id,dir)
				w_percent_all = np.load(datapath + 'percent_delta_rates_K_all.npy') # (ncell,nstim)
				w_all = np.load(datapath + 'delta_rates_K_all.npy') # (ncell,nstim)
				
			except:
				print('Error: not found ',run_id,data_id,dir)
				continue

			print(w_all.shape)
			sham_percent = w_percent_all[:,:,-4] # (K,ncell)
			sham = w_all[:,:,-4]

			sham_percent[sham_percent>50] = np.nan
			sham_percent[sham_percent<-50] = np.nan
			print(np.nanmean(sham_percent))
			infl_percent = w_percent_all[:,:,:-4] # exclude sham, drift, bias, npile (K,ncell,ntarget)
			infl_percent[infl_percent>50] = np.nan
			infl_percent[infl_percent<-50] = np.nan

			dprimes_percent = np.zeros((infl_percent.shape[1],infl_percent.shape[2]))
			for i in range(infl_percent.shape[1]):
				for j in range(infl_percent.shape[2]):
					dprimes_percent[i,j] = d_prime(sham_percent[:,i],infl_percent[:,i,j])

			# mean correction
			avg = np.nanmean(sham_percent)
			sham_percent = sham_percent - avg
			infl_percent = infl_percent - avg
			


			
			print('sham mean',np.mean(sham))
			infl = w_all[:,:,:-4] # exclude sham, drift, bias, npile (ncell,ntarget)
			thres = 500
			infl[infl>thres] = np.nan
			infl[infl<-thres] = np.nan
			sham[sham>thres] = np.nan
			sham[sham<-thres] = np.nan
			# mean correction
			avg = np.nanmean(sham)
			sham = sham - avg
			infl = infl - avg

			



			# Calculate the d-prime for each infl weight
			dprimes = np.zeros((infl.shape[1],infl.shape[2]))
			for i in range(infl.shape[1]):
				for j in range(infl.shape[2]):
					dprimes[i,j] = d_prime(sham[:,i],infl[:,i,j])

			infl_mean = np.mean(infl,axis=0)
			infl_mean_percent = np.mean(infl_percent,axis=0)
			print('sham shape',sham.shape)
			print('infl shape',infl_mean.shape)

			np.save(datapath + 'dprimes_rates_percent.npy',dprimes_percent)
			np.save(datapath + 'influence_rates_percent.npy',infl_mean_percent)
			np.save(datapath + 'sham_rates_percent.npy',sham_percent)

			np.save(datapath + 'dprimes_rates.npy',dprimes)
			np.save(datapath + 'influence_rates.npy',infl_mean)
			np.save(datapath + 'sham_rates.npy',sham)





