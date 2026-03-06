import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
import h5py 
import argparse
import warnings

from influence_helpers import construct_X_y

def weight_to_fr(params, X):
	"""
	Creates a GLM model from saved parameters and makes predictions on new data.

	Parameters:
		param_file (np arr): containing the saved GLM parameters. (nstim,)
		X (numpy.ndarray): The new design matrix for prediction. 

	Returns:
		numpy.ndarray: Predicted values.
	"""

	# Ensure the design matrix has a constant column if necessary
	if not np.all(X[:, 0] == 1):
		X = sm.add_constant(X)

	# Initialize a dummy GLM model to attach the parameters
	# Use a simple placeholder for the initialization
	dummy_y = np.ones(X.shape[0])  # Dummy target variable
	model = sm.GLM(dummy_y, X, family=sm.families.Poisson())

	# Attach the saved parameters to the model
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
        
		model_results = model.fit()
		model_results.params[:] = params

		# Make predictions with the model using the new design matrix
		rates = model_results.predict(X) #(ntrials,)

		all_delta = []
		all_perc_delta = []
		nstim = params.shape[0]
		for target in range(nstim):
			model1 = sm.GLM(dummy_y, X, family=sm.families.Poisson())
			model_results = model1.fit()
			params1 = params.copy()
			params1[target] = 0
			model_results.params[:] = params1
			rates1 = model_results.predict(X) #(ntrials,)
			delta_rates = rates - rates1
			perc_delta = delta_rates/rates1
			all_delta.append(delta_rates)
			all_perc_delta.append(perc_delta)

	return np.array(all_delta),np.array(all_perc_delta)


if __name__ == '__main__':
	# parser = argparse.ArgumentParser('Parse input parameters')
	# parser.add_argument("--dataset", action='store',dest="data_id",type=int,help="FOV")
	# args = parser.parse_args()
	# data_id = args.data_id
	# datasets = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
	# 		 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
	# 		 31,32,33,34]
	# datasets = [24,25
	# 		 ]
	datasets = [31]


	run_id = '2024-11-02'
	interaction =  'direction_single'
	npil = True
	bias = True

	
	for data_id in datasets:
		print("-----------converting FOV {} ------------".format(data_id))
		datapath = "/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/shareData/"
		savepath = "/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/analysis_data/{}_noreg/shared_data{}/sig/global/".format(run_id,data_id)
		plotpath = "/scratch/dldevel/kong/Downloads/kaschube-lab/Influence_mapping/plots/{}_noreg/shared_data{}/w2rates/".format(run_id,data_id)
		if not os.path.exists(plotpath):
				os.makedirs(plotpath)

		pp = PdfPages(plotpath + 'w2rates_percent_K_{}.pdf'.format(data_id))
		pp0 = PdfPages(plotpath + 'w2rates_K_{}.pdf'.format(data_id))
		pp1 = PdfPages(plotpath + 'w_vs_rates_percent_K_{}.pdf'.format(data_id))
		pp2 = PdfPages(plotpath + 'w_vs_rates_K_{}.pdf'.format(data_id))
		# dir_ls = np.arange(0,16)
		dir_ls = [15]
		# dir_ls = [5,7]
		for dir in dir_ls:
			try:
				weights = np.load(savepath + 'dir{}/glm_weights_sham_cv_all.npy'.format(dir))
				print('shape of saved weights',weights.shape)
			except:
				print('Error: fov{} dir{} weights does not exist'.format(data_id,dir))
				continue
			try:
				all_rates = np.load(savepath + 'dir{}/delta_rates_K_all.npy'.format(dir))
				all_perc_rates = np.load(savepath + 'dir{}/percent_delta_rates_K_all.npy'.format(dir))
				print('shape of saved rates',all_rates.shape)
				print('nans in rates',np.sum(np.isnan(all_rates)))
				print('inf in rates',np.sum(np.isinf(all_rates)))
				print('shape of saved percent rates',all_perc_rates.shape)
				print('nans in percent rates',np.sum(np.isnan(all_perc_rates)))
				print('inf in percent rates',np.sum(np.isinf(all_perc_rates)))
				continue
			except:
				print('fov{} dir{} not converted, converting ......'.format(data_id,dir))
				

			K = weights.shape[0]
			all_rates = []
			all_perc_rates = []
			for rep in range(K):
				# w = np.nanmean(weights,axis=0) # (ncell,nstim)
				w = weights[rep,:,:] # (ncell,nstim)
				ncell,nstim = w.shape
				X,y = construct_X_y(datapath + 'ShareData_{}.mat'.format(data_id),bias,interaction,dir=dir,drift=True,neurophil=npil,verbose=True) # (ntrial,nstim), (ntrial,ncell)

				
				
				all_delta = []
				all_perc_delta = []
				for icell in range(ncell):
					delta,percent_delta = weight_to_fr(w[icell,:], X) # (nstim,ntrial)
					delta_avg = np.nanmean(delta,axis=1) # (nstim,)
					percent_delta_avg = np.nanmean(percent_delta,axis=1)
					all_delta.append(delta_avg)
					all_perc_delta.append(percent_delta_avg)
					

				all_delta = np.array(all_delta) # (ncell,nstim)
				all_perc_delta = np.array(all_perc_delta)

				all_rates.append(all_delta)
				all_perc_rates.append(all_perc_delta)


			all_rates = np.array(all_rates) # (K,ncell,nstim)
			all_perc_rates = np.array(all_perc_rates) # (K,ncell,nstim)
			np.save(savepath + 'dir{}/delta_rates_K_all.npy'.format(dir),all_rates)
			np.save(savepath + 'dir{}/percent_delta_rates_K_all.npy'.format(dir),all_perc_rates)


			## compare the percent delta rates with weights
			fig,ax = plt.subplots(1,2,figsize=(10,5))
			im = ax[0].imshow(np.nanmean(weights,axis=0),aspect='auto',interpolation='none', cmap='RdBu_r',norm=colors.TwoSlopeNorm(vcenter=0))
			ax[0].set_title('dir{} weights'.format(dir))
			ax[0].set_xlabel('stimulus')
			ax[0].set_ylabel('cell')
			ax[0].set_xticks(np.arange(0,nstim,1))
			fig.colorbar(im,ax=ax[0])

			im1 = ax[1].imshow(np.nanmean(all_perc_rates,axis=0),aspect='auto',interpolation='none',cmap='RdBu_r',norm=colors.TwoSlopeNorm(vcenter=0))
			ax[1].set_title('dir{} delta rates'.format(dir))
			ax[1].set_xlabel('stimulus')
			ax[1].set_ylabel('cell')
			ax[1].set_xticks(np.arange(0,nstim,1))
			fig.colorbar(im1,ax=ax[1])
			plt.tight_layout()
			pp.savefig(fig)
			plt.close(fig)

			## compare the delta rates with weights
			fig0,ax0 = plt.subplots(1,2,figsize=(10,5))
			im0 = ax0[0].imshow(np.nanmean(weights,axis=0),aspect='auto',interpolation='none', cmap='RdBu_r',norm=colors.TwoSlopeNorm(vcenter=0))
			ax0[0].set_title('dir{} weights'.format(dir))
			ax0[0].set_xlabel('stimulus')
			ax0[0].set_ylabel('cell')
			ax0[0].set_xticks(np.arange(0,nstim,1))
			fig0.colorbar(im0,ax=ax0[0])

			im2 = ax0[1].imshow(np.nanmean(all_rates,axis=0),aspect='auto',interpolation='none',cmap='RdBu_r',norm=colors.TwoSlopeNorm(vcenter=0))
			ax0[1].set_title('dir{} delta rates'.format(dir))
			ax0[1].set_xlabel('stimulus')
			ax0[1].set_ylabel('cell')
			ax0[1].set_xticks(np.arange(0,nstim,1))
			fig0.colorbar(im2,ax=ax0[1])
			plt.tight_layout()
			pp0.savefig(fig0)
			plt.close(fig0)




			## correlate weights with delta rates
			fig1 = plt.figure()
			plt.scatter(weights.flatten(),all_perc_rates.flatten())
			plt.xlabel('weights')
			plt.ylabel('delta rates')
			plt.title('dir{} '.format(dir))
			pp1.savefig(fig1)
			plt.close(fig1)

			fig2 = plt.figure()
			plt.scatter(weights.flatten(),all_rates.flatten())
			plt.xlabel('weights')
			plt.ylabel('delta rates')
			plt.title('dir{} '.format(dir))
			pp2.savefig(fig2)
			plt.close(fig2)




		pp.close()
		pp0.close()
		pp2.close()
		pp1.close()
		print('done fov{}'.format(data_id))

			








