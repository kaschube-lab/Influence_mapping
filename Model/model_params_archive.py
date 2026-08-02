"""
Archive of parameter presets explored during development.

These parameter configurations were used during the research process but did
NOT appear in the final paper figures. They are kept here for reference.

The presets used in the paper are in model_params.py:
  - None / 'default'  : default Mexican-hat ring attractor
  - 'eg_mh'           : example MH connectivity (Fig. 4b)
  - 'eg_cross_pop'    : example cross-population connectivity (Fig. 4c)
"""

import torch
import numpy as np
try:
	from .ring import ff_connections, recurrent_connections, RingModel
except ImportError:
	from ring import ff_connections, recurrent_connections, RingModel


def archived_params(kw, unit_ff=False):
	"""
	Return archived (non-figure) parameter presets.

	Available presets:
	  'no_norm', 'weak_mh', 'low_self_inhibition', 'strong_ei', 'strong_eie',
	  'strong_eie_weak_rec', 'analytic', 'rand_cross_pop', 'rand_mh',
	  'border', 'eg1_cd', 'eg2_cd', 'eg1_mh', 'backup', 'backup1', 'backup2'
	"""

	if kw == 'no_norm':
		N = 100
		sigma_w = 5
		inh_a = 2
		sigma_m = 5
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		w = 0.8 * ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 1, 'wie': 1, 'wei': 1, 'wii': 1,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'weak_mh':
		N = 100
		sigma_w = 10
		inh_a = 2
		sigma_m = 5
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		w = 0.8 * ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': 0.5,
			'wee': 1, 'wie': 1, 'wei': 1, 'wii': 1,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'low_self_inhibition':
		N = 100
		sigma_w = 10
		inh_a = 2
		sigma_m = 5
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		w = 0.8 * ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 0.01, 'wie': 0.32, 'wei': 0.32, 'wii': 0.1,
			'noise_level': 0, 'sigma_ie': 1, 'sigma_ii': 2, 'sigma_ei': 2,
			'ii_ap': 0.11, 'ii_an': 0.1, 'ii_sp': sigma_m, 'ii_sn': sigma_m * 0.8,
		}
		m = recurrent_connections(N, rtype='noisy_MH_lile_inh', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'strong_ei':
		N = 100
		sigma_w = 15
		inh_a = 1.8
		sigma_m = 5
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		w = ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 0.01, 'wie': 0.15, 'wei': 0.4, 'wii': 0.05,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'strong_eie':
		N = 100
		sigma_w = 15
		inh_a = 1.8
		sigma_m = 5
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		w = ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 0.5, 'wie': 2, 'wei': 2, 'wii': 0.5,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'strong_eie_weak_rec':
		N = 100
		sigma_w = 10
		inh_a = 2.5
		sigma_m = 5
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		w = ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 0.1, 'wie': 0.8, 'wei': 0.8, 'wii': 0.1,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'analytic':
		N = 100
		sigma_w = 5
		inh_a = 2
		sigma_m = 5
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		w = ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 0.1, 'wie': 0.8, 'wei': 0.8 / inh_a, 'wii': 0.1 / inh_a,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'rand_cross_pop':
		N = 100
		inh_a = 1.9
		sigma_m = 18
		sigma_w = sigma_m
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		if unit_ff:
			w = 2 * ff_connections(N, fftype='uniform', params=ffparams)
		else:
			w = 2 * ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 1.1, 'wie': 3.3, 'wei': 3.3, 'wii': 2.5,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='rand_MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'rand_mh':
		N = 100
		inh_a = 1.9
		sigma_m = 18
		sigma_w = sigma_m
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		if unit_ff:
			w = 2 * ff_connections(N, fftype='uniform', params=ffparams)
		else:
			w = 2 * ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 2.4, 'wie': 2.48, 'wei': 2.48, 'wii': 2.5,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='rand_MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'border':
		N = 100
		sigma_w = 12
		inh_a = 1.8
		sigma_m = 10
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		w = ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 1.3, 'wie': 2.6, 'wei': 2.6, 'wii': 2.5,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
			'border_flag': True,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'eg1_cd':
		N = 100
		inh_a = 1.5
		sigma_m = 2
		sigma_w = sigma_m
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		if unit_ff:
			w = 2 * ff_connections(N, fftype='uniform', params=ffparams)
		else:
			w = 2 * ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 1.1, 'wie': 2.3, 'wei': 2.3, 'wii': 1.12,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'eg2_cd':
		N = 100
		inh_a = 1.5
		sigma_m = 2
		sigma_w = sigma_m
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		if unit_ff:
			w = 2 * ff_connections(N, fftype='uniform', params=ffparams)
		else:
			w = 2 * ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 0.9, 'wie': 2.5, 'wei': 2.5, 'wii': 1.15,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'eg1_mh':
		N = 100
		inh_a = 1.5
		sigma_m = 2
		sigma_w = sigma_m
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		if unit_ff:
			w = 2 * ff_connections(N, fftype='uniform', params=ffparams)
		else:
			w = 2 * ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 1.1, 'wie': 0.9, 'wei': 0.9, 'wii': 1.12,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'backup':
		N = 100
		inh_a = 1.5
		sigma_m = 7
		sigma_w = sigma_m
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		if unit_ff:
			w = ff_connections(N, fftype='uniform', params=ffparams)
		else:
			w = ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 1.01, 'wie': 3, 'wei': 3, 'wii': 1.5,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'backup1':
		N = 100
		inh_a = 1.5
		sigma_m = 6.5
		sigma_w = sigma_m
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		if unit_ff:
			w = ff_connections(N, fftype='uniform', params=ffparams)
		else:
			w = ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 1.01, 'wie': 1.5, 'wei': 1.5, 'wii': 2,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	elif kw == 'backup2':
		N = 100
		inh_a = 1.5
		sigma_m = 11
		sigma_w = sigma_m
		ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
		if unit_ff:
			w = ff_connections(N, fftype='uniform', params=ffparams)
		else:
			w = ff_connections(N, fftype='Gaussian', params=ffparams)
		rec_params = {
			'N': N, 'npop': 2, 'sigma': sigma_m, 'r': None,
			'wee': 1.01, 'wie': 2, 'wei': 2, 'wii': 1.02,
			'sigma_ie': sigma_m, 'sigma_ii': sigma_m * inh_a, 'sigma_ei': sigma_m * inh_a,
		}
		m = recurrent_connections(N, rtype='MH', params=rec_params)
		return {
			'N': N, 'W': torch.tensor(m, dtype=torch.float32),
			'W_ff': torch.tensor(w, dtype=torch.float32), 'npop': 2, 'sigma': sigma_m,
		}

	else:
		raise ValueError(f"Unknown archived preset: '{kw}'")


def vanilla_ring():
	"""Classic single-population ring attractor (exploratory; not in paper)."""
	N = 100
	sigma_w = 10
	inh_a = 2
	sigma_m = 5
	ffparams = {'N': N, 'npop': 2, 'sigma': sigma_w}
	w = 0.8 * ff_connections(N, fftype='Gaussian', params=ffparams)
	rec_params = {
		'N': N, 'npop': 2, 'sigma': sigma_m, 'r': 0.9,
		'wee': 1, 'wie': 1, 'wei': 1 / inh_a, 'wii': 1 / inh_a,
		'sigma_ie': 5, 'sigma_ii': 5 * inh_a, 'sigma_ei': 5 * inh_a,
	}
	m = recurrent_connections(N, rtype='MH', params=rec_params)
	import torch
	network_params = {
		'N': N, 'npop': 2, 'tau': 0.01, 'f': torch.sigmoid,
		'W': torch.tensor(m, dtype=torch.float32),
		'W_ff': torch.tensor(w, dtype=torch.float32),
	}
	return RingModel(network_params)
