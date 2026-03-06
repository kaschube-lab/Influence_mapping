"""
basic helper functions

Created on 2024-04-23
Created by: Deyue
"""

import numpy as np
import scipy.stats as stats

def sort_together(arr1, arr2):
	"""
	Sort two arrays together, return idex
	"""
	idx = np.argsort(arr1)
	return arr1[idx], arr2[idx], idx


def rescale_array(arr1, arr2):
    mean_arr1 = np.mean(arr1)
    mean_arr2 = np.mean(arr2)
    range_arr1 = np.max(arr1) - np.min(arr1)
    range_arr2 = np.max(arr2) - np.min(arr2)
    
    scaled_arr1 = ((arr1 - mean_arr1) / range_arr1) * range_arr2 + mean_arr2
    
    return scaled_arr1

def get_pairwise_distances(X):
    """
    Get pairwise distances between all points in X
    """
    n = X.shape[0]
    dist = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            dist[i,j] = np.linalg.norm(X[i] - X[j])
    return dist

def sort_into_distance_bins(dist,resp,dist_bins):
    """
    Sort distances into n_bins, tgether with resp 
    dist: 1d array of distances
    resp: 1d array of responses
    dist_bins: array of bin edges

    returns:
    sorted_resp: list of arrays of responses
    avg_resp: list of average responses
    """
    sorted_resp = []
    avg_resp = []
    std_resp = []
    for i in range(len(dist_bins)-1):
        bin_idx = np.where((dist >= dist_bins[i]) & (dist < dist_bins[i+1]))[0]
        sorted_resp.append(resp[bin_idx])
        avg_resp.append(np.nanmean(resp[bin_idx]))
        std_resp.append(np.nanstd(resp[bin_idx]))
    return sorted_resp, np.array(avg_resp), np.array(std_resp)

    
def find_significant_thresholds(arr, p_value=0.05):
    # remove nans
    arr = arr[~np.isnan(arr)]
    # Calculate mean and standard deviation
    mean = np.mean(arr)
    std_dev = np.std(arr)
    
    # For a two-tailed test, we need to divide the p-value by 2
    # z_score = stats.norm.ppf(1 - p_value / 2)
    z_score = stats.norm.ppf(1 - p_value ) 
    
    # Calculate the upper and lower thresholds
    lower_threshold = mean - z_score * std_dev
    upper_threshold = mean + z_score * std_dev
    
    return lower_threshold, upper_threshold

def get_p_values(dist, arr):
    # Remove nans
    dist = dist[~np.isnan(dist)]
    # Calculate mean and standard deviation of the distribution
    mean = np.mean(dist)
    std_dev = np.std(dist)
    
    # Function to calculate the p-value for a single value
    def p_value(x):
        z_score = (x - mean) / std_dev
        # Two-tailed p-value
        # p_val = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
        p_val = 1 - stats.norm.cdf(np.abs(z_score))
        return p_val
    
    # Calculate p-values for all values in arr
    p_values = np.array([p_value(x) for x in arr])
    
    return p_values

