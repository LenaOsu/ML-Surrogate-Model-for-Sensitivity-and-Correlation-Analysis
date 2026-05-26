import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv



bestfit_dict = {}
sigma_dict = {}

def normalize(name):
    
    if "ND280" in name:
        return name.replace("ND280 Detector Systematics/","")
    if "Flux" in name:
        return name.replace("Flux Systematics/","")
    if "Cross-Section (binned) Systematics" in name:
        return name.replace("Cross-Section (binned) Systematics/","")
    if "Cross-Section Systematics" in name:
        return name.replace("Cross-Section Systematics/","")
    return name


def extract_systematics_names(systematics, root_file):

    for sys_name in systematics:

        path = f"FitterEngine/postFit/Hesse/errors/{sys_name}/values/postFitErrors_TH1D"
        scan = root_file[path]

        values = np.array(scan.values()) # central values ~ bestfit values
        errors = None

        try:
            errors = np.array(scan.errors())
        except:
            errors = np.sqrt(scan.variances())

        labels = scan.axes[0].labels()

        print(f"{sys_name} -> {len(labels)} params")

        for name, val, err in zip(labels, values, errors):

            n = normalize(name)
            key = (sys_name, n)

            bestfit_dict[key] = val # bestfit values
            sigma_dict[key] = err # uncertainties

    return bestfit_dict, sigma_dict

def covariance_matrix_extraction(root_file, bestfit_dict, ):

    # Next, we will need to extract the covariance matrix stocked in the root file or yours
    cov_obj = root_file[
        "FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D"# cov matrix without PCA in first place
    ]

    cov = cov_obj.values()

    labels = list(cov_obj.axes[0].labels())
    assert labels == list(cov_obj.axes[1].labels())
    cov_labels = [normalize(n) for n in labels]

    mean = []
    indices = []
    matched_keys = []

    bestfit_lookup = {} #a dictionnary to keep same parameters as in the cov matrix, so they match

    for k in bestfit_dict:
        bestfit_lookup[k[1]] = k # we make correspond the name of the parameter to its quantity, so k[1]~ param. name, k~(quantity, param. name)

    for i, cov_name in enumerate(cov_labels):

        if cov_name in bestfit_lookup: #names which are actually matching
            key = bestfit_lookup[cov_name]

            mean.append(bestfit_dict[key])# we rebuild in the same order
            indices.append(i)
            matched_keys.append(key)

    mean = np.array(mean) #converted in numpy array
    cov_reduced = cov[np.ix_(indices, indices)]

    print("\nmean shape:", mean.shape)
    print("cov shape:", cov_reduced.shape)

    print("cov symmetric:", np.allclose(cov_reduced, cov_reduced.T))# check cov symmetry

    eigvals = np.linalg.eigvals(cov_reduced)
    print("cov positive definite:", np.all(eigvals >= 0))# check cov positive definite

    return mean, cov_reduced, matched_keys

def modify_covariance(cov, corr_strength=1.0, scale=1.0):# the corr strenght highlights the new correlation (1.0 = same corr, 0.0 = none or diag matrix)

    # corr matrix extraction from cov
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    # modification of corr
    corr_mod = corr_strength * corr + (1 - corr_strength) * np.eye(len(cov))

    # reconstruction
    cov_mod = corr_mod * np.outer(std, std)

    # global scaling
    cov_mod *= scale

    return cov_mod

def sample_theta(mean, cov, n):

    return np.random.multivariate_normal(mean, cov, size=n)

def propagate_to_hist(y_nominal, theta, mean, R):

    delta = (theta - mean)

    shift = R @ delta   

    max_factor = np.maximum(1+shift, 0)

    #return y_nominal * (1 + shift)
    return y_nominal * max_factor
    #return y_nominal*np.exp(shift)

def histogram_prediction_pseudo_data(predictions, samples, variables, reactions, mean, cov):
    
    R_dict = {}

    for s in samples:
        for v in variables:
            for r in reactions:

                key = (s, v, r)

                y = predictions[s][v][r]["values"]

                if key not in R_dict:
                    n_bins = len(y)
                    n_params = len(mean)
                    R_dict[key] = np.random.normal(0, 0.02, size=(n_bins, n_params))

                R = R_dict[key]

                theta_samples = sample_theta(mean, cov, 1000)
                print("theta_samples:",len(theta_samples))

                y_mouch = []

                for theta in theta_samples:
        
                    yy = propagate_to_hist(y, theta, mean, R) 
                    y_mouch.append(yy)

                y_mouch = np.array(y_mouch)
                print(len(y_mouch))

                Bins = np.arange(len(y))

                plt.figure(figsize=(8,5))

                for i in range(y_mouch.shape[0]):
                    plt.scatter(Bins, y_mouch[i], color='blue', alpha=0.05)

                mean_bins = y_mouch.mean(axis=0)
                std_bins  = y_mouch.std(axis=0)

    return mean_bins, std_bins, y_mouch

def Mahalanobis_distance(cov, X, mean, Y_norm, Y_base, X_pca):

    print("distance start to be calculated.")

    cov_inv = inv(cov)
    diff = X - mean
    dist = np.sqrt(np.einsum('ij,jk,ik->i', diff, cov_inv, diff))  # Distance de Mahalanobis
    
    idx_sort = np.argsort(dist)
                
    X_sorted = X[idx_sort]
    Y_sorted = Y_norm[idx_sort]
    X_pca = X_pca[idx_sort]
    dist = dist[idx_sort]
                
    n_bins = Y_base.shape[1]
            
    bins = np.arange(n_bins)                

    return bins, X_sorted, Y_sorted, dist, cov_inv, diff

