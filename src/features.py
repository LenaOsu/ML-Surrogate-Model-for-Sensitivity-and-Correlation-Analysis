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

def build_prediction(samples, variables, reactions, predictions):


    output = {}


    for s in samples:
        output[s] = {}


        for v in variables:
            output[s][v] = {}


            for r in reactions:


                y = predictions[s][v][r]["values"]
                output[s][v][r] = y


    return output

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

def build_prediction(predictions, samples, variables, reactions):
    output = {}

    for s in samples:
        output[s] = {}

        for v in variables:
            output[s][v] = {}

            for r in reactions:
                output[s][v][r] = predictions[s][v][r]["values"]

    return output

def make_base_fn(s, v, r, predictions):

    y0 = predictions[s][v][r]["values"]

    def fn(theta, mean):
        # modèle simple cohérent
        delta = theta - mean
        return y0 * (1 + 0.01 * np.mean(delta))

    return fn

def compute_R_per_hist(mean, sigma, samples, variables, reactions, predictions):
    
    R_dict = {}

    for s in samples:
        s = normalize(s)
        for v in variables:
            v = normalize(v)
            for r in reactions:
                r = normalize(r)

                fn = make_base_fn(s, v, r, predictions)

                y0 = fn(mean, mean) # prediction at bestfit values, should be close to nominal prediction
                n_bins = len(y0)
                n_params = len(mean)

                R = np.zeros((n_bins, n_params))

                for j in range(n_params):

                    theta_p = mean.copy()
                    theta_m = mean.copy()

                    theta_p[j] += sigma[j]
                    theta_m[j] -= sigma[j]

                    y_p = fn(theta_p, mean)
                    y_m = fn(theta_m, mean)
                    print("y_p - y_m =", np.mean(np.abs(y_p - y_m)))

                    R[:, j] = (y_p - y_m) / (2 * sigma[j])

                R_dict[(s,v,r)] = R

    return R_dict

def propagate_to_hist(y_nominal, theta, mean, R):

    delta = theta - mean
    shift = R @ delta

    shift = np.tanh(shift)

    return y_nominal * (1 + 0.1 * shift)

def histogram_prediction_pseudo_data(
    predictions,
    R_dict,
    samples,
    variables,
    reactions,
    mean,
    cov,
    n_samples=1000
):

    output = {}

    for s in samples:
        output[s] = {}

        for v in variables:
            output[s][v] = {}

            for r in reactions:

                y = predictions[s][v][r]["values"]
                R = R_dict[(s, v, r)]

                theta_samples = sample_theta(mean, cov, n_samples)

                y_list = []

                for theta in theta_samples:
                    y_list.append(propagate_to_hist(y, theta, mean, R))

                y_list = np.array(y_list)

                output[s][v][r] = {
                    "pseudo": y_list,
                    "mean": y_list.mean(axis=0),
                    "std": y_list.std(axis=0)
                }

    return output

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

