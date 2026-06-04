
import numpy as np
import pandas as pd
import os
import uproot
import matplotlib.pyplot as plt
from src.preparation import preparation
from src.extraction_data import extraction_data
from src.visualization import nominal_prediction_visualization
from src.visualization import shifted_prediction_visualization
from src.features import compute_R_per_hist, normalize
from src.features import extract_systematics_names
from src.features import covariance_matrix_extraction
from src.features import modify_covariance
from src.features import sample_theta
from src.features import make_base_fn, propagate_to_hist
from src.features import build_prediction
from src.features import histogram_prediction_pseudo_data
from src.extraction_data import names_goups
from src.extraction_data import get_index_in_group
from models.model_used import training_model
from models.model_used import models
from src.features import Mahalanobis_distance
from src.visualization import scatter_PCA
from src.visualization import plot_3D
from src.visualization import heatmap_visualization, interactive_plot_3D
from numpy.linalg import inv
from functools import partial

import ipywidgets as widgets
from IPython.display import display

def main():

    print("Let's start this analysis !")

    print("Starting the main function...")

    systematics, samples, variables, reactions = preparation()
    print("Systematics:", systematics)
    print("Samples:", samples)
    print("Variables:", variables)
    print("Reactions:", reactions)

    file_path = r"C:\Users\lenao\Project_emulator_python\inputs\fdsFit_Martini1pi_2021.root"
    root_file = uproot.open(file_path)
    predictions, y, edges = extraction_data(samples, variables, reactions, root_file)

    print("Normalization of systematics names:")
    bestfit_dict, sigma_dict = extract_systematics_names(systematics, root_file)
    print("Extracted systematics names.")

    print("Extracting covariance matrix...")
    mean, cov_reduced, matched_keys = covariance_matrix_extraction(root_file, bestfit_dict)
    print("Covariance matrix extracted and reduced.")

    print("Modifying covariance matrix...")
    cov_modified = modify_covariance(cov_reduced, corr_strength=0.5, scale=1.0)
    print("Covariance matrix modified.")
    sigma = np.sqrt(np.diag(cov_modified))
    print("Standard deviations extracted from modified covariance matrix.")

    R_dict = compute_R_per_hist(mean, sigma, samples, variables, reactions, predictions)
    print("R is calculated.")


    n_bins = len(y)
    n_params = len(mean)

    print("Building pseudo-data...")
    output = histogram_prediction_pseudo_data(predictions, R_dict, samples, variables, reactions, mean, cov_modified, n_samples=1000)

    shifted_prediction_visualization(predictions, samples, variables, reactions, mean, cov_modified, R_dict)

    param_names, groups, cov_labels = names_goups(root_file)

    for g, idxx in groups.items():
        print(g, ":", len(idxx))
    print("Total parameters:", len(param_names))

    print("Training model...")
    datasets  = training_model(mean, cov_modified, predictions, samples, variables, reactions, R_dict, groups)
    print("Model trained. Datasets prepared.")

    models_dict = models(datasets)
    example_key = list(datasets.keys())[0]

    print("Using key:", example_key)

    rf_model = models_dict[example_key]["rf"]
    X_test = models_dict[example_key]["X_test"]


    #bins, X_sorted, Y_sorted, dist, cov_inv, diff = Mahalanobis_distance(cov_modified, X_test, mean, Y_test, y_nominal, X_pca)
    #print("Mahalanobis distance calculated.")

    print("distance start to be calculated.")
    cov_inv = inv(cov_modified)
    diff = X_test - mean
    dist = np.sqrt(np.einsum('ij,jk,ik->i', diff, cov_inv, diff))  # Distance de Mahalanobis
    idx_sort = np.argsort(dist)
    dist = dist[idx_sort]

    #scatter_PCA(X_pca=X_pca, dist=dist)
    print("scatter plot PCA done.")

    ui, out = interactive_plot_3D(cov_labels, X_test, rf_model)
                
    display(ui, out)

    heatmap_visualization(mean, cov_modified, predictions, samples, variables, reactions, R_dict, groups, cov_labels)


if __name__ == "__main__":
    main()