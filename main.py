
import numpy as np
import pandas as pd
import os
import uproot
import matplotlib.pyplot as plt
from src.preparation import preparation
from src.extraction_data import extraction_data
from src.visualization import nominal_prediction_visualization
from src.visualization import shifted_prediction_visualization
from src.features import normalize
from src.features import extract_systematics_names
from src.features import covariance_matrix_extraction
from src.features import modify_covariance
from src.features import sample_theta
from src.features import propagate_to_hist
from src.extraction_data import build_prediction
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

    print("Sampling theta values...")
    samples_theta = sample_theta(mean, cov_modified, 1000)
    print("Theta values sampled.")

    n_bins = len(y)
    n_params = len(mean)

    print("Propagating to histogram space...")
    theta_samples = sample_theta(mean, cov_modified, 1000)

    print("Building prediction...")
    y_nominal = build_prediction(samples, variables, reactions, predictions)
    y_pseudo_data = histogram_prediction_pseudo_data(predictions, samples, variables, reactions, mean, cov_modified)

    #nominal_prediction_visualization(predictions, samples, variables, reactions)
    shifted_prediction_visualization(predictions, samples, variables, reactions, y_pseudo_data=y_pseudo_data)

    param_names, groups, cov_labels = names_goups(root_file)

    for g, idxx in groups.items():
        print(g, ":", len(idxx))
    print("Total parameters:", len(param_names))

    print("Training model...")
    R_dict = {}
    X_train, Y_train, X_test, Y_test, datasets = training_model(mean, cov_modified, predictions, samples, variables, reactions, R_dict, groups, cov_labels)
    print("Model trained. Training set size:", len(X_train), "Test set size:", len(X_test)) 

    models_dict, modelRF, X_pca, coef = models(X_train, Y_train, X_test, Y_test, param_names, datasets, samples, variables, reactions)
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

    ui, out = interactive_plot_3D(cov_labels, X_test, modelRF)
                
    display(ui, out)

    #heatmap_visualization(mean, cov_modified, predictions, samples, variables, reactions, R_dict, groups, cov_labels)


if __name__ == "__main__":
    main()