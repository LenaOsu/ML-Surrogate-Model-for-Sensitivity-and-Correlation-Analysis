import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

from src.features import normalize


def extraction_data(samples, variables, reactions, root_file):

    print("Starting the data extraction function...")
    predictions = {}

    #one simply extract the postfit prediction in momentum and direction from the original fitter output
    for sample in samples:

        for variable in variables:

            base_path = f"FitterEngine/postFit/samples/histograms/{sample}/{variable}/ReactionCode"

            try:
                available_reactions = root_file[base_path].keys()
            except:
                print(f"Missing path: {base_path}")
                continue

            for reaction in reactions:
                r = normalize(reaction)
                key = (normalize(sample), normalize(variable), normalize(reaction))
                predictions[key] = {}

                try:
                    hist = root_file[f"{base_path}/{reaction}/MC_TH1D"]

                    values = hist.values()
                    edges = hist.axes[0].edges()

                    predictions[key] = {
                        "values": np.array(values), # number of events per bin
                        "edges": edges # pmu/cos edges
                    }

                    y = predictions[key]["values"]
                    edges = predictions[key]["edges"]
                    print(type(predictions))
                    print(type(sample), sample)
                    print(type(variable), variable)
                    print(type(reaction), reaction)


                except Exception as e:
                    print(f"Erreur {reaction}: {e}")
                    print(type(predictions))
                    print(type(sample), sample)
                    print(type(variable), variable)
                    print(type(reaction), reaction)
                    
    return predictions, y, edges


def names_goups(root_file):

    cov_obj = root_file[
        "FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D"# cov matrix without PCA in first place
    ]

    cov = cov_obj.values()

    labels = list(cov_obj.axes[0].labels())
    assert labels == list(cov_obj.axes[1].labels())
    
    cov_labels = [normalize(n) for n in labels]
    print(cov_labels)

    param_names = np.array(cov_labels)

    #want to divide plots by systematics, here are flux, xsec and detector
    groups = {
        "flux": np.arange(0, 100),
        "det":  np.arange(100, 100 + 552),
        "xsec": np.arange(100+552, 100+551+56),
        "xsecEb": np.arange(100+551+56, 100+551+56+4),
    }

    return param_names, groups, cov_labels


def get_index_in_group(name, group, param_names, groups):

    idx_group = groups[group]              # indices globaux du groupe
    names_group = param_names[idx_group]   # labels du groupe
    matches = np.where(names_group == name)[0]

    if len(matches) == 0:
        raise ValueError(f"{name} not found in group {group}")

    return idx_group[matches[0]]