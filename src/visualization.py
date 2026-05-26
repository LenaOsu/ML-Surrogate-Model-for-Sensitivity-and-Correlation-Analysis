from matplotlib import widgets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.features import sample_theta
from src.features import propagate_to_hist
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import Ridge
#import unmap
import plotly.io as pio
pio.renderers.default = "browser"



def nominal_prediction_visualization(predictions, samples, variables, reactions):

    for sample in samples:
        for variable in variables:
            for reaction in reactions:

                try:
                    y = predictions[sample][variable][reaction]["values"]
                    edges = predictions[sample][variable][reaction]["edges"]

                    plt.figure(figsize=(6,4))

                    plt.step(edges[:-1], y, where="post", label="prediction")
                    plt.bar(edges[:-1], y, width=np.diff(edges), alpha=0.3, align="edge")

                    plt.xlabel("Bin")
                    plt.ylabel("Event number")
                    plt.title(f"{sample} - {variable} - {reaction}")
                    plt.grid()
                    plt.legend()
                    #plt.show()
                
                except Exception as e:
                    print(f"Erreur {reaction}: {e}")
    return


def shifted_prediction_visualization(predictions, samples, variables, reactions, y_pseudo_data=None):

    for sample in samples:
        for variable in variables:
            for reaction in reactions:

                try:
                    y = predictions[sample][variable][reaction]["values"]
                    edges = predictions[sample][variable][reaction]["edges"]

                    Bins = np.arange(len(y))

                    plt.figure(figsize=(8,5))

                    for i in range(y_pseudo_data.shape[0]):
                        plt.scatter(Bins, y_pseudo_data[i], color='blue', alpha=0.05)

                    mean_bins = y_pseudo_data.mean(axis=0)
                    std_bins  = y_pseudo_data.std(axis=0)

                    plt.figure(figsize=(6,4))

                    plt.step(Bins, mean_bins, color='red', linewidth=2, label='mean pseudo data')
                    plt.fill_between(Bins, mean_bins-std_bins, mean_bins+std_bins, color='red', alpha=0.3, label='±1σ')

                    plt.step(Bins, y, color='black', linewidth=2, label='nominal')

                    plt.title(f"{sample} | {variable} | {reaction}")
                    plt.xlabel("bin")
                    plt.ylabel("events")
                    plt.legend()
                    plt.grid(True)
                    #plt.show()
                
                except Exception as e:
                    print(f"Erreur {reaction}: {e}")
    return

def scatter_PCA(X_pca, dist):

    fig = go.Figure(data=[go.Scatter3d(
                x=X_pca[:,0],
                y=X_pca[:,1],
                z=dist,
                mode='markers',
                marker=dict(
                    size=3,
                    color=dist,
                    colorscale='Viridis'
                    )
                )])
    fig.update_layout(title="Systematics landscape (PCA + Mahalanobis)")
    fig.show()

    return

from functools import partial
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go


def plot_3D(param_name, cov_labels, X_test, modelRF, EPS=1e-8):

    param_names = list(cov_labels)

    if param_name not in param_names:
        print(f"Parameter '{param_name}' not found.")
        return

    p_idx = param_names.index(param_name)

    X_var = X_test.copy()

    delta = np.std(X_test[:, p_idx]) * 0.5

    X_var[:, p_idx] += delta

    Y_var = modelRF.predict(X_var)
    Y_base = modelRF.predict(X_test)

    Z = (Y_var - Y_base) / (Y_base + EPS)

    n_predictions = Z.shape[0]
    n_bins = Z.shape[1]

    predictions = np.arange(n_predictions)
    bins = np.arange(n_bins)

    P, B = np.meshgrid(predictions, bins, indexing='ij')

    zmax = np.max(np.abs(Z))

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=B,
            y=P,
            z=Z,
            colorscale='RdBu',
            reversescale=True,
            cmin=-zmax,
            cmax=zmax,
            opacity=0.95,
            colorbar=dict(title="ΔY/Y")
        )
    )

    fig.add_trace(
        go.Surface(
            x=B,
            y=P,
            z=np.zeros_like(Z),
            opacity=0.15,
            showscale=False
        )
    )

    fig.update_layout(
        title=f"ML Impact Surface : {param_name}",
        width=1400,
        height=850,
        template='plotly_dark',
        scene=dict(
            xaxis_title="Bins",
            yaxis_title="Predictions",
            zaxis_title="Impact",
            camera=dict(
                eye=dict(x=1.8, y=1.5, z=1.2)
            )
        )
    )

    fig.show(renderer="browser")


def interactive_plot_3D(cov_labels, X_test, modelRF):

    param_names = list(cov_labels)

    search_box = widgets.Text(
        placeholder='Search parameter...',
        description='Search:',
        layout=widgets.Layout(width='400px')
    )

    dropdown = widgets.Dropdown(
        options=param_names,
        description='Param:',
        layout=widgets.Layout(width='700px')
    )

    def update_dropdown(change):

        search = change["new"].lower()

        filtered = [
            p for p in param_names
            if search in p.lower()
        ]

        if len(filtered) > 0:
            dropdown.options = filtered

    search_box.observe(update_dropdown, names='value')

    ui = widgets.VBox([
        search_box,
        dropdown
    ])

    plot_func = partial(
        plot_3D,
        cov_labels=cov_labels,
        X_test=X_test,
        modelRF=modelRF
    )

    out = widgets.interactive_output(
        plot_func,
        {"param_name": dropdown}
    )

    return ui, out


def heatmap_visualization(mean, cov, predictions, samples, variables, reactions, R_dict, groups, cov_labels):
    
    param_names = np.array(cov_labels)          
    datasets = {}
    n_params = len(mean)


    for s in samples:
        for v in variables:
            for r in reactions:
                    y_nominal = predictions[s][v][r]["values"]

                    key = (s, v, r)
                    if key not in R_dict:
                        R_dict[key] = np.random.normal(
                            0, 0.02,
                            size=(len(y_nominal), len(mean))
                        )
                    R = R_dict[key]
        
                    theta_samples = sample_theta(mean, cov, 1000)
                    X = np.array(theta_samples)
                    Y = []

                    for theta in X:
                        yy = propagate_to_hist(y_nominal, theta, mean, R)
                        Y.append(yy)

                    Y = np.array(Y)
                    Y_norm = Y/(y_nominal + 1e-8) - 1
                
                    corr_sys = np.corrcoef(X, rowvar=False)    

                    for g, idx in groups.items():
        
                        if len(idx) < 2:
                            continue
        
                        corr_sub = np.corrcoef(X[:, idx], rowvar=False)
      
                        plt.figure(figsize=(5,4))
                        sns.heatmap(
                            corr_sub,
                            cmap="coolwarm",
                            center=0,
                            xticklabels=[param_names[i] for i in idx],
                            yticklabels=[param_names[i] for i in idx]
                        )
                        plt.title(f"Sys-Sys {g}")
                        plt.xticks(rotation=90)
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        plt.show()

                    corr_sys_bin = np.corrcoef(X.T, Y_norm.T)[:X.shape[1], X.shape[1]:]
        
                    for g, idx in groups.items():
        
                        if len(idx) == 0:
                            continue
        
                        corr_sub = corr_sys_bin[idx, :]
        
                        plt.figure(figsize=(8,6))
                        sns.heatmap(
                            corr_sub,
                            cmap="coolwarm",
                            center=0,
                            yticklabels=[param_names[i] for i in idx]
                        )
                        plt.title(f"Sys → Bin {g}")
                        plt.xlabel("bins")
                        plt.ylabel("parameters")
                        plt.tight_layout()
                        plt.show()
                            
                    param_names = np.array(param_names)  # IMPORTANT FIX numpy -> list

    return 

def model_superoposition_visu():




    return