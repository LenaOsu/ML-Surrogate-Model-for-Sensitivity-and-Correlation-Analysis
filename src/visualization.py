from matplotlib import widgets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.features import sample_theta
from src.features import propagate_to_hist
from src.features import compute_R_per_hist
from src.features import normalize
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
        s = normalize(sample)
        for variable in variables:
            v = normalize(variable)
            for reaction in reactions:
                r = normalize(reaction)

                try:
                    y = predictions[s][v][r]["values"]
                    edges = predictions[s][v][r]["edges"]

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

def shifted_prediction_visualization(predictions, samples, variables, reactions, mean, cov, R_dict):

    for s in samples:
        s = normalize(s)
        for v in variables:
            v = normalize(v)
            for r in reactions:
                r = normalize(r)

                try:
                    y = predictions[s][v][r]["values"]
                    R = R_dict[(s, v, r)]

                    theta_samples = sample_theta(mean, cov, 1000)

                    pseudo = np.array([propagate_to_hist(y, theta, mean, R) for theta in theta_samples])

                    y = propagate_to_hist(y, mean, mean, R)  # nominal = bestfit

                    bins = np.arange(len(y))

                    plt.figure(figsize=(8,5))

                    for i in range(pseudo.shape[0]):
                        plt.scatter(bins, pseudo[i], color='blue', alpha=0.05)

                    mean_bins = pseudo.mean(axis=0)
                    std_bins = pseudo.std(axis=0)

                    plt.step(bins, mean_bins, color='red', linewidth=2, label='mean pseudo')
                    plt.fill_between(bins, mean_bins-std_bins, mean_bins+std_bins, color='red', alpha=0.3)

                    plt.step(bins, y, color='black', linewidth=2, label='nominal')

                    plt.title(f"{s} | {v} | {r}")
                    plt.legend()
                    plt.grid()
                    plt.show()

                except Exception as e:
                    print(f"Erreur {s}-{v}-{r}: {e}")


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
            dropdown.value = filtered[0] 

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

    #return ui, out

def heatmap_visualization(predictions, R_dict, mean, cov,
                          samples, variables, reactions,
                          groups, param_names):

    print(type(samples[0]), samples[0])
    print(type(variables[0]), variables[0])
    print(type(reactions[0]), reactions[0])

    for s in samples:
        s = normalize(s)
        for v in variables:
            v = normalize(v)
            for r in reactions:
                r = normalize(r)

                try:
                    y = predictions[s][v][r]["values"]
                    R = R_dict[(s, v, r)]

                    theta_samples = sample_theta(mean, cov, 1000)

                    X = np.array(theta_samples)
                    Y = np.array([
                        propagate_to_hist(y, theta, mean, R)
                        for theta in X
                    ])

                    y0 = y
                    Y_norm = Y / (y0 + 1e-8) - 1

                    corr_sys_bin = np.corrcoef(
                        X.T, Y_norm.T
                    )[:X.shape[1], X.shape[1]:]

                    for g, idx in groups.items():
                        if len(idx) == 0:
                            continue

                        plt.figure(figsize=(8,6))

                        sns.heatmap(
                            corr_sys_bin[idx, :],
                            cmap="coolwarm",
                            center=0,
                            yticklabels=[param_names[i] for i in idx]
                        )

                        plt.title(f"Sys → Bin {g}")
                        plt.xlabel("bins")
                        plt.ylabel("parameters")
                        plt.show()

                except Exception as e:
                    print(f"Erreur {s}-{v}-{r}: {e}")