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

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from matplotlib import pyplot as plt
import seaborn as sns
from src.features import normalize, sample_theta, propagate_to_hist, compute_R_per_hist
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error

def nominal_prediction_visualization(predictions, samples, variables, reactions):

    for sample in samples:
        s = normalize(sample)
        for variable in variables:
            v = normalize(variable)
            for reaction in reactions:
                r = normalize(reaction)

                key = (s, v, r)

                try:
                    y = predictions[key]["values"]
                    edges = predictions[key]["edges"]

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
                    y = predictions[(s, v, r)]["values"]
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
    print("1")

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
    print("2")

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
    print("3")

    out = widgets.interactive_output(
        plot_func,
        {"param_name": dropdown}
    )
    print("4")

    return ui, out

def heatmap_visualization(predictions, R_dict, mean, cov,
                          samples, variables, reactions,
                          groups, param_names):

    print("Starting heatmap visualization...")

    for sample in samples:
        for variable in variables:
            for reaction in reactions:

                key = (normalize(sample), normalize(variable), normalize(reaction))
                print("niceee")
                #print("key:", key.shape if isinstance(key, np.ndarray) else key)

                #try:
                y = predictions[key]["values"]
                print(type(predictions))
                print(type(sample), sample)
                print(type(variable), variable)
                print(type(reaction), reaction)
                R = R_dict[key]

                print("heyy")

                theta_samples = sample_theta(mean, cov, 1000)

                X = np.array(theta_samples)
                Y = np.array([
                    propagate_to_hist(y, theta, mean, R)
                    for theta in X
                ])

                y0 = y
                #Y_norm = Y / (y0 + 1e-8) - 1
                Y_norm = (Y - y0)/(np.std(Y, axis=0) + 1e-8)#normalization per bin

                print("bouuuuu")

                corr_sys_bin = np.corrcoef(
                    X.T, Y_norm.T
                )[:X.shape[1], X.shape[1]:]
                print("bin-wise correlation variance:",np.std(corr_sys_bin, axis=0).mean())

                print("blblblbl")

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
                    print("miam")

                    plt.title(f"Sys → Bin {g}")
                    plt.xlabel("bins")
                    plt.ylabel("parameters")
                    plt.show()

                #except Exception as e:
                 #   print(type(predictions))
                  #  print(type(sample), sample)
                   # print(type(variable), variable)
                   # print(type(reaction), reaction)
                    #print("so thats here, but why seriously ?")
                    #print(f"Erreur {sample}-{variable}-{reaction}: {e}")

def plot_learning_curve(model, X, Y, title, ax, train_sizes=np.linspace(0.1, 1.0, 8)):
    """
    LE graphique de reference pour diagnostiquer under/overfitting.

    - Si train_score ET test_score sont bas et convergent -> sous-apprentissage
      (le modele est trop simple, ajouter des donnees ne changera rien : il
      faut un modele plus complexe ou plus de features utiles).
    - Si train_score est haut mais test_score reste bas avec un grand ecart
      qui ne se resorbe pas quand n augmente -> sur-apprentissage
      (le modele a trop de capacite pour le volume de donnees disponible).
    - Si les deux convergent vers un score haut -> le modele va bien.

    Attention : Y ici est multi-sortie (un vecteur par bin). sklearn calcule
    alors un R2 moyen sur toutes les sorties, ce qui suffit pour le diagnostic
    global mais masque des differences bin-par-bin (a creuser en second temps
    si besoin, bin par bin).
    """
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, Y,
        train_sizes=train_sizes,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    ax.plot(train_sizes_abs, train_mean, "o-", color="tab:blue", label="Score train (CV)")
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color="tab:blue")
    ax.plot(train_sizes_abs, test_mean, "o-", color="tab:red", label="Score validation (CV)")
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.15, color="tab:red")

    ax.set_xlabel("Taille de l'echantillon d'entrainement")
    ax.set_ylabel("R2 score")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def diagnose_all_models(datasets, example_key=None, n_estimators_range=None, alphas=None):
    """
    A appeler apres models(). Produit, pour UNE clé (sample, variable, reaction)
    donnee (ou la premiere par defaut) :
      1. les learning curves des 4 modeles (under/overfitting global)
      2. la validation curve du Ridge selon alpha (choix de regularisation)
      3. la validation curve de la RandomForest selon n_estimators
    """
    if example_key is None:
        example_key = list(datasets.keys())[0]

    X = datasets[example_key]["X"]
    Y = datasets[example_key]["Y"]

    print(f"Diagnostic pour la clé : {example_key}")
    print(f"X shape: {X.shape} (n_samples, n_features) -- attention si n_features "
          f"est proche de n_samples*0.8 (train), risque fort de surapprentissage.")

    # 1. Learning curves des 4 familles de modeles
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    models_to_check = {
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        "MLP (64,64)": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000,
                                     early_stopping=True, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                max_features=0.5, random_state=42, n_jobs=-1)
    }
    for ax, (name, model) in zip(axes.ravel(), models_to_check.items()):
        plot_learning_curve(model, X, Y, name, ax)
    plt.tight_layout()
    plt.savefig("outputs/plots/learning_curves.png", dpi=130)
    plt.show()

    # 2. Validation curve : effet de la regularisation Ridge sur under/overfitting
    if alphas is None:
        alphas = np.logspace(-3, 3, 13)
    train_scores, test_scores = validation_curve(
        Ridge(), X, Y, param_name="alpha", param_range=alphas,
        cv=5, scoring="r2", n_jobs=-1
    )
    plt.figure(figsize=(7, 4.5))
    plt.semilogx(alphas, train_scores.mean(axis=1), "o-", label="Score train", color="tab:blue")
    plt.semilogx(alphas, test_scores.mean(axis=1), "o-", label="Score validation", color="tab:red")
    plt.xlabel("alpha (force de regularisation)")
    plt.ylabel("R2 score")
    plt.title("Ridge : choix de alpha\n(alpha trop petit=overfitting, trop grand=underfitting)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/plots/ridge_validation_curve.png", dpi=130)
    plt.show()

    # 3. Validation curve : effet du nombre d'arbres sur la RandomForest
    if n_estimators_range is None:
        n_estimators_range = [10, 25, 50, 100, 200, 300, 500]
    train_scores, test_scores = validation_curve(
        RandomForestRegressor(min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=-1),
        X, Y, param_name="n_estimators", param_range=n_estimators_range,
        cv=5, scoring="r2", n_jobs=-1
    )
    plt.figure(figsize=(7, 4.5))
    plt.plot(n_estimators_range, train_scores.mean(axis=1), "o-", label="Score train", color="tab:blue")
    plt.plot(n_estimators_range, test_scores.mean(axis=1), "o-", label="Score validation", color="tab:red")
    plt.xlabel("n_estimators")
    plt.ylabel("R2 score")
    plt.title("Random Forest : effet du nombre d'arbres")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/plots/rf_validation_curve.png", dpi=130)
    plt.show()


def plot_mlp_loss_curve(mlp_model, title="MLP - loss train vs validation"):
    """
    A appeler avec le modele MLP deja entraine (trained["mlp"]), a condition
    qu'il ait ete cree avec early_stopping=True (c'est le cas dans la version
    mise a jour de models() ci-dessus). Equivalent exact du graphique
    loss train/val qu'on a trace pour le reseau PyTorch : la encore, si la
    loss de validation (ici approximee par validation_scores_, un score R2 au
    lieu d'une loss) se degrade alors que loss_curve_ (train) continue de
    baisser, c'est le signe du sur-apprentissage.
    """
    if not hasattr(mlp_model, "validation_scores_"):
        print("Le modele MLP n'a pas ete entraine avec early_stopping=True, "
              "impossible de recuperer la courbe de validation.")
        return

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(mlp_model.loss_curve_, color="tab:blue", label="Loss train")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss train", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(mlp_model.validation_scores_, color="tab:red", label="R2 validation")
    ax2.set_ylabel("R2 validation", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    plt.title(title)
    fig.tight_layout()
    plt.savefig("outputs/plots/mlp_loss_curve.png", dpi=130)
    plt.show()