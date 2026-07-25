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


def training_model(mean, cov, predictions, samples, variables, reactions, R_dict, groups):

    datasets = {}

    for s in samples:
        s = normalize(s)
        for v in variables:
            v = normalize(v)
            for r in reactions:
                r = normalize(r)

                key = (s, v, r)

                y_nominal = predictions[key]["values"]
                R = R_dict[key]

                theta_samples = sample_theta(mean, cov, 1000)
                X = np.array(theta_samples)

                Y = np.array([
                    propagate_to_hist(y_nominal, theta, mean, R)
                    for theta in X
                ])

                Y_norm = Y / (y_nominal + 1e-8) - 1

                datasets[key] = {
                    "X": X,
                    "Y": Y_norm,
                    "y_nominal": y_nominal
                }

                print(f"{key} -> X {X.shape}, Y {Y_norm.shape}")

    return datasets

def models(datasets):



    models_dict = {}

    for key, data in datasets.items():
        X = data["X"]
        Y = data["Y"]

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y,
            test_size=0.2,
            random_state=42
        )

        modelsplusRF = {
            "linear": LinearRegression(),
            "ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),  # alpha choisi par CV plutot que fixe
            "mlp": MLPRegressor(
                hidden_layer_sizes=(64, 64),
                max_iter=2000,
                early_stopping=True,      # reserve 10% des donnees train comme validation interne
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=42
            ),
            "rf": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                max_features=0.5,
                random_state=42,
                n_jobs=-1
            )
        }

        print(f"\nTraining model for {key}")
        print("X:", X_train.shape, "Y:", Y_train.shape)

        trained = {}
        scores = {}  # <-- stocke le diagnostic under/overfitting par modele

        plt.figure(figsize=(10,6))

        idx = 0  # premier événement du test

        plt.plot(
            Y_test[idx],
            color="black",
            linewidth=3,
            label="Truth"
        )

        for name, model in modelsplusRF.items():

            model.fit(X_train, Y_train)

            Y_pred_train = model.predict(X_train)
            Y_pred_test = model.predict(X_test)

            # R2 et MSE sur train ET test : c'est la comparaison des deux qui
            # donne le diagnostic, jamais un score seul.
            r2_train = r2_score(Y_train, Y_pred_train)
            r2_test = r2_score(Y_test, Y_pred_test)
            mse_train = mean_squared_error(Y_train, Y_pred_train)
            mse_test = mean_squared_error(Y_test, Y_pred_test)
            gap = r2_train - r2_test

            scores[name] = {
                "r2_train": r2_train, "r2_test": r2_test,
                "mse_train": mse_train, "mse_test": mse_test,
                "gap": gap
            }

            # regle de lecture simple, a affiner selon tes propres seuils :
            if r2_train < 0.5 and r2_test < 0.5:
                diagnostic = "SOUS-APPRENTISSAGE (train et test faibles)"
            elif gap > 0.15:
                diagnostic = "SUR-APPRENTISSAGE (gros ecart train/test)"
            else:
                diagnostic = "OK (train et test proches et corrects)"

            print(f"[{name}] R2 train={r2_train:.3f} | R2 test={r2_test:.3f} "
                  f"| gap={gap:.3f} | MSE train={mse_train:.4g} | MSE test={mse_test:.4g} "
                  f"-> {diagnostic}")

            Y_pred = Y_pred_test
            trained[name] = model

            plt.plot(
                Y_pred[idx],
                linewidth=2,
                label=f"{name} (R2 test={r2_test:.2f})"
            )

        plt.xlabel("Bin")
        plt.ylabel("Normalized variation")
        plt.title(f"Histogram prediction comparison : {key}")
        plt.legend()
        plt.grid(True)
        plt.show()

        # PCA (optionnel mais propre)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train)

        models_dict[key] = {
            "models": trained,
            "rf": trained["rf"],
            "X_test": X_test,
            "Y_test": Y_test,
            "X_pca": X_pca,
            "scores": scores,   # diagnostic chiffre, reutilisable ensuite
        }

    return models_dict


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