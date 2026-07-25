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
