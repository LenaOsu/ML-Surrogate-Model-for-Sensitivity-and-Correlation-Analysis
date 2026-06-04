
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from src.features import normalize, sample_theta, propagate_to_hist, compute_R_per_hist
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
import numpy as np

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

    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.decomposition import PCA

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
            "ridge": Ridge(),
            "mlp": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500),
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

            Y_pred = model.predict(X_test)
            print("name:", name, "Y_pred shape:", Y_pred.shape)
            trained[name] = model
            print("idx:", idx, "Y_pred[idx] shape:", Y_pred[idx].shape)

            plt.plot(
                Y_pred[idx],
                linewidth=2,
                label=name
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
            "X_pca": X_pca
        }

    return models_dict