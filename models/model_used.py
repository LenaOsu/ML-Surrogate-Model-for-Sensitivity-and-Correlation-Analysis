
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from src.features import sample_theta
from src.features import propagate_to_hist
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA



def training_model(mean, cov, predictions, samples, variables, reactions, R_dict, groups, cov_labels):
    
    param_names = np.array(cov_labels)          
    datasets = {}
    n_params = len(mean)

    for s in samples:
        for v in variables:
            for r in reactions:
        
                    y_nominal = predictions[s][v][r]["values"]
                    n_bins = len(y_nominal)
                    
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
        
                    print("X:", X.shape, "Y:", Y.shape)
        
                    Y_norm = Y/(y_nominal + 1e-8) - 1
        
                    corr_sys = np.corrcoef(X, rowvar=False)    

                    for g, idx in groups.items():
        
                        if len(idx) < 2:
                            continue
        
                        corr_sub = np.corrcoef(X[:, idx], rowvar=False)

                    corr_sys_bin = np.corrcoef(X.T, Y_norm.T)[:X.shape[1], X.shape[1]:]
        
                    for g, idx in groups.items():
        
                        if len(idx) == 0:
                            continue
        
                        corr_sub = corr_sys_bin[idx, :]
        
                            
                    param_names = np.array(param_names)  # IMPORTANT FIX numpy -> list
                    EPS = 1e-8

                    X_train, X_test, Y_train, Y_test = train_test_split(
                        X, Y_norm, test_size=0.2, random_state=42
                    )

    return X_train, Y_train, X_test, Y_test, datasets

def models(x_train, y_train, x_test, y_test, param_names, datasets, samples, variables, reactions):

    modelsplusRF = {
        "linear": LinearRegression(),
        "ridge": Ridge(),
        "mlp": MLPRegressor(hidden_layer_sizes=(64,64), max_iter=500),
        "rf" : RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        max_features=0.5,
        bootstrap=True,
        random_state=42,
        n_jobs=-1)
    }

    plt.figure(figsize=(8,5))
    plt.plot(y_test[0], label="true", linewidth=3, color="black")
                

    for name, model in modelsplusRF.items():
                

        model.fit(x_train, y_train)
        Y_pred = model.predict(x_test)
                
        plt.plot(Y_pred[0], label=name)# first prediction
                

    plt.legend(loc="best")
    plt.title("Model comparison on histogram (per bin)")
    plt.grid(True)
    plt.show()

    for s in samples:
        for v in variables:
            for r in reactions:

                for name, model in modelsplusRF.items():

                    model.fit(x_train, y_train)
                    Y_pred = model.predict(x_test)
                
    
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(x_train)
                    modelRF = modelsplusRF["rf"]
                    modelRF.fit(x_train, y_train)
                    Y_base = modelRF.predict(x_test)
                    comp = pca.components_[0]
                    idx = np.argsort(np.abs(comp))[::-1]
                    #print("\nTop PC1:")

                    #for i in idx[:10]:
                        #print(param_names[i], comp[i])

                    datasets[(s,v,r)] = (x_test, Y_base)
                    ridge = Ridge()
                    ridge.fit(x_train, y_train)

                    coef = ridge.coef_


    return modelsplusRF, modelRF, X_pca, coef


