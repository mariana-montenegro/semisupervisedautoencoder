import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif

def feature_selection(x_scaled, y, fs_method='lasso', k_values=None, lasso_alphas=None):
    selected_features = []
    
    if fs_method == 'lasso':
        if lasso_alphas is None:
            lasso_alphas = [0.001]
        for alpha in lasso_alphas:
            lasso_model = Lasso(alpha=alpha)
            sfm = SelectFromModel(lasso_model)
            x_selected = sfm.fit_transform(x_scaled, y)
            selected_features.append(x_selected)
    
    elif fs_method == 'pearson':
        if k_values is None:
            k_values = [3, 5, 10]
        for k in k_values:
            corrs = [np.abs(np.corrcoef(x_scaled[:, i], y)[0, 1]) for i in range(x_scaled.shape[1])]
            top_k_indices = np.argsort(corrs)[-k:]
            x_selected = x_scaled[:, top_k_indices]
            selected_features.append(x_selected)
    
    elif fs_method == 'anova':
        if k_values is None:
            k_values = [50, 100, 200]
        for k in k_values:
            selector = SelectKBest(f_classif, k=k)
            x_selected = selector.fit_transform(x_scaled, y)
            selected_features.append(x_selected)
    
    else:
        raise ValueError("Invalid feature selection method. Choose 'lasso', 'pearson', or 'anova'.")
    
    return selected_features


def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)

    return data_pca

def plot_pca_scatter(data_pca, labels):
    # Plot the 2D scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=labels, palette='viridis', s=50)
    plt.title('2D Scatter Plot of Data in PCA Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Label')
    plt.show()


def plot_decision_regions(x, y, classifier, test_idx=None, resolution=0.02, title='Decision Regions'):
    bordeaux = '#831F3D'
    blue = '#56C5D0'
    markers = ('s', 'x', 'o', '^', 'v')
    colors = (bordeaux, blue)
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    legend_labels = {0: 'Control', 1: 'Patient'}
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], 
                    y=x[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=legend_labels[cl],  # Use custom legend labels
                    edgecolor='black')

    if test_idx is not None:
        X_test, y_test = x[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.tight_layout()
    plt.show()