# pyright: reportPrivateImportUsage=false

# Reference: https://www.kaggle.com/code/karimsaieh/pca-principal-component-analysis-without-sklearn

from jax import jit, random, vmap
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.decomposition import PCA as sk_PCA

KEY = jax.random.PRNGKey(42)
np.random.seed(42)

@jit
def fit(X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    corr = jnp.cov(X, rowvar=False)
    eig_vectors, eig_values, _ = jnp.linalg.svd(corr)
    ex_var = (eig_values / jnp.sum(eig_values))*100
    return eig_values, eig_vectors, ex_var

@jit
def transformToNewSpace(X, W):
    return X.dot(W)

class PCA:
    def __init__(self, X: jnp.ndarray, out_dim: int, norm: bool=True) -> None:
        if norm:
            self.X = (X - X.mean()) / X.std(ddof=0)
        else:
            self.X = X

        self.n = out_dim
        self.eig_vals = None
        self.eig_vecs = None

    def selectFeature(self):
        if (self.eig_vals is None) or (self.eig_vecs is None):
            raise ValueError("No features to select, since fit has not yet been called!")

        eig_pairs = [(np.abs(self.eig_vals[i]), self.eig_vecs[:,i]) for i in range(len(self.eig_vals))] # type: ignore
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        W = np.hstack([eig_pairs[x][1].reshape(self.eig_vecs.shape[0], 1) for x in range(self.n)]) # type: ignore
        return W

    def fit(self):
        eig_vals, eig_vecs, self.ex_var = fit(self.X)
        self.eig_vecs = eig_vecs
        self.eig_vals = eig_vals

    def transform(self):
        return transformToNewSpace(self.X, self.selectFeature())

    def fit_transform(self) -> jnp.ndarray:
        self.fit()
        return self.transform()


def jax_test(X, model, ax=None):
    dim_red = model.fit_transform()
    dim_red = pd.DataFrame(dim_red, columns=['x','y'])
    dim_red['class'] = y
    if ax is not None:
        ax[0].set_title('Jax')
        g = sns.scatterplot(dim_red, x='y', y='x', hue='class', ax=ax[0])


def sk_test(X, model, ax=None):
    dim_red = model.fit_transform(X)
    dim_red = pd.DataFrame(dim_red, columns=['x','y'])
    dim_red['class'] = y
    if ax is not None:
        ax[1].set_title('Sklearn')
        g = sns.scatterplot(dim_red, x='y', y='x', hue='class', ax=ax[1])


if __name__ == "__main__":
    features = 10
    samples = 8000
    classes = 4
    means = np.array([x+[1]*8 for x in [[1,1],[1,-1],[-1,1],[-1,-1]]])*8
    red = 2

    X = []
    y = []
    for c, mean in zip(range(classes), means):
        X.append(np.random.normal(mean, size=(samples,features)))
        y.append(np.full(samples, f'test{c}'))
    X = np.vstack(X)
    y = np.concatenate(y)

    test = pd.DataFrame(X)
    # print(test.corr())

    model = PCA(X, red) # type: ignore
    sk_model = sk_PCA(n_components=2)

    fig, ax = plt.subplots(1,2) # type: ignore
    jax_test(X, model, ax)
    X = (X - X.mean()) / X.std(ddof=0)
    sk_test(X, sk_model, ax)

    ex_var = pd.DataFrame(model.ex_var)
    ex_var['component'] = np.arange(ex_var.shape[0])
    ex_var.rename({0:'variance'}, inplace=True)
    # print(ex_var)
    fig2, ax2 = plt.subplots(1,2) # type: ignore
    sns.barplot(ex_var, x='component', y=0, ax=ax2[0])

    ex_var = pd.DataFrame(sk_model.explained_variance_ratio_*100)
    ex_var['component'] = np.arange(ex_var.shape[0])
    ex_var.rename({0:'variance'}, inplace=True)
    sns.barplot(ex_var, x='component', y=0, ax=ax2[1])
    plt.show()