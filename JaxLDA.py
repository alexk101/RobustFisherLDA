# pyright: reportPrivateImportUsage=false

import jax
from jax import vmap, jit
import jax.numpy as jnp
import numpy as np
from functools import partial
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import FisherLDA


@jit
def get_mask(y: jnp.ndarray, n: jnp.ndarray):
    def _get_mask(y: jnp.ndarray, n: jnp.ndarray):
        mask = y==n
        return mask.flatten()
    vmapMask = vmap(_get_mask, in_axes=(None,0))
    return vmapMask(y, n)


@jit
def computeMeanVec(X: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    def _computeMeanVec(X: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(X, where=mask.reshape(-1,1), axis=0)
    vmapCMV = vmap(_computeMeanVec, in_axes=(None,1))
    return vmapCMV(X, mask)


@jit
def computeAllMeanVec(X, y, n):
    mask = get_mask(y.reshape(-1,1), n).T
    return computeMeanVec(X,mask)


@partial(jit, static_argnames=["f_n"])
def computeWithinScatterMatrices(X, mask, mv, f_n):
    def jax_bruh_1(mask, mv):
        def _jax_bruh(row, mv):
            diff = row-mv
            return jnp.outer(diff, diff)

        vmapBruh = vmap(_jax_bruh, in_axes=(0,0))
        col_mask = mask.reshape(-1,1)
        # This mean mask is kinda a waste of memory, but the only way I could think
        # of to vmap the class means outer product

        mean_mask = jnp.where(col_mask, mv, jnp.zeros(f_n)) # stinky hack for the vmap
        row_mask = jnp.where(col_mask, X, jnp.zeros(f_n))
        return jnp.sum(vmapBruh(row_mask, mean_mask), axis=0)

    jax_bruh_fuck = vmap(jax_bruh_1, in_axes=(0, 0))
    return jnp.sum(jax_bruh_fuck(mask, mv), axis=0)


@partial(jit, static_argnames=["feature_no"])
def computeBetweenClassScatterMatrices(X, y, mean_vectors, classes, feature_no):
    # 2.2 Between-class scatter matrix
    overall_mean = jnp.mean(X, axis=0).reshape(feature_no, 1)
    def computeSingleClass(class_n, mv):
        n = jnp.sum(y==class_n)
        diff = mv.reshape(feature_no, 1) - overall_mean
        return n * jnp.dot(diff, diff.T)
    vmapComputeSingleClass = vmap(computeSingleClass, in_axes=(0, 0))
    return jnp.sum(vmapComputeSingleClass(classes, mean_vectors), axis=0)


# Shitter
@jit
def computeEigenDecom(S_W, S_B):
    m = 10^-6 # add a very small value to the diagonal of your matrix before inversion
    inv = jnp.linalg.inv(S_W+jnp.eye(S_W.shape[1])*m).dot(S_B)
    eig_vectors, eig_values, _ = jnp.linalg.svd(inv)
    ex_var = (eig_values / jnp.sum(eig_values))*100
    return eig_values, eig_vectors, ex_var


# Something very fishy
@partial(jit, static_argnames=["feature_no"])
def selectFeature(eig_vals, eig_vecs, feature_no):
    _, inds = jax.lax.top_k(eig_vals, feature_no)

    out_vec = jnp.zeros((eig_vecs.shape[0], feature_no), jnp.float64)
    
    for out_ind, ind in enumerate(inds):
        out_vec = out_vec.at[:,out_ind].set(eig_vecs[:,ind])
    return out_vec


@jit
def transformToNewSpace(X, W, mean_vectors):
    def apply_class(mv):
        return mv.dot(W)
    X_trans = X.dot(W)
    vmapApplyClass = vmap(apply_class, in_axes=(0))
    mean_vecs_trans = vmapApplyClass(mean_vectors)
    return X_trans, mean_vecs_trans


class LDA:
    def __init__(self, X, y, n):
        self.X = X

        if issubclass(y.dtype.type, jnp.integer):
            self.y = y
        else:
            org_labels = np.unique(y)
            self.label_map = {x:label for x, label in zip(range(len(org_labels)), org_labels)}
            self.unq = jnp.array(list(self.label_map.keys()), dtype=jnp.int32)
            new_y = np.zeros(y.size, dtype=jnp.int32)
            for label, org in self.label_map.items():
                new_y[y==org]=label
            self.y = jnp.array(new_y)
        self.n = n
        
        self.mask = get_mask(self.y, self.unq)

    def fit(self):
        mean_vectors = computeMeanVec(self.X, self.mask.T)
        within = computeWithinScatterMatrices(self.X, self.mask, mean_vectors, self.X.shape[1])
        between = computeBetweenClassScatterMatrices(self.X, self.y, mean_vectors, self.unq, self.X.shape[1])
        self.e_val, self.e_vec, self.ex_var = computeEigenDecom(within, between)
        W = FisherLDA.selectFeature(self.e_val, self.e_vec, self.n)
        self.red, means_red = transformToNewSpace(self.X, W, mean_vectors)
        return self.red


if __name__ == "__main__":

    features = 10
    samples = 1000
    classes = 4
    means = np.array([x+[1]*8 for x in [[1,1],[1,-1],[-1,1],[-1,-1]]])*3
    red = 2

    X = []
    y = []
    for c, mean in zip(range(classes), means):
        X.append(np.random.normal(mean, size=(samples,features)))
        y.append(np.full(samples, f'test{c}'))
    X = np.vstack(X)
    y = np.concatenate(y)

    model = LDA(X, y, red)
    dim_red = model.fit()
    dim_red = pd.DataFrame(dim_red, columns=['x','y'])
    dim_red['class'] = y
    sns.scatterplot(dim_red, x='y', y='x', hue='class')
    plt.show()
    