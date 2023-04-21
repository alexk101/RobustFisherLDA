# pyright: reportPrivateImportUsage=false

import jax
from jax import vmap, jit
import jax.numpy as jnp
import numpy as np
from functools import partial
import util
import warnings
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import FisherLDA

### ComputeMeanVec ###
@jit
def get_mask(y: jnp.ndarray, n: jnp.ndarray):
    def _get_mask(y: jnp.ndarray, n: jnp.ndarray):
        mask = y==n
        return mask.flatten()
    vmapMask = vmap(_get_mask, in_axes=(None,0))
    return vmapMask(y, n)

@jit
def computeMeanVec(X: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Compues the d-dimensional mean vectors for the class defined by mask

    Args:
        X (jnp.ndarray): Feature matrix
        mask (jnp.ndarray): Mask defining class membership

    Returns:
        jnp.ndarray: _description_
    """
    def _computeMeanVec(X: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(X, where=mask.reshape(-1,1), axis=0)
    vmapCMV = vmap(_computeMeanVec, in_axes=(None,1))
    return vmapCMV(X, mask)

@jit
def computeAllMeanVec(X, y, n):
    """
    Step 1: Computing the d-dimensional mean vectors for different class
    """
    mask = get_mask(y.reshape(-1,1), n).T
    return computeMeanVec(X,mask)


### End ComputeMeanVec ###
### ComputeWithinScatterMatrix ###

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

### End ComputeWithinScatterMatrix ###

### Start computeBetweenClassScatterMatrices ###

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


### End computeBetweenClassScatterMatrices ###

### Start computeEigenDecom ###

# Shitter
@jit
def computeEigenDecom(S_W, S_B):
    """
    Step 3: Solving the generalized eigenvalue problem for the matrix S_W^-1 * S_B
    """
    m = 10^-6 # add a very small value to the diagonal of your matrix before inversion
    eig_vals, eig_vecs = util.eig(jnp.linalg.inv(S_W+jnp.eye(S_W.shape[1])*m).dot(S_B))
    return jnp.real(eig_vals), eig_vecs


### End computeEigenDecom ###

## Start selectFeature ### 

# Something very fishy
@partial(jit, static_argnames=["feature_no"])
def selectFeature(eig_vals, eig_vecs, feature_no):
    """
    Step 4: Selecting linear discriminants for the new feature subspace
    """
    # 4.1. Sorting the eigenvectors by decreasing eigenvalues
    # Make a list of (eigenvalue, eigenvector) tuples
    _, inds = jax.lax.top_k(eig_vals, feature_no)

    out_vec = jnp.zeros((eig_vecs.shape[0], feature_no), jnp.float64)
    
    for out_ind, ind in enumerate(inds):
        out_vec = out_vec.at[:,out_ind].set(eig_vecs[:,ind])
    return out_vec

### End selectFeature ### 

### Start transformToNewSpace ###

def transformToNewSpace(X, W, mean_vectors):
    """
    Step 5: Transforming the samples onto the new subspace
    """
    def apply_class(mv):
        return mv.dot(W)
    X_trans = X.dot(W)
    vmapApplyClass = vmap(apply_class, in_axes=(0))
    mean_vecs_trans = vmapApplyClass(mean_vectors)
    return X_trans, mean_vecs_trans


### End transformToNewSpace ###

class LDA:
    def __init__(self, X, y, n):
        self.X = X
        self.y = y
        self.n = n
        self.unq = jnp.unique(y)
        self.mask = get_mask(y, self.unq)

    def fit(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_vectors = computeMeanVec(X, self.mask.T)
            within = computeWithinScatterMatrices(X, self.mask, mean_vectors, self.X.shape[1])
            between = computeBetweenClassScatterMatrices(X, y, mean_vectors, self.unq, self.X.shape[1])
            e_val, e_vec = FisherLDA.computeEigenDecom(within, between)
            W = FisherLDA.selectFeature(e_val, e_vec, self.n)
            red, means_red = transformToNewSpace(X, W, mean_vectors)
            return red


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = 10
        samples = 1000
        classes = 4
        means = np.array([x+[1]*8 for x in [[1,1],[1,-1],[-1,1],[-1,-1]]])*3
        red = 2

        X = []
        y = []
        for c, mean in zip(range(classes), means):
            X.append(np.random.normal(mean, size=(samples,features)))
            y.append(np.full(samples, c))
        X = np.vstack(X)
        y = np.concatenate(y)

        # X = random.uniform(KEY,(samples, features))
        # y = random.randint(KEY, (1, samples),  0, classes)[0]

        model = LDA(X, y, red)
        dim_red = model.fit()
        dim_red = pd.DataFrame(dim_red, columns=['x','y'])
        dim_red['class'] = y
        sns.scatterplot(dim_red, x='y', y='x', hue='class')
        plt.show()
        