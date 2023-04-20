# pyright: reportPrivateImportUsage=false

import jax
from jax import random, vmap, jit
import jax.numpy as jnp
import numpy as np
import timeit
from rich import print
from typing import Callable
from functools import partial
import util
import warnings
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

KEY = jax.random.PRNGKey(42)


### ComputeMeanVec ###
def np_computeMeanVec(X, y, n):
    """
    Step 1: Computing the d-dimensional mean vectors for different class
    """
    mean_vectors = []
    for cl in n:
        mask = y==cl
        mean_vectors.append(np.mean(X[mask], axis=0))
    return np.stack(mean_vectors)

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

# numpy
def np_computeWithinScatterMatrices(X, y, feature_no, uniqueClass, mean_vectors):
    # 2.1 Within-class scatter matrix
    S_W = np.zeros((feature_no, feature_no))
    for cl, mv in zip(uniqueClass, mean_vectors):
        #  class_sc_mat = np.zeros((feature_no, feature_no))  # scatter matrix for every class
        for row in X[y == cl]:
            # row, mv = row.reshape(feature_no,1), mv.reshape(feature_no,1)   # make column vectors
            # class_sc_mat += (row-mv).dot((row-mv).T)
            diff = row-mv
            S_W += np.outer(diff, diff)
        # S_W += class_sc_mat                                # sum class scatter matrices
    return S_W


@partial(jit, static_argnames=["f_n"])
def jax_computeWithinScatterMatrices(X, mask, mv, f_n):
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
def np_computeBetweenClassScatterMatrices(X, y, feature_no, mean_vectors, classes):
    # 2.2 Between-class scatter matrix
    overall_mean = np.mean(X, axis=0)

    S_B = np.zeros((feature_no, feature_no))
    for i, mean_vec in zip(classes, mean_vectors): # modified for multiclass
        n = X[y==i,:].shape[0] # modified for multiclass
        mean_vec = mean_vec.reshape(feature_no, 1) # make column vector
        overall_mean = overall_mean.reshape(feature_no, 1) # make column vector
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return S_B


@partial(jit, static_argnames=["feature_no"])
def jax_computeBetweenClassScatterMatrices(X, y, mean_vectors, classes, feature_no):
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

def np_computeEigenDecom(S_W, S_B):
    """
    Step 3: Solving the generalized eigenvalue problem for the matrix S_W^-1 * S_B
    """
    m = 10^-6 # add a very small value to the diagonal of your matrix before inversion
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W+np.eye(S_W.shape[1])*m).dot(S_B))
    return eig_vals, eig_vecs

# Shitter
@jit
def jax_computeEigenDecom(S_W, S_B):
    """
    Step 3: Solving the generalized eigenvalue problem for the matrix S_W^-1 * S_B
    """
    m = 10^-6 # add a very small value to the diagonal of your matrix before inversion
    eig_vals, eig_vecs = util.eig(jnp.linalg.inv(S_W+jnp.eye(S_W.shape[1])*m).dot(S_B))
    return jnp.real(eig_vals), eig_vecs


### End computeEigenDecom ###

## Start selectFeature ### 

def np_selectFeature(eig_vals, eig_vecs, feature_no):
    """
    Step 4: Selecting linear discriminants for the new feature subspace
    """
    # 4.1. Sorting the eigenvectors by decreasing eigenvalues
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low by the value of eigenvalue
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    # 4.2. Choosing k eigenvectors with the largest eigenvalues - here I choose the first two eigenvalues
    W = np.hstack([eig_pairs[x][1].reshape(eig_vecs.shape[0], 1) for x in range(feature_no)])
    # log('Matrix W: \n{}'.format(W.real))

    return W

# Something very fishy
@partial(jit, static_argnames=["feature_no"])
def jax_selectFeature(eig_vals, eig_vecs, feature_no):
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

def np_transformToNewSpace(X, W, mean_vectors):
    """
    Step 5: Transforming the samples onto the new subspace
    """
    X_trans = X.dot(W)
    mean_vecs_trans = []
    for mv in mean_vectors:
        mean_vecs_trans.append(mv.dot(W))
    return X_trans, np.array(mean_vecs_trans)


def jax_transformToNewSpace(X, W, mean_vectors):
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

def time_func(f1: Callable, f2: Callable, args1, args2=None):
    print(f"Comparing np and jax {f1.__name__.split('_')[-1]}...")
    # Unoptimized
    org_speed = timeit.timeit(lambda : f1(*args1), number=1000)
    print(f'unoptimized: {org_speed}')
    if args2 is None:
        args2 = args1

    # Optimized
    speed = timeit.timeit(lambda : f2(*args2), number=1000)
    print(f'optimized: {speed}')
    print(f'speed-up: {org_speed/speed:.2f}x\n')

    res1 = f1(*args1)
    res2 = f2(*args2)
    # print(res1)
    # print(res2)
    if not isinstance(res1, tuple):
        err = res2.astype(jnp.float32) - jnp.array(res1, dtype=jnp.float32)
        err = jnp.mean(jnp.abs(err))
    else:
        err = []
        for x,y in zip(res1, res2):
            err_temp = x.astype(jnp.float32) - jnp.array(y, dtype=jnp.float32)
            err.append(jnp.mean(jnp.abs(err_temp)))

    print(f"err: {err}\n")


def benchmark():
    features = 20
    samples = 10000
    classes = 5
    red = features // 2

    X_np = np.array(random.uniform(KEY,(samples, features)))
    y_np = np.array(random.randint(KEY, (1, samples),  0, classes)[0])
    n_np = np.unique(y_np)

    # Reference
    mean_org = np_computeMeanVec(X_np, y_np, n_np)
    np_within = np_computeWithinScatterMatrices(X_np, y_np, features, n_np, mean_org)
    np_between = np_computeBetweenClassScatterMatrices(X_np, y_np, features, mean_org, n_np)
    np_e_val, np_e_vec = np_computeEigenDecom(np_within, np_between)
    np_W = np_selectFeature(np_e_val, np_e_vec, red)
    np_red = np_transformToNewSpace(X_np, np_W, mean_org)

    X_jax = jnp.array(X_np, dtype=jnp.float32)
    y_jax = jnp.array(y_np, dtype=jnp.int32)
    n_jax = jnp.array(n_np, dtype=jnp.int32)
    # Optimized
    mean_vectors = computeAllMeanVec(X_jax, y_jax, n_jax)

    mask = get_mask(y_jax,n_jax)
    jax_within = jax_computeWithinScatterMatrices(X_jax, mask, mean_vectors, features)
    jax_between = jax_computeBetweenClassScatterMatrices(X_jax, y_jax, mean_vectors, n_jax, features)
    jax_e_val, jax_e_vec = jax_computeEigenDecom(jax_within, jax_between)
    jax_W = jax_selectFeature(np_e_val, np_e_vec, red)
    jax_red = jax_transformToNewSpace(X_jax, np_W, mean_vectors)


    # Timing
    # Func 1
    np_args = X_np, y_np, n_np
    jax_args = X_jax, y_jax, n_jax
    time_func(np_computeMeanVec, computeAllMeanVec, np_args, jax_args)

    # # Func 2
    # np_args = X_np, y_np, features, n_np, mean_org
    # jax_args = X_jax, mask, mean_vectors, features
    # time_func(np_computeWithinScatterMatrices, jax_computeWithinScatterMatrices, np_args, jax_args)

    # Func 3
    np_args = X_np, y_np, features, mean_org, n_np
    jax_args = X_jax, y_jax, mean_vectors, n_jax, features
    time_func(np_computeBetweenClassScatterMatrices, jax_computeBetweenClassScatterMatrices, np_args, jax_args)

    # Func 4
    np_args = np_within, np_between
    jax_args = jax_within, jax_between
    time_func(np_computeEigenDecom, jax_computeEigenDecom, np_args, jax_args)

    # Func 5
    np_args = np_e_val, np_e_vec, red
    jax_args = jax_e_val, jax_e_vec, red
    time_func(np_selectFeature, jax_selectFeature, np_args, jax_args)

    # Func 6
    np_args = X_np, np_W, mean_org
    jax_args = X_jax, np_W, mean_vectors
    time_func(np_transformToNewSpace, jax_transformToNewSpace, np_args, jax_args)


def LDA(X, y, n):
    unq = jnp.unique(y)
    mask = get_mask(y,unq)

    mean_vectors = computeMeanVec(X, mask.T)
    within = jax_computeWithinScatterMatrices(X, mask, mean_vectors, X.shape[1])
    between = jax_computeBetweenClassScatterMatrices(X, y, mean_vectors, unq, X.shape[1])
    e_val, e_vec = np_computeEigenDecom(within, between)
    W = np_selectFeature(e_val, e_vec, n)
    red, means_red = jax_transformToNewSpace(X, W, mean_vectors)
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
        print(X.shape)
        print(y.shape)

        # X = random.uniform(KEY,(samples, features))
        # y = random.randint(KEY, (1, samples),  0, classes)[0]

        dim_red = LDA(X, y, red)
        dim_red = pd.DataFrame(dim_red, columns=['x','y'])
        dim_red['class'] = y
        print(dim_red)
        sns.scatterplot(dim_red, x='y', y='x', hue='class')
        plt.show()
        