# pyright: reportPrivateImportUsage=false

from numba import jit, int32, float32, boolean
import numpy as np
import timeit
from rich import print
from typing import Callable
import warnings


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

@jit(boolean[:,:](int32[:], int32[:], boolean[:,:]), nopython=True, parallel=True)
def get_mask(y: np.ndarray, n: np.ndarray, mask):
    for ind, c in enumerate(n):
        mask[ind] = y==c
    return mask

@jit(float32[:,:](float32[:,:], int32[:], int32[:], float32[:,:]), nopython=True, parallel=True)
def computeAllMeanVec(X, y, n, mv):
    """
    Step 1: Computing the d-dimensional mean vectors for different class
    """
    for ind, cl in enumerate(n):
        mask = y==cl
        mv[ind] = np.mean(X[mask])
    return mv


# ### End ComputeMeanVec ###
# ### ComputeWithinScatterMatrix ###

# # numpy
# def np_computeWithinScatterMatrices(X, y, feature_no, uniqueClass, mean_vectors):
#     # 2.1 Within-class scatter matrix
#     S_W = np.zeros((feature_no, feature_no))
#     for cl, mv in zip(uniqueClass, mean_vectors):
#         #  class_sc_mat = np.zeros((feature_no, feature_no))  # scatter matrix for every class
#         for row in X[y == cl]:
#             # row, mv = row.reshape(feature_no,1), mv.reshape(feature_no,1)   # make column vectors
#             # class_sc_mat += (row-mv).dot((row-mv).T)
#             diff = row-mv
#             S_W += np.outer(diff, diff)
#         # S_W += class_sc_mat                                # sum class scatter matrices
#     return S_W


# @jit
# def jax_computeWithinScatterMatrices(X, mask, mv):
#     # 2.1 Within-class scatter matrix
#     S_W = np.zeros((X.shape[1], X.shape[1]))
#     for row_mask, mv in zip(mask, mean_vectors):
#         #  class_sc_mat = np.zeros((feature_no, feature_no))  # scatter matrix for every class
#         for row in X[row_mask]:
#             # row, mv = row.reshape(feature_no,1), mv.reshape(feature_no,1)   # make column vectors
#             # class_sc_mat += (row-mv).dot((row-mv).T)
#             diff = row-mv
#             S_W += np.outer(diff, diff)
#         # S_W += class_sc_mat                                # sum class scatter matrices
#     return S_W


# ### End ComputeWithinScatterMatrix ###

# ### Start computeBetweenClassScatterMatrices ###
# def np_computeBetweenClassScatterMatrices(X, y, feature_no, mean_vectors, classes):
#     # 2.2 Between-class scatter matrix
#     overall_mean = np.mean(X, axis=0)

#     S_B = np.zeros((feature_no, feature_no))
#     for i, mean_vec in zip(classes, mean_vectors): # modified for multiclass
#         n = X[y==i,:].shape[0] # modified for multiclass
#         mean_vec = mean_vec.reshape(feature_no, 1) # make column vector
#         overall_mean = overall_mean.reshape(feature_no, 1) # make column vector
#         S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
#     return S_B


# @jit
# def jax_computeBetweenClassScatterMatrices(X, y, mean_vectors, classes, feature_no):
#     overall_mean = np.mean(X, axis=0).reshape(feature_no, 1)

#     S_B = np.zeros((feature_no, feature_no))
#     for i, mean_vec in zip(classes, mean_vectors): # modified for multiclass
#         n = X[y==i,:].shape[0] # modified for multiclass
#         mean_vec = mean_vec.reshape(feature_no, 1) # make column vector
#         diff = mean_vec - overall_mean
#         S_B += n * diff.dot(diff.T)
#     return S_B


# ### End computeBetweenClassScatterMatrices ###

# ### Start computeEigenDecom ###

# def np_computeEigenDecom(S_W, S_B):
#     """
#     Step 3: Solving the generalized eigenvalue problem for the matrix S_W^-1 * S_B
#     """
#     m = 10^-6 # add a very small value to the diagonal of your matrix before inversion
#     eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W+np.eye(S_W.shape[1])*m).dot(S_B))
#     return eig_vals, eig_vecs

# # Shitter
# @jit
# def jax_computeEigenDecom(S_W, S_B):
#     """
#     Step 3: Solving the generalized eigenvalue problem for the matrix S_W^-1 * S_B
#     """
#     m = 10^-6 # add a very small value to the diagonal of your matrix before inversion
#     eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W+np.eye(S_W.shape[1])*m).dot(S_B))
#     return eig_vals, eig_vecs


# ### End computeEigenDecom ###

# ## Start selectFeature

# def selectFeature(eig_vals, eig_vecs, feature_no):
#     """
#     Step 4: Selecting linear discriminants for the new feature subspace
#     """
#     # 4.1. Sorting the eigenvectors by decreasing eigenvalues
#     # Make a list of (eigenvalue, eigenvector) tuples
#     eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

#     # Sort the (eigenvalue, eigenvector) tuples from high to low by the value of eigenvalue
#     eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

#     # 4.2. Choosing k eigenvectors with the largest eigenvalues - here I choose the first two eigenvalues
#     W = np.hstack((eig_pairs[0][1].reshape(feature_no, 1), eig_pairs[1][1].reshape(feature_no, 1)))
#     # log('Matrix W: \n{}'.format(W.real))

#     return W

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
        err = res2.astype(np.float32) - np.array(res1, dtype=np.float32)
        err = np.mean(np.abs(err))
    else:
        err = []
        for x,y in zip(res1, res2):
            err_temp = x.astype(np.float32) - np.array(y, dtype=np.float32)
            err.append(np.mean(np.abs(err_temp)))

    print(f"err: {err}\n")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = 20
        samples = 10000
        classes = 5

        X_np = np.random.uniform(0, 1, (samples, features))
        y_np = np.random.randint(0, classes, (1, samples))[0]
        n_np = np.unique(y_np)

        print('X_np')
        print(X_np)

        print('y_np')
        print(y_np)

        print('n_np')
        print(n_np)

        # Reference
        mean_org = np_computeMeanVec(X_np, y_np, n_np)
        # np_within = np_computeWithinScatterMatrices(X_np, y_np, features, n_np, mean_org)
        # np_between = np_computeBetweenClassScatterMatrices(X_np, y_np, features, mean_org, n_np)
        # np_eigen = np_computeEigenDecom(np_within, np_between)

        X_jax = np.array(X_np, dtype=np.float32)
        y_jax = np.array(y_np, dtype=np.int32)
        n_jax = np.array(n_np, dtype=np.int32)
        # Optimized
        mean_vectors = np.empty((n_jax.size, features), dtype=np.float32)
        mean_vectors = computeAllMeanVec(X_jax, y_jax, n_jax, mean_vectors)

        mask = np.empty(shape=[n_jax.size, y_jax.size], dtype=bool)
        mask = get_mask(y_jax,n_jax, mask)
        # jax_within = jax_computeWithinScatterMatrices(X_jax, mask, mean_vectors)
        # jax_between = jax_computeBetweenClassScatterMatrices(X_jax, y_jax, mean_vectors, n_jax, features)
        # jax_eigen = jax_computeEigenDecom(jax_within, jax_between)

        # Timing
        # Func 1
        np_args = X_np, y_np, n_np
        jax_args = X_jax, y_jax, n_jax, mean_vectors
        time_func(np_computeMeanVec, computeAllMeanVec, np_args, jax_args)

        # # Func 2
        # np_args = X_np, y_np, features, n_np, mean_org
        # jax_args = X_jax, mask, mean_vectors
        # time_func(np_computeWithinScatterMatrices, jax_computeWithinScatterMatrices, np_args, jax_args)

        # # Func 3
        # np_args = X_np, y_np, features, mean_org, n_np
        # jax_args = X_jax, y_jax, mean_vectors, n_jax, features
        # time_func(np_computeBetweenClassScatterMatrices, jax_computeBetweenClassScatterMatrices, np_args, jax_args)

        # # Func 3
        # np_args = np_within, np_between
        # jax_args = jax_within, jax_between
        # time_func(np_computeEigenDecom, jax_computeEigenDecom, np_args, jax_args)