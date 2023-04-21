import jax
from jax import random
import jax.numpy as jnp
from typing import Callable
import timeit
import JaxLDA
import FisherLDA
import numpy as np
import warnings
from rich import print

KEY = jax.random.PRNGKey(42)


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
    mean_org = FisherLDA.computeMeanVec(X_np, y_np, n_np)
    np_within = FisherLDA.computeWithinScatterMatrices(X_np, y_np, features, n_np, mean_org)
    np_between = FisherLDA.computeBetweenClassScatterMatrices(X_np, y_np, features, mean_org, n_np)
    np_e_val, np_e_vec = FisherLDA.computeEigenDecom(np_within, np_between)
    np_W = FisherLDA.selectFeature(np_e_val, np_e_vec, red)
    np_red = FisherLDA.transformToNewSpace(X_np, np_W, mean_org)

    X_jax = jnp.array(X_np, dtype=jnp.float32)
    y_jax = jnp.array(y_np, dtype=jnp.int32)
    n_jax = jnp.array(n_np, dtype=jnp.int32)

    # Optimized
    mean_vectors = JaxLDA.computeAllMeanVec(X_jax, y_jax, n_jax)

    mask = JaxLDA.get_mask(y_jax,n_jax)
    jax_within = JaxLDA.computeWithinScatterMatrices(X_jax, mask, mean_vectors, features)
    jax_between = JaxLDA.computeBetweenClassScatterMatrices(X_jax, y_jax, mean_vectors, n_jax, features)
    jax_e_val, jax_e_vec = JaxLDA.computeEigenDecom(jax_within, jax_between)
    jax_W = JaxLDA.selectFeature(np_e_val, np_e_vec, red)
    jax_red = JaxLDA.transformToNewSpace(X_jax, jax_W, mean_vectors)


    # Timing
    # Func 1
    np_args = X_np, y_np, n_np
    jax_args = X_jax, y_jax, n_jax
    time_func(FisherLDA.computeMeanVec, JaxLDA.computeAllMeanVec, np_args, jax_args)

    # # Func 2
    np_args = X_np, y_np, features, n_np, mean_org
    jax_args = X_jax, mask, mean_vectors, features
    time_func(FisherLDA.computeWithinScatterMatrices, JaxLDA.computeWithinScatterMatrices, np_args, jax_args)

    # Func 3
    np_args = X_np, y_np, features, mean_org, n_np
    jax_args = X_jax, y_jax, mean_vectors, n_jax, features
    time_func(FisherLDA.computeBetweenClassScatterMatrices, JaxLDA.computeBetweenClassScatterMatrices, np_args, jax_args)

    # Func 4
    np_args = np_within, np_between
    jax_args = jax_within, jax_between
    time_func(FisherLDA.computeEigenDecom, JaxLDA.computeEigenDecom, np_args, jax_args)

    # Func 5
    np_args = np_e_val, np_e_vec, red
    jax_args = jax_e_val, jax_e_vec, red
    time_func(FisherLDA.selectFeature, JaxLDA.selectFeature, np_args, jax_args)

    # Func 6
    np_args = X_np, np_W, mean_org
    jax_args = X_jax, np_W, mean_vectors
    time_func(FisherLDA.transformToNewSpace, JaxLDA.transformToNewSpace, np_args, jax_args)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        benchmark()