# pyright: reportPrivateImportUsage=false

# Reference: https://gist.github.com/jcreinhold/ebf27f997f4c93c2f637c3c900d6388f

from jax import jit, random
import jax
import jax.numpy as jnp
import numpy as np


KEY = jax.random.PRNGKey(42)

@jit
def norm_p(M: jnp.ndarray, p: int):
    return jnp.sum(jnp.power(M, p))

@jit
def mu_init(D: jnp.ndarray):
    return jnp.prod(jnp.array(D.shape)) / (4 * norm_p(D, 2))

@jit
def lambda_init(D_shape: jnp.ndarray):
    return 1 / jnp.sqrt(jnp.max(D_shape))

@jit
def shrink(M, tau):
    return jnp.sign(M) * jax.nn.relu(jnp.abs(M) - tau)  # hack to save memory

@jit
def init_tol(D: jnp.ndarray):
    return 1e-7 * norm_p(jnp.abs(D), 2)

# very cool: ie the difference in svd's
# https://re-ra.xyz/Differences-of-SVD-methods-in-numpy,-tensorflow-and-pytorch/#differences
@jit
def svd_threshold(M, tau):
    U, s, V = jnp.linalg.svd(M, full_matrices=False)
    return jnp.matmul(U, jnp.matmul(jnp.diag(shrink(s, tau)), V))

@jit
def fit(D, mu, mu_inv, Y, S, lmbda, tol=0.01, max_iter=1000):#, iter_print=100):
    def loop_bound(args):
        _, _, _, i, err = args
        return (err > tol) & (i < max_iter)

    def train(args):
        Yk, Lk, Sk, i, _ = args
        Lk = svd_threshold(D - Sk + mu_inv * Yk, mu_inv)
        Sk = shrink(D - Lk + (mu_inv * Yk), mu_inv * lmbda)
        Yk = Yk + mu * (D - Lk - Sk)
        err = norm_p(jnp.abs(D - Lk - Sk), 2) / norm_p(D, 2)
        i += 1
        return Yk, Lk, Sk, i, err

    i, err = 0, jnp.inf
    Sk, Yk, Lk = S, Y, jnp.zeros_like(D)
    _, Lk, Sk, _, _ = jax.lax.while_loop(
        loop_bound,
        train,
        (Yk, Lk, Sk, i, err)
    )
    return Lk, Sk


class RPCA():
    def __init__(self, D, mu=None, lmbda=None) -> None:
        self.D = D
        self.S = jnp.zeros_like(self.D)
        self.Y = jnp.zeros_like(self.D)
        self.mu = mu or mu_init(self.D).item()
        self.mu_inv = 1 / self.mu
        self.lmbda = lmbda or lambda_init(jnp.array(D.shape))

    def fit(self):
        return fit(
            self.D,
            self.mu,
            self.mu_inv,
            self.Y,
            self.S,
            self.lmbda
        )

if __name__ == "__main__":
    features = 20
    samples = 10000
    classes = 5
    red = features // 2

    D = random.uniform(KEY,(samples, features))
    test = RPCA(D)
    d, s = test.fit()
    d = np.array(d.T)
    y = d[0]
    x = np.arange(len(y))
    print(x)
