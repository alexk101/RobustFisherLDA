import unittest
from hypothesis import given, settings
from hypothesis.extra import numpy
from hypothesis.strategies import integers, composite
import jax.numpy as jnp
from rich import print
from JaxLDA import *


@composite
def training_data(draw):
    x_n = draw(integers(min_value=200, max_value=1000))
    f = draw(integers(min_value=2, max_value=10))

    x = draw(numpy.arrays(shape=(x_n, f), dtype="float32"))
    y = draw(numpy.arrays(shape=(1, x_n), elements=integers(min_value=2, max_value=10), dtype="int32"))[0,...]
    return x, y


class TestClassMeans(unittest.TestCase):
    @given(training_data())
    @settings(deadline=None)
    def test_decode_inverts_encode(self, data):
        x, y = data
        n = np.unique(y)

        # Numpy version
        np_mean = np_computeMeanVec(x, y, n)

        # JAX version
        jax_mean = computeAllMeanVec(jnp.array(x), jnp.array(y), jnp.array(n))

        err = jnp.mean(jnp.abs(jnp.array(np_mean)-jax_mean))
        print(err)
        self.assertTrue(np.isnan(err) or err <= 1e-3)

if __name__ == "__main__":
    unittest.main()