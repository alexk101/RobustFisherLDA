# pyright: reportPrivateImportUsage=false

import random
import numpy as np
import jax.numpy as jnp
import jax
from jax.experimental import host_callback
from typing import Tuple

def eig(matrix: jnp.ndarray):
	def _eig_host(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
		"""Wraps jnp.linalg.eig so that it can be jit-ed on a machine with GPUs."""
		eigenvalues_shape = jax.ShapeDtypeStruct(matrix.shape[:-1], complex)
		eigenvectors_shape = jax.ShapeDtypeStruct(matrix.shape, complex)
		return host_callback.call(
			# We force this computation to be performed on the cpu by jit-ing and
			# explicitly specifying the device.
			jax.jit(jnp.linalg.eig, backend='cpu'),
			matrix.astype(complex),
			result_shape=(eigenvalues_shape, eigenvectors_shape),
		)
	return jax.jit(_eig_host, device=jax.devices("gpu")[0])(matrix)


def divide(dataX, dataY, alpha):
	'''
	divide a dataset into two parts, usually training and testing set
	'''
	[positiveX, negativeX] = split(dataX, dataY)
	posNum1 = int (len(positiveX) * alpha)
	negNum1 = int (len(negativeX) * alpha)
	posNum2 = len(positiveX) - posNum1
	negNum2 = len(negativeX) - negNum1

	posOrder = np.random.permutation(len(positiveX))
	negOrder = np.random.permutation(len(negativeX))

	dataX1 = []
	dataY1 = []
	dataX2 = []
	dataY2 = []

	for i in range(posNum1):
		dataX1.append(positiveX[posOrder[i]])
		dataY1.append(1)
	for i in range(posNum2):
		dataX2.append(positiveX[posOrder[i + posNum1]])
		dataY2.append(1)
	for i in range(negNum1):
		dataX1.append(negativeX[negOrder[i]])
		dataY1.append(-1)
	for i in range(negNum2):
		dataX2.append(negativeX[negOrder[i + negNum1]])
		dataY2.append(-1)

	dataX1 = jnp.array(dataX1)
	dataY1 = jnp.array(dataY1)
	dataX2 = jnp.array(dataX2)
	dataY2 = jnp.array(dataY2)

	return dataX1, dataY1, dataX2, dataY2

def resample(dataX, dataY):
	'''
	sample an equivalent size dataset uniformly from the original one
	'''
	instances = len(dataX)

	sampleX = []
	sampleY = []

	for i in range(instances):
		chosen = random.randint(0, instances-1)
		sampleX.append(dataX[chosen])
		sampleY.append(dataY[chosen])

	return [sampleX, sampleY]

def split(dataX, dataY):
	'''
	divide the whole dataset into positive and negative ones
	'''
	instances = len(dataX)
	positiveX = []
	negativeX = []

	for i in range(instances):
		if dataY[i] == 1:
			positiveX.append(dataX[i])
		else:
			negativeX.append(dataX[i])

	positiveX = jnp.array(positiveX)
	negativeX = jnp.array(negativeX)

	return positiveX, negativeX

