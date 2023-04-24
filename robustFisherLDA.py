# pyright: reportPrivateImportUsage=false

import numpy as np
import load
import util
import jax.numpy as jnp
from jax import jit, vmap
import jax
from functools import partial

def rho(sample_covs: jnp.ndarray, nominal_cov: jnp.ndarray):
	@jit
	def rho_calc(sample_covs: jnp.ndarray, nominal_cov: jnp.ndarray):
		def single_rho(cov_matrix: jnp.ndarray):
			return jnp.linalg.norm(cov_matrix - nominal_cov, ord='fro').astype(float)

		vmap_rho = vmap(single_rho, in_axes=(0))
		return jnp.max(vmap_rho(sample_covs))
	return max(rho_calc(sample_covs, nominal_cov), 0.0)


def estimate(trainX, trainY, resample_num):
	sample_pos_means = []
	sample_pos_covs = []
	sample_neg_means = []
	sample_neg_covs = []

	for i in range(resample_num):
		[sampledX, sampledY] = util.resample(trainX, trainY)
		positiveX, negativeX = util.split(sampledX, sampledY)

		sample_pos_means.append(jnp.mean(positiveX, 0))
		sample_neg_means.append(jnp.mean(negativeX, 0))
		sample_pos_covs.append(jnp.cov(positiveX.T))
		sample_neg_covs.append(jnp.cov(negativeX.T))

	sample_pos_covs = jnp.array(sample_pos_covs)
	sample_neg_covs = jnp.array(sample_neg_covs)
	sample_pos_means = jnp.array(sample_pos_means)
	sample_neg_means = jnp.array(sample_neg_means)

	nominal_pos_mean = jnp.mean(sample_pos_means, 0)
	nominal_neg_mean = jnp.mean(sample_neg_means, 0)
	nominal_pos_cov = jnp.mean(sample_pos_covs, 0)
	nominal_neg_cov = jnp.mean(sample_neg_covs, 0)

	sample_pos_means_cov = jnp.cov(sample_pos_means.T)
	sample_neg_means_cov = jnp.cov(sample_neg_means.T)

	P_pos = jnp.linalg.inv(sample_pos_means_cov + jnp.eye(sample_pos_means_cov.shape[0]) * 1e-8) / len(trainX)
	P_neg = jnp.linalg.inv(sample_neg_means_cov + jnp.eye(sample_pos_means_cov.shape[0]) * 1e-8) / len(trainX)
	rho_pos = rho(sample_pos_covs, nominal_pos_cov)
	rho_neg = rho(sample_neg_covs, nominal_neg_cov)

	return (nominal_pos_mean, P_pos, nominal_neg_mean, P_neg,
		nominal_pos_cov, rho_pos, nominal_neg_cov, rho_neg)

@jit
def accuracy(predict: jnp.ndarray, testY: jnp.ndarray):
	return jnp.sum(predict==testY)/testY.size


@partial(jit, static_argnames=["n"])
def predict(n: int, testX: jnp.ndarray, threshold: float, positive_lower: bool, w: jnp.ndarray):
	def predict_single(x, predict):
		pred = (jnp.dot(x, w) > threshold) == positive_lower
		p1 = lambda val: val+1
		p2 = lambda val: val-1
		return jax.lax.cond(pred, p1, p2, predict)
	predict = jnp.zeros(n)
	vmap_predict = vmap(predict_single, in_axes=(0, 0))
	return vmap_predict(testX, predict)


@partial(jit, static_argnames=["n_features", "dimension"], backend='cpu')
def train(M0, pos_mean, neg_mean, dimension, M1, M2, k1_in, k2_in, n_features, learning_rate: float=0.1, tol: float=1e-5):
	def loop_bound(args):
		k1_gradient, k2_gradient = args[-2], args[-1]
		loss = jnp.linalg.norm(jnp.concatenate((k1_gradient, k2_gradient), axis = 0))
		jax.debug.print('loss: {x}', x=loss)
		return loss > tol

	def gradient(M, k, k_norm, tail, sign: int):
		k_head = sign * (jnp.eye(dimension) * k_norm ** 2 - jnp.dot(M, jnp.dot(k, k.T))) / (k_norm ** 3)
		return jnp.dot(k_head, tail)

	def M_norm(matrix, vector):
		squared = vector.T @ matrix @ vector
		old = jnp.sqrt(squared[0][0])
		return old

	def iteration(args):
		x1, x2, k1, k2, k1_norm, k2_norm, k1_gradient, k2_gradient = args
		tail = jnp.dot(M0, x1 - x2 + pos_mean - neg_mean)
		k1_gradient = gradient(M1, k1, k1_norm, tail, 1)
		k2_gradient = gradient(M2, k2, k2_norm, tail, -1)
		k1 -= k1_gradient * learning_rate
		k2 -= k2_gradient * learning_rate
		k1_norm = M_norm(M1, k1)
		k2_norm = M_norm(M2, k2)
		x1 = k1 / k1_norm
		x2 = k2 / k2_norm
		return x1, x2, k1, k2, k1_norm, k2_norm, k1_gradient, k2_gradient

	k1_norm = M_norm(M1, k1_in)
	k2_norm = M_norm(M2, k2_in)
	x1 = k1_in / k1_norm
	x2 = k2_in / k2_norm
	k1_gradient, k2_gradient = jnp.full((n_features, 1), jnp.inf), jnp.full((n_features, 1), jnp.inf)
	result = jax.lax.while_loop(
        loop_bound,
        iteration,
        (x1, x2, k1_in, k2_in, k1_norm, k2_norm, k1_gradient, k2_gradient)
    )
	return result[0], result[1]


def mainRobustFisherLDAtest(dataset, alpha, resample_num=100, split_token=','):
	data_file = dataset + '/' + dataset + '.data'
	data_loader = load.loader(file_name = data_file, split_token = split_token)
	[dataX, dataY] = data_loader.load()
	n_features = data_loader.n_features

	[trainX, trainY, testX, testY] = util.divide(dataX, dataY, alpha)

	[pos_mean, pos_P, neg_mean, neg_P, pos_cov, pos_rho, neg_cov, neg_rho] = estimate(trainX, trainY, resample_num)

	M = pos_cov + neg_cov + jnp.eye(n_features) * (pos_rho + neg_rho)
	M0 = jnp.linalg.inv(M)

	M1 = pos_P
	M2 = neg_P
	[train_pos_X, train_neg_X] = util.split(trainX, trainY)
	train_pos_X = jnp.array(train_pos_X)
	train_neg_X = jnp.array(train_neg_X)
	k1 = jnp.mean(train_pos_X, axis = 0).reshape(n_features, 1)
	k2 = jnp.mean(train_neg_X, axis = 0).reshape(n_features, 1)
	k1 = k1 / jnp.linalg.norm(k1)
	k2 = k2 / jnp.linalg.norm(k2)
	pos_mean = pos_mean.reshape(n_features, 1)
	neg_mean = neg_mean.reshape(n_features, 1)

	x1, x2 = train(M0, pos_mean, neg_mean, n_features, M1, M2, k1, k2, n_features)

	w = jnp.dot(M0, x1 - x2 + pos_mean - neg_mean).reshape(n_features)
	print(f'w: {w}')

	train_pos_mean = jnp.mean(train_pos_X, axis = 0)
	train_neg_mean = jnp.mean(train_neg_X, axis = 0)
	threshold = float(jnp.dot(w, (train_pos_mean + train_neg_mean) / 2.0))
	print(f'threshold: {threshold}')
	positive_lower = True if jnp.dot(train_pos_mean - train_neg_mean, w) > 0 else False

	predictions = predict(testY.size, testX, threshold, positive_lower, w)
	print(predictions)
	return accuracy(predictions, jnp.array(testY))

if __name__ == '__main__':

	dataset = ['ionosphere', 'sonar']  # choose the dataset
	dataset = dataset[0]
	sol = mainRobustFisherLDAtest(dataset, 0.5)
	print(sol)