import numpy as np
import load
import util

def estimate(trainX, trainY, resample_num):
	sample_pos_means = []
	sample_pos_covs = []
	sample_neg_means = []
	sample_neg_covs = []

	for i in range(resample_num):
		[sampledX, sampledY] = util.resample(trainX, trainY)
		[positiveX, negativeX] = util.split(sampledX, sampledY)

		sample_pos_means.append(np.mean(positiveX, 0))
		sample_neg_means.append(np.mean(negativeX, 0))
		sample_pos_covs.append(np.cov(np.array(positiveX).T))
		sample_neg_covs.append(np.cov(np.array(negativeX).T))

	nominal_pos_mean = np.mean(sample_pos_means, 0)
	nominal_neg_mean = np.mean(sample_neg_means, 0)
	nominal_pos_cov = np.mean(sample_pos_covs, 0)
	nominal_neg_cov = np.mean(sample_neg_covs, 0)

	sample_pos_means_cov = np.cov(np.array(sample_pos_means).T)
	sample_neg_means_cov = np.cov(np.array(sample_neg_means).T)


	np.linalg.cholesky(sample_pos_means_cov+ np.eye(sample_pos_means_cov.shape[0]) * 1e-8)
	np.linalg.cholesky(sample_neg_means_cov+ np.eye(sample_neg_means_cov.shape[0]) * 1e-8)
	P_pos = np.linalg.inv(sample_pos_means_cov + np.eye(sample_pos_means_cov.shape[0]) * 1e-8) / len(trainX)
	P_neg = np.linalg.inv(sample_neg_means_cov + np.eye(sample_pos_means_cov.shape[0]) * 1e-8) / len(trainX)
	np.linalg.cholesky(P_pos+ np.eye(sample_neg_means_cov.shape[0]) * 1e-3)
	np.linalg.cholesky(P_neg+ np.eye(sample_neg_means_cov.shape[0]) * 1e-3)

	rho_pos = 0
	rho_neg = 0

	for cov_matrix in sample_pos_covs:
		dis = np.linalg.norm(cov_matrix - nominal_pos_cov, ord='fro')
		rho_pos = max(dis, rho_pos) # type: ignore

	for cov_matrix in sample_neg_covs:
		dis = np.linalg.norm(cov_matrix - nominal_neg_cov, ord='fro')
		rho_neg = max(dis, rho_neg) # type: ignore

	return [nominal_pos_mean, P_pos, nominal_neg_mean, P_neg,
		nominal_pos_cov, rho_pos, nominal_neg_cov, rho_neg]

def mainRobustFisherLDAtest(dataset, alpha, resample_num=100, split_token=','):
	data_file = dataset + '/' + dataset + '.data'
	data_loader = load.loader(file_name = data_file, split_token = split_token)
	[dataX, dataY] = data_loader.load()
	dimension = data_loader.dimension

	[trainX, trainY, testX, testY] = util.divide(dataX, dataY, alpha)

	[pos_mean, pos_P, neg_mean, neg_P, pos_cov, pos_rho, neg_cov, neg_rho] = estimate(trainX, trainY, resample_num)

	M = pos_cov + neg_cov + np.eye(dimension) * (pos_rho + neg_rho)
	M0 = np.linalg.inv(M)

	M1 = pos_P
	M2 = neg_P
	[train_pos_X, train_neg_X] = util.split(trainX, trainY)
	k1 = np.mean(train_pos_X, axis = 0).reshape(dimension, 1)
	k2 = np.mean(train_neg_X, axis = 0).reshape(dimension, 1)
	k1 = k1 / np.linalg.norm(k1)
	k2 = k2 / np.linalg.norm(k2)
	k1_norm = util.M_norm(M1, k1)
	k2_norm = util.M_norm(M2, k2)
	x1 = k1 / k1_norm
	x2 = k2 / k2_norm
	pos_mean = pos_mean.reshape(dimension, 1)
	neg_mean = neg_mean.reshape(dimension, 1)

	while True:
		tail = np.dot(M0, x1 - x2 + pos_mean - neg_mean)
		k1_head = (np.eye(dimension) * k1_norm ** 2 - np.dot(M1, np.dot(k1, k1.T))) / (k1_norm ** 3)
		k2_head = - (np.eye(dimension) * k2_norm ** 2 - np.dot(M2, np.dot(k2, k2.T))) / (k2_norm ** 3)
		k1_gradient = np.dot(k1_head, tail)
		k2_gradient = np.dot(k2_head, tail)
		k1 -= k1_gradient * 0.01
		k2 -= k2_gradient * 0.01
	
		if np.linalg.norm(np.concatenate((k1_gradient, k2_gradient), axis = 0)) < 1e-5:
			break
		k1_norm = util.M_norm(M1, k1)
		k2_norm = util.M_norm(M2, k2)
		x1 = k1 / k1_norm
		x2 = k2 / k2_norm

	w = np.dot(M0, x1 - x2 + pos_mean - neg_mean).reshape(dimension)

	train_pos_mean = np.mean(train_pos_X, axis = 0)
	train_neg_mean = np.mean(train_neg_X, axis = 0)
	threshold = np.dot(w, (train_pos_mean + train_neg_mean) / 2.0)
	positive_lower = True if np.dot(train_pos_mean - train_neg_mean, w) > 0 else False

	predict = np.zeros(len(testY))
	testNum = len(testY)
	for i in range(testNum):
		value = np.dot(testX[i], w)
		if (value > threshold) == positive_lower:
			predict[i] = 1
		else:
			predict[i] = -1

	rightNum = 0
	for i in range(testNum):
		if predict[i] == testY[i]:
			rightNum += 1

	return float(rightNum)/float(testNum)

if __name__ == '__main__':

	dataset = ['ionosphere', 'sonar']  # choose the dataset
	dataset = dataset[0]
	sol = mainRobustFisherLDAtest(dataset, 0.5)
	print(sol)