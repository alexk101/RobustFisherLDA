from load import loader as loader
from sklearn.preprocessing import LabelEncoder
import numpy as np
from log import log
from matplotlib import pyplot as plt
import util


def computeMeanVec(X, y, uniqueClass):
    """
    Step 1: Computing the d-dimensional mean vectors for different class
    """
    mean_vectors = []
    for cl in uniqueClass:
        mask = y==cl
        mean_vectors.append(np.mean(X[mask], axis=0))
    return np.stack(mean_vectors)


def computeWithinScatterMatrices(X, y, feature_no, uniqueClass, mean_vectors):
    # 2.1 Within-class scatter matrix
    S_W = np.zeros((feature_no, feature_no))
    for cl, mv in zip(uniqueClass, mean_vectors):
        for row in X[y == cl]:
            diff = row-mv
            S_W += np.outer(diff, diff)
    return S_W


def computeBetweenClassScatterMatrices(X, y, feature_no, mean_vectors, classes):
    # 2.2 Between-class scatter matrix
    overall_mean = np.mean(X, axis=0)

    S_B = np.zeros((feature_no, feature_no))
    for i, mean_vec in zip(classes, mean_vectors): # modified for multiclass
        n = X[y==i,:].shape[0] # modified for multiclass
        mean_vec = mean_vec.reshape(feature_no, 1) # make column vector
        overall_mean = overall_mean.reshape(feature_no, 1) # make column vector
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return S_B


def computeEigenDecom(S_W, S_B):
    """
    Step 3: Solving the generalized eigenvalue problem for the matrix S_W^-1 * S_B
    """
    m = 10^-6 # add a very small value to the diagonal of your matrix before inversion
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W+np.eye(S_W.shape[1])*m).dot(S_B))
    ex_var = (eig_vals / np.sum(eig_vecs))*100
    return eig_vals, eig_vecs, ex_var


def selectFeature(eig_vals, eig_vecs, feature_no):
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
    return W


def transformToNewSpace(X, W, mean_vectors):
    """
    Step 5: Transforming the samples onto the new subspace
    """
    X_trans = X.dot(W)
    mean_vecs_trans = []
    for mv in mean_vectors:
        mean_vecs_trans.append(mv.dot(W))
    return X_trans, np.array(mean_vecs_trans)


def computeErrorRate(X_trans, mean_vecs_trans, y):
    """
    Compute the error rate
    """

    """
    Project to the second largest eigenvalue
    """
    uniqueClass = np.unique(y)
    threshold = 0
    for i in range(len(uniqueClass)):
        threshold += mean_vecs_trans[i][1]
    threshold /= len(uniqueClass)
    log("threshold: {}".format(threshold))

    errors = 0
    for (i,cl) in enumerate(uniqueClass):
        label = cl
        tmp = X_trans[y==label, 1]
        # compute the error numbers for class i
        num = len(tmp[tmp<threshold]) if mean_vecs_trans[i][1] > threshold else len(tmp[tmp>=threshold])
        log("error rate in class {} = {}".format(i, num*1.0/len(tmp)))
        errors += num


    errorRate = errors*1.0/X_trans.shape[0]
    log("Error rate for the second largest eigenvalue = {}".format(errorRate))
    log("Accuracy for the second largest eigenvalue = {}".format(1-errorRate))


    """
    Project to the largest eigenvalue - and return
    """
    uniqueClass = np.unique(y)
    threshold = 0
    for i in range(len(uniqueClass)):
        threshold += mean_vecs_trans[i][0]
    threshold /= len(uniqueClass)
    log("threshold: {}".format(threshold))

    errors = 0
    for (i,cl) in enumerate(uniqueClass):
        label = cl
        tmp = X_trans[y==label, 0]
        # compute the error numbers for class i
        num = len(tmp[tmp<threshold]) if mean_vecs_trans[i][0] > threshold else len(tmp[tmp>=threshold])
        log("error rate in class {} = {}".format(i, num*1.0/len(tmp)))
        errors += num


    errorRate = errors*1.0/X_trans.shape[0]
    log("Error rate = {}".format(errorRate))
    log("Accuracy = {}".format(1-errorRate))

    return 1-errorRate, threshold


def plot_step_lda(X_trans, y, label_dict, uniqueClass, dataset, threshold):

    ax = plt.subplot(111)
    for label,marker,color in zip(range(1, len(uniqueClass)+1),('^', 's'),('blue', 'red')):
        plt.scatter(x=X_trans[:,0].real[y == label],
                    y=X_trans[:,1].real[y == label],
                    marker=marker, # type: ignore
                    color=color,
                    alpha=0.5,
                    label=label_dict[label]
                    )

    plt.xlabel('LDA1')
    plt.ylabel('LDA2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('Fisher LDA: {} data projection onto the first 2 linear discriminants'.format(dataset))

    # plot the the threshold line
    [bottom, up] = ax.get_ylim()
    #plt.axvline(x=threshold.real, ymin=bottom, ymax=0.3, linewidth=2, color='k', linestyle='--')
    plt.axvline(threshold.real, linewidth=2, color='g')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    #plt.tight_layout
    plt.show()


def mainFisherLDAtest(dataset='sonar', alpha=0.5):
    # load data
    path = dataset + '/' + dataset + '.data'
    load = loader(path)
    [X, y] = load.load()
    [X, y, testX, testY] = util.divide(X, y, alpha)
    X = np.array(X)
    testX = np.array(testX)

    feature_no = X.shape[1] # define the dimension
    sample_no = X.shape[0] # define the sample number

    # preprocessing
    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y) + 1 # type: ignore
    testY = label_encoder.transform(testY) + 1 # type: ignore
    uniqueClass = np.unique(y) # define how many class in the outputs
    label_dict = {}   # define the label name
    for i in range(1, len(uniqueClass)+1):
        label_dict[i] = "Class"+str(i)
    log(label_dict)

    print(f'x: {X}, {X.shape}')
    print(f'y: {y}, {y.shape}')
    # Step 1: Computing the d-dimensional mean vectors for different class
    print(f'unique: {uniqueClass}')
    mean_vectors = computeMeanVec(X, y, uniqueClass)
    print(f'mean_vectors: {mean_vectors}')

    # Step 2: Computing the Scatter Matrices
    S_W = computeWithinScatterMatrices(X, y, feature_no, uniqueClass, mean_vectors)
    S_B = computeBetweenClassScatterMatrices(X, y, feature_no, mean_vectors, uniqueClass)

    # Step 3: Solving the generalized eigenvalue problem for the matrix S_W^-1 * S_B
    eig_vals, eig_vecs, ex_var = computeEigenDecom(S_W, S_B)

    # Step 4: Selecting linear discriminants for the new feature subspace
    W = selectFeature(eig_vals, eig_vecs, feature_no)

    # Step 5: Transforming the samples onto the new subspace
    X_trans, mean_vecs_trans = transformToNewSpace(testX, W, mean_vectors)
    print(f'testX: {testX.shape}')
    print(f'X_trans: {X_trans.shape}')

    # Step 6: compute error rate
    accuracy, threshold = computeErrorRate(X_trans, mean_vecs_trans, testY)


    # plot
    #plot_step_lda(X_trans, testY, label_dict, uniqueClass, dataset, threshold)

    return accuracy


if __name__ == "__main__":
    dataset = ['ionosphere', 'sonar']  # choose the dataset
    alpha = 0.6 # choose the train data percentage
    accuracy = mainFisherLDAtest(dataset[1], alpha)
    print(accuracy)


