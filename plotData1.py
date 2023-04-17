import sys
import load
import matplotlib.pyplot as plt
from sklearn import decomposition

if len(sys.argv)<2:
	print('Usage: python plotData1.py <data_file>')

[X, Y] = load.loader(sys.argv[1]).load()
label = list(map(lambda x:{-1:'red', 1:'blue'}[x], Y))

pca=decomposition.PCA(3)
X=pca.fit_transform(X)
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],color=label)
plt.show()
exit(0)