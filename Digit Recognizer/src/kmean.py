#!/usr/bin/python

import pandas as pd
import numpy as np
from scipy import cluster
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

if __name__ == '__main__':
	print "load training dataset..."
	train=pd.read_csv('...Digit_Recognizer/train.csv')

	print "extracting features from original training set..."
	train_data=train.values[:,1:]
	train_target=train.ix[:,0]

	print "initiating PCA..."
	pca=PCA(n_components=0.90,whiten=True)
	train_data_pca=pca.fit_transform(train_data)

	print "using elbow method to select number of clusters in KMeans model...."
	initial = [cluster.vq.kmeans(train_data_pca,i) for i in range(1,10)]
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot([var for (cent,var) in initial])
	plt.grid(True)
	plt.xlabel('Number of clusters')
	plt.ylabel('distortion')
	plt.title('Elbow for K-means clustering')
	plt.show()

	print "initiating KMean model using number of cluster from Elbow plot..."
	Kmean=KMeans(init='k-means++',n_clusters=6)

	print "training data from original training set..."
	Kmean.fit(train_data_pca,train_target)

	print "loading original test set..."
	test=pd.read_csv('...Digit_Recognizer/test.csv')
		
	print "extract and transform test set data..."
	submit_data=test.values
	submit_data_pca=pca.transform(submit_data)

	print "predicting original test set..."	
	prediction=Kmean.predict(submit_data_pca)

	print "writing out submission file..."
	pd.DataFrame({"ImageId": range(1,len(prediction)+1), "Label": prediction}).to_csv('K-means.csv', index=False, header=True)
