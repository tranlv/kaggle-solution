"""
	The script generate submission file for the kaggle contest "Digit recognizer",An
	image recognition contest whose challenge was to classify handwritten single digits

	dimensionality reduction with PCA
	Predictive model: Kmeans
"""


import pandas as pd
import numpy as np
from scipy import cluster
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def main():
	#loading original train dataset
	train=pd.read_csv('/train.csv')

	#features from training set
	train_data=train.values[:,1:]
	
	#targetf rom training set
	train_target=train.ix[:,0]

	#pro-processing train features
	my_pca=PCA(n_components=0.90,whiten=True)
	train_data_pca=my_pca.fit_transform(train_data)

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

	# training Kmean
	model=KMeans(init='k-means++',n_clusters=6)

	#fitting the model
	Kmean.fit(train_data_pca,train_target)

	#loading original test dataset
	test_data=pd.read_csv('/test.csv')
		
	#features from test data
	test_features=test_data.values
	
	#pro-processing test features
	pca_test_features=my_pca.transform(test_features)

	#predicting from test set
	prediction=model.predict(pca_test_features)

	#preparing submission file
	pd.DataFrame({"ImageId": range(1,len(prediction)+1), "Label": prediction}).to_csv('K-means.csv', index=False, header=True)

if __name__=="__main__":
	main()
