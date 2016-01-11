"""
	The script generate submission file for the kaggle contest "Digit recognizer",An
	image recognition contest whose challenge was to classify handwritten single digits

	dimensionality reduction with PCA+randomforest selection
	Predictive model: svm
"""


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import decimal



def main():
	decimal.getcontext().prec=4
	#loading original train dataset
	train_data=pd.read_csv('...Digit_Recognizer/train.csv')
	
	#features from training set
	train_features=train.values[:,1:]
	
	#target from training set
	train_target=train.ix[:,0]
	
	#pre-processing train features
	my_pca=PCA(n_components=0.90,whiten=True)
	pca_train_features=pca.fit_transform(train_features)

	#selecting feature using Random Forest
	rfc=RandomForestClassifier()
	rfc.fit(pca_train_features,train_target)
	final_train_features=rfc.transform(pca_train_features)
	
	#training SVM model
	model=SVC(kernel='rbf')

	#Grid search for model evaluation 
	C_power=[decimal.Decimal(x) for x in list(range(-5,17,2))]                                    
	gamma_power=[decimal.Decimal(x) for x in list(range(-15,5,2))]
	grid_search_params={'C':list(np.power(2, C_power)),'gamma':list(np.power(2, gamma_power))}
	gs=GridSearchCV(estimator=svc,param_grid=grid_search_params,scoring='accuracy',n_jobs=-1,cv=3)

	#fitting the model
	gs.fit(final_train_features,train_target)

	#loading original test dataset
	test_data=pd.read_csv('...Digit_Recognizer/test.csv')
		
	#pro-processing test features
	test_features=test_data.values
	pca_test_features=my_pca.transform(test_features)
	final_test_features=rfc.transform(pca_test_features)

	#predicting from test set	
	prediction=gs.predict(final_test_features)

	#preparing submission file
	pd.DataFrame({"ImageId": range(1,len(prediction)+1), "Label": prediction}).to_csv('second_attempt.csv', index=False, header=True)

if __name__ == '__main__':
		main()