import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib as plt
import decimal



if __name__ == '__main__':
	decimal.getcontext().prec=4
	print "load training dataset..."
	train=pd.read_csv('...Digit_Recognizer/train.csv')
	
	print "extracting features from original training set..."
	train_data=train.values[:,1:]
	train_target=train.ix[:,0]
	
	print "initiating PCA..."
	pca=PCA(n_components=0.90,whiten=True)
	train_data_pca=pca.fit_transform(train_data)

	print "starting selecting feature using Random Forest..."
	rfc=RandomForestClassifier()
	rfc.fit(train_data_pca,train_target)
	train_data_pca_final=rfc.transform(train_data_pca)
	
	print "initiating SVM model..."
	svc=SVC(kernel='rbf')

	print "starting Grid search for model evaluation  ..."
	C_power=[decimal.Decimal(x) for x in list(range(-5,17,2))]                                    
	gamma_power=[decimal.Decimal(x) for x in list(range(-15,5,2))]
	grid_search_params={'C':list(np.power(2, C_power)),'gamma':list(np.power(2, gamma_power))}
	gs=GridSearchCV(estimator=svc,param_grid=grid_search_params,scoring='accuracy',n_jobs=-1,cv=3)

	print "training data from original training set..."
	gs.fit(train_data_pca_final,train_target)

	print "loading original test set..."
	test=pd.read_csv('...Digit_Recognizer/test.csv')
		
	print "extract and transform test set data..."
	submit_data=test.values
	submit_data_pca=pca.transform(submit_data)
	submit_data_pca_final=rfc.transform(submit_data_pca)

	print "predicting original test set..."	
	prediction=gs.predict(submit_data_pca_final)

	print "writing out submission file..."
	pd.DataFrame({"ImageId": range(1,len(prediction)+1), "Label": prediction}).to_csv('PCA-SVM.csv', index=False, header=True)
	