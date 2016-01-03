#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split,cross_val_score,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from multiprocessing import cpu_count

if __name__== '__main__':
	print "load training dataset..."
	training=pd.read_csv('.../train.csv')

	print "extracting features from original training set..."
	label_encoder= LabelEncoder()
	PdDistrict_transform_training_set=label_encoder.fit_transform(training['PdDistrict'])
	Dow_transform_training_set=label_encoder.fit_transform(training['DayOfWeek'])
	training_data=pd.DataFrame({'X':training['X'],'Y': training['Y'],'PdDistrict_transform':PdDistrict_transform_training_set,'Dow_transform':Dow_transform_training_set})
	training_data=training_data.values
	training_target=training['Category'].astype('category')

	print "initiating model..."
	rfc=RandomForestClassifier()

	print "starting Grid search for model evaluation ..."
	grid_search_params={'criterion':['gini','entropy'],'n_estimators':[16,32,64,128],'max_features':['auto','log2','sqrt']}
	gs=GridSearchCV(estimator=rfc,param_grid=grid_search_params,scoring='log_loss',n_jobs=-1)

	print "training data from original training set..."
	gs.fit(training_data,training_target)

	print "loading original test set..."
	test=pd.read_csv('.../test.csv')

	print "extracting feature from original test set..."
	PdDistrict_transform_test_set=label_encoder.fit_transform(test['PdDistrict'])
	Dow_transform_test_set=label_encoder.fit_transform(test['DayOfWeek'])
	submit_data=pd.DataFrame({'X':test['X'],'Y': test['Y'],'PdDistrict_transform':PdDistrict_transform_test_set,'Dow_transform':Dow_transform_test_set})
	submit_data=submit_data.values

	print "predicting original test set..."
	prediction=gs.predict_proba(submit_data)

	print "writing out submission file..."
	submit=pd.DataFrame({'Id':test['Id']})
	submit[rfc.classes_]=pd.DataFrame(prediction)	    
	submit.to_csv('Random_Forest_crime_classifier.csv', index=False, header=True)


