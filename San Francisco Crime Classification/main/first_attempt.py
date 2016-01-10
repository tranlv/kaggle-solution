"""
	The script generate submission file for the kaggle contest_data "San Francisco Crime Classification", 
	A classification contest_data whose challenge was to Predict the category of crimes that occurred in San Francisco  

	Predictive model: random forest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_data_split,cross_val_score,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from multiprocessing import cpu_count

main():
	#loading original train dataset
	train_data=pd.read_csv('.../train.csv')

	#pro-processing train features
	label_encoder= LabelEncoder()
	PdDistrict_transform_train_data=label_encoder.fit_transform(train_data['PdDistrict'])
	Dow_transform_train_data=label_encoder.fit_transform(train_data['DayOfWeek'])
	train_data=pd.DataFrame({'X':train_data['X'],'Y': train_data['Y'],'PdDistrict_transform':PdDistrict_transform_train_data,'Dow_transform':Dow_transform_train_data})
	train_features=train_data.values
	train_target=train_data['Category'].astype('category')

	# training random forest
	model=RandomForestClassifier()

	print "starting Grid search for model evaluation ..."
	grid_search_params={'criterion':['gini','entropy'],'n_estimators':[16,32,64,128],'max_features':['auto','log2','sqrt']}
	gs=GridSearchCV(estimator=model,param_grid=grid_search_params,scoring='log_loss',n_jobs=-1)

	#fitting the model
	gs.fit(train_features,train_data_target)

	#loading test_data data
	test_data=pd.read_csv('.../test_data.csv')

	#pro-processing test features
	PdDistrict_transform_test_data=label_encoder.fit_transform(test_data['PdDistrict'])
	Dow_transform_test_data=label_encoder.fit_transform(test_data['DayOfWeek'])
	test_features=pd.DataFrame({'X':test_data['X'],'Y': test_data['Y'],'PdDistrict_transform':PdDistrict_transform_test_data,'Dow_transform':Dow_transform_test_data})
	test_features=test_features.values

	#predicting the test set
	prediction=gs.predict_proba(test_features)

	#preparing submission file
	submit=pd.DataFrame({'Id':test_data['Id']})
	submit[rfc.classes_]=pd.DataFrame(prediction)	    
	submit.to_csv('Random_Forest_crime_classifier.csv', index=False, header=True)

if __name__=="__main__":
	main()

