"""
	The script generate score for the performance evaluation for kaggle contest "bag of words meets bag of popcorn",
using split data from only train data set	
	Text represents using tf-idf
	Predictive model: naive_bayes
"""


from sklearn.metric import roc_auc_score,roc_curve
import first_attempt
from sklearn.cross_validation import train_test_split	

def main():
	#loading and preprocessing original train dataset
	train_data=pd.read_csv("/data/labeledTrainData.tsv", header=0,delimiter="\t", quoting=3)
	
	# Split 80-20 train vs test data
	split_train_features, split_test_features, split_train_target, split_test_target= 
								train_test_split(train_features,train_target,test_size=0.20,random_state=0)
	#pre-processing split train 
	vectorizer= TfidfVectorizer(stop_words='english')
	split_train_features=corpus_preprocessing(split_train_features)
	split_train_features=vectorizer.fit_transform(split_train_features)
	tsvd=TruncatedSVD(100)
	tsvd.fit(split_train_features)
	split_train_features=tsvd.transform(split_train_features)

	#pre-processing split test features
	split_test_features=corpus_preprocessing(split_test_features)
	split_test_features=vectorizer.transform(split_test_features)
	split_test_features=tsvd.transform(split_test_features=)

	#fit and predict using split data
	model.fit(split_train_features,split_train_target)
	split_prediction=model.predict(split_test_features)
	score=roc_auc_score(split_test_target, split_predict)
	print (score(split_test_target, split_predict))
	

if __name__=="__main__":
	main()