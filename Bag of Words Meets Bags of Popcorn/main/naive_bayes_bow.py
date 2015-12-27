#!/usr/bin/python

import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer,CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from scipy.stats import sem
from sklearn.cross_validation import cross_val_score,KFold

def evaluate_cross_validation(nb,X,y,K):
	cv=KFold(len(y),K,shuffle=True,random_state=0)
	scores=cross_val_score(nb,X,y,scoring='roc_auc',cv=cv)
	print scores
	print ("Mean score:{0:.3f}(+/-{1:.3d})").format(np.mean(scores),sem(scores))
	return np.mean(scores)
	
def lemmatize(token, tag):
	if tag[0] in ['v','n']:
		return lemmatizer.lemmatize(token, tag[0])
	return token

def corpus_preprocessing(corpus):
	preprocessed_corpus = []
	for sentence in corpus:	
		#remove HTML/XML characters and punctuation
		soup=BeautifulSoup(sentence).get_text()
		letters_only = re.sub("[^a-zA-Z]", " ",soup )
				
		#Stemming
		stemmer=PorterStemmer()
		stemmed_words=[stemmer.stem(t) for t in word_tokenize(letters_only.lower()) ]
		
		#lemmatization
		lemmatizer = WordNetLemmatizer()
		tagged_words = pos_tag(stemmed_words)
		stemmed_and_lemmatized_words=[lemmatize(token, tag) for token, tag in tagged_words]
		
		clean_review=" ".join(w for w in stemmed_and_lemmatized_words)
		preprocessed_corpus.append(clean_review)		
		
	return preprocessed_corpus
	
if __name__ == "__main__":
	print "loading and preprocessing original train dataset..."
	train_data=pd.read_csv(".../labeledTrainData.tsv", header=0,delimiter="\t", quoting=3)
	preprocessed_train_data=corpus_preprocessing(train_data.review)	
						
	print "selecting sklean class for representation model..."
	nb1 = Pipeline ([('vect',CountVectorizer(stop_words='english')),('nb',MultinomialNB()),])
	nb2 = Pipeline ([('vect',HashingVectorizer(stop_words='english')),('nb',MultinomialNB()),])
	nb3 = Pipeline ([('vect',TfidfVectorizer (stop_words='english')),('nb',MultinomialNB()),])
	nbs=[nb1,nb2,nb3]
	best_score=0
	for nb in nbs:
		score=evaluate_cross_validation(nb,preprocessed_train_data,train_data.review,5)
		if score>best_score:
			best_score=score
			chosen_model=nb

	print "training bayer Naives model with chosen representation model.... "	
	chosen.model.fit_transform( preprocessed_train_data, train_data.sentiment )	

	print "loading and preprocessing original test dataset..."
	test_data = pd.read_csv(".../testData.tsv", header=0, delimiter="\t",quoting=3 )
	preprocessed_test_data=corpus_preprocessing(test_data.review)

	print "predicting original test set..."	
	prediction = chosen_model.predict(preprocessed_test_data)

	print "writing out submission file..."
	pd.DataFrame( data={"id":test_data["id"], "sentiment":prediction} ).to_csv("naive_Baiyes_bow.csv", index=False, quoting=3 )	

	




