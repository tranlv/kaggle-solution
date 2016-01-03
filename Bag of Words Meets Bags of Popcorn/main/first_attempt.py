"""
	The script generate submission file for the kaggle contest "bag of words meets bag of popcorn", a 
natural language processing contest whose challenge was to predict IMDB movies sentiment from 
multi-paragraph movie reviews.

	Text represents using tf-idf
	Predictive model: naive_bayes
"""

import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from sklearn.naive_bayes import MultinomialNB


porter=nltk.PorterStemmer()
wnl = nltk.WordNetLemmatizer()

def lemmatize_with_potter(token,tag):
    if tag[0].lower() in ['v','n']:
        return  porter.stem(token)
    return token

def lemmatize_with_WordNet(token,tag):
    if tag[0].lower() in ['v','n']:
        return wnl.lemmatize(token)
    return token

def corpus_preprocessing(corpus):
    preprocessed_corpus = []
    for sentence in corpus:	
        #remove HTML and punctuation
        soup=BeautifulSoup(sentence).get_text()
        letters_only = re.sub("[^a-zA-Z]"," ",soup )

        #Stemming
        tokens=nltk.word_tokenize(letters_only.lower())
        tagged_words=nltk.pos_tag(tokens)
        stemmed_text_with_potter=[lemmatize_with_potter(token,tag) for token,tag in tagged_words]

        #lemmatization
        tagged_words_after_stem=nltk.pos_tag(stemmed_text_with_potter)
        stemmed_and_lemmatized_text=[lemmatize_with_WordNet(token,tag) for token,tag in tagged_words_after_stem]
		
		#join all the tokens
        clean_review=" ".join(w for w in stemmed_and_lemmatized_words)
        preprocessed_corpus.append(clean_review)		

    return preprocessed_corpus
	
def main():
	#loading and preprocessing original train dataset
	train_data=pd.read_csv("C:/Users/vutran/Desktop/github/kaggle/Bag of Words Meets Bags of Popcorn/data/labeledTrainData.tsv", header=0,delimiter="\t", quoting=3)
	
	#features from training set
	train_features=train_data.review

	#pro-processing train features
	train_features=corpus_preprocessing(train_features)
	vectorizer= TfidfVectorizer(stop_words='english')
	train_features=vectorizer.fit_transform(train_features)
	
	# training baive_naives
	model=MultinomialNB()

	#fitting the model
	model.fit(train_features,train_target)
	
	#reading test data
	test_data=pd.read_csv("http://localhost:8888/tree/data/testData.tsv", header=0,delimiter="\t", quoting=3)

	#features from test data
	test_features=test_data.review

	#pre-processing test features
	test_features=corpus_preprocessing(test_features)
	test_features=vectorizer.fit_transform(test_features)

	#predicting the sentiment for test set
	prediction=model.predict(test_features)
	
	#preparing submission file
	pd.DataFrame( data={"id":test_data["id"], "sentiment":prediction} ).to_csv("naive_Baiyes_bow.csv", index=False, quoting=3 )	

if __name__=="__main__":
	main()




