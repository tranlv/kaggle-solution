
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re 
import nltk.data
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from multiprocessing import cpu_count
import csv
import logging,gensim
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.models import mlp
 
tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')

def preprocessed_sentence(sentence,stop_words=False):
	#remove HTML/XML characters and punctuation
	soup=BeautifulSoup(sentence).get_text()
	letters_only = re.sub("[^a-zA-Z]", " ",soup ).split()
	if stop_words:
		stops = set(stopwords.words("english")) 
		letters_only = [w for w in letters_only if not w in stops] 
	return " ".join(letters_only)
		
	
# split paragraph into wordlist and preprocessed text	
def paragraph_into_wordlist_with_preprocessing(paragraph,tokenizer):
	preprocessed_wordlist = []
	paragraph=paragraph.decode("utf8")
	sentences=tokenizer.tokenize(paragraph.strip())		
	preprocessed_wordlist+=(w for w in preprocessed_sentence(sentence,stop_words=True).split() for sentence in sentences)				
	return preprocessed_wordlist
	
def create_bag_of_centroids( wordlist, word_centroid_map ):
    num_centroids = max( word_centroid_map.values() ) + 1
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids

	
if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

	print "define a class to iterate sentence :http://rare-technologies.com/word2vec-tutorial/#app "
	class MySentences(object):
		def __init__(self, dirname):
			self.dirname = dirname
			
		def __iter__(self):
			with open(".../data science projects/Bag_of_Words_Meets_Bags_of_Popcorn/labeledTrainData.tsv") as tsvfile:
				labeled_data = csv.reader(tsvfile, delimiter="\t")  
				for paragraph in labeled_data:
					paragraph=paragraph[2].decode("utf8")
					sentences=tokenizer.tokenize(paragraph.strip())	
					for sentence in sentences:
						yield preprocessed_sentence(sentence)	
					
			with open(".../labeledTrainData.tsv") as tsvfile:
				unlabeled_data = csv.reader(tsvfile, delimiter="\t")  
				for paragraph in unlabeled_data:
					paragraph=paragraph[2].decode("utf8")
					sentences=tokenizer.tokenize(paragraph.strip())	
					for sentence in sentences:
						yield preprocessed_sentence(sentence)			
				
	print "Training word2vec model..."
	sentences = MySentences('C:/Users/VU') 	
	model = gensim.models.Word2Vec(sentences, min_count=5, workers = cpu_count())
	model.save('my_model')	

	print " clustering words trained from word2vec model..."
	word_vectors = model.syn0
	num_clusters=word_vectors.shape[0]/5
	mean_clustering=KMeans(n_clusters=num_clusters)
	idx=kmean_clustering.fit_predict(word_vectors)

	print"assigning cluster number for each words... "
	word_centroid_map=dict(zip(model.index2word,idx))

	print "loading original labeled train dataset..."
	labeled_train_data=pd.read_csv(".../labeledTrainData.tsv", header=0,delimiter="\t", quoting=3)

	print "training centroids for each review from original labeled training set... "
	train_centroids = np.zeros(( labeled_train_data.review.size, num_clusters),dtype="float32" )
	for i in range(0,labeled_train_data.review.size-1) :	
		wordlist=paragraph_into_wordlist_with_preprocessing(labeled_train_data.review[i])
		train_centroids[i] = create_bag_of_centroids( wordlist, word_centroid_map )
		
	data_set=np.append(train_centroids,labeled_train_data.sentiment,axis=1)
	
	print "initiating neural network model..."
	hidden_layer = mlp.Sigmoid(layer_name='hidden', dim=np.mean([2,train_centroids.shape[1]])
	output_layer = mlp.Softmax(n_classes=2,layer_name= 'output')
	layers = [hidden_layer, output_layer]
	
	model = mlp.MLP(layers)
	trainer = sgd.SGD(termination_criterion=EpochCounter(400))
	trainer.setup(model, data_set)
	
	# train neural net until the termination criterion is true
	while True:
    trainer.train(dataset=data_set)
    model.monitor.report_epoch()
    model.monitor()
    if not trainer.continue_learning(model):
        break
		
	
	print "loading original test dataset..."
	test_data=pd.read_csv(".../testData.tsv", header=0,delimiter="\t", quoting=3)
	
	print "training centroids for each review from original test set... "
	test_centroids = np.zeros(( test_Data["review"].size, num_clusters), dtype="float32" )
	for i in range(0,test_data.review.size-1):
		wordlist=paragraph_into_wordlist_with_preprocessing(test_data.review[i],tokenizer)
		test_centroids[i] = create_bag_of_centroids( wordlist, word_centroid_map )
		
	print "predicting original test set..."
	prediction = model.fprop(theano.shared(inputs, name='inputs'))
	
	print "writing out submission file..."
	pd.DataFrame( data={"id":test_data["id"], "sentiment":prediction} ).to_csv("naive_Baiyes_Word2vec.csv", index=False, quoting=3 )	

