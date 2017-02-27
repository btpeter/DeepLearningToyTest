import csv
import itertools
import numpy as np
import nltk
import time
import sys
import operator
import io
import array

SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"

FILENAME = "data/reddit-comments-2015-08.csv"
VOCABULARY_SIZE = 8000
MIN_SENT_CHARACTERS = 0

def load_data(filename=FILENAME, vocabulary_size = VOCABULARY_SIZE, min_sent_characters = MIN_SENT_CHARACTERS):
	word_to_index = []
	index_to_word = []

	print("Reading CSV file...")

	with open(filename, 'rb') as f:
		reader = csv.reader(f, skipinitialspace=True)
		reader.next()
		sentences=itertools.chain(*[nltk.sent_tokenize(x[0].decode("utf-8").lower()) for x in reader])

		# filtter
		sentences=[s  for s in sentences if len(x)>min_sent_characters]
		sentences=[s for s in sentences if "http" not in s]

		# add tags
		sentences = ["%s %s %s"% (SENTENCE_START_TOKEN, s, SENTENCE_END_TOKEN) for s in sentences]

	print ("All sentences count : %d" %(len(sentences)))

	# tokenize
	tokenized_sentences=[ nltk.word_tokenize(s) for s in sentences]

	# statistic word frequence
	word_freq=nltk.FreqDist(nltk.chain(*tokenized_sentences))

	
	# get the most common words and build index_to_word and word_to_index
	vocab = word_freq.most_common(vocabulary_size-1)
	index_to_word = [x[0] for x in vocab]
	index_to_word.append(UNKNOWN_TOKEN)
	# index for each word
	word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

	for i, sent in enumerate(tokenized_sentences):
		tokenized_sentences[i] = [w if w in index_to_word else UNKNOWN_TOKEN for w in sent]

	# x_data and labels
	x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
	y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

	return x_train, y_train, word_to_index, index_to_word


def softmax(x):
	xt = np.exp(x - np.max(x))
	return xt / np.sum(xt)


def save_model_parameters(outfile, model):
	U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
	np.savez(outfile, U=U, V=V, W=W)
	print "Save model parameters to %s. " % outfile

def load_model_parameters(path, model):
	npzfile = np.load(path)
	U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
	model.hidden_dim = U.shape[0]
	model.word_dim = U.shape[1]
	model.U.set_value(U)
	model.V.set_value(V)
	model.W.set_value(W)
	print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])

