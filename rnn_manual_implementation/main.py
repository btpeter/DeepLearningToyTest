from util import *
from NumpyRNNTest import *
from TheanoRNNTest import *
import numpy as np
from datetime import datetime
import sys

VOCABULARY_SIZE = 8000
HIDDEN_DIM = 80



def train_with_sdg(model, x_train, y_train, learning_rate = 0.005, nepoch=10, evaluate_loss_after=1):
	losses = []
	num_examples_seen = 0
	for epoch in range(nepoch):

		if(epoch % evaluate_loss_after == 0):
			loss = model.calculate_loss(x_train, y_train)
			losses.append((num_examples_seen, loss))
			time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
			print "%s: Loss after num_examples_seen=%d epoch=%d: %f" %(time, num_examples_seen, epoch, loss)

			# Adjust the learning rate if loss increase
			if(len(losses) > 1 and losses[-1][1] > losses[-1][1]):
				learning_rate = learning_rate * 0.5
				print "Settring learning rate to %f" %(learning_rate)
			sys.stdout.flush()

		# for each training example
		for i in range(len(y_train)):
			# one SGD step
			model.sgd_step(x_train[i], y_train[i], learning_rate)
			num_examples_seen += 1





if __name__ == '__main__':
	np.random.seed(10)
	model = RNNNumpy(VOCABULARY_SIZE, HIDDEN_DIM)
	#model = RNNTheano(VOCABULARY_SIZE, HIDDEN_DIM)
	trX, trY, word_to_idx, idx_to_word = load_data()
	train_with_sdg(model, trX[0:100], trY[0:100])
	#o, s = model.forward_propagation(trX[10])
	#print o.shape
	#print s.shape
	#print len(o)
	#print o	
	#predictions = model.predict(trX[10])
	#print predictions.shape
	#print predictions

	#print "Expected Loss for random predictions: %f" %np.log(VOCABULARY_SIZE)
	#print "Actual loss: %f" %model.calculate_loss(trX[:1000], trY[:1000])