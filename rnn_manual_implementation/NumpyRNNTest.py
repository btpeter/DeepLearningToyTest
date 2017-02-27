import numpy as np 
from util import *
import operator

class RNNNumpy:

	def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate

		self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
		self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
		self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))


	def forward_propagation(self, x):
		
		# The total number of time steps
		T = len(x)

		s = np.zeros((T+1, self.hidden_dim))
		s[-1] = np.zeros(self.hidden_dim)

		o = np.zeros((T, self.word_dim))

		# for each time step
		for t in np.arange(T):
			s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
			o[t] = softmax(self.V.dot(s[t]))

		return [o, s]

	def predict(self, x):
		o, s = self.forward_propagation(x)
		return np.argmax(o, axis=1)

	def calculate_total_loss(self, x, y):
		L = 0
		# for each sentence
		for i in np.arange(len(y)):
			o, s = self.forward_propagation(x[i])
			# we only care about our prediction of the "correct" words
			correct_word_predictions = o[np.arange(len(y[i])), y[i]]
			# add to the loss based on how off we were
			L += -1 * np.sum(np.log(correct_word_predictions))
		return L

	def calculate_loss(self, x, y):
		# divide the total loss by the number of training examples
		N = np.sum((len(y_i) for y_i in y))
		return self.calculate_total_loss(x, y)/N


	def bptt(self, x, y):
		T = len(y)
		o, s = self.forward_propagation(x)
		dLdU = np.zeros(self.U.shape)
		dLdV = np.zeros(self.V.shape)
		dLdW = np.zeros(self.W.shape)
		delta_o = o
		delta_o[np.arange(len(y)), y] -= 1
		for t in np.arange(T)[::-1]:
			
			dLdV += np.outer(delta_o[t], s[t].T)
			delta_t = self.V.T.dot(delta_o[t]) * (1-(s[t] ** 2))

			# Backpropagation throug time
			for bptt_step in np.arange(max(0, t-self.bptt_truncate))[::-1]:

				dLdW += np.outer(delta_t, s[bptt_step-1])
				dLdU[:,x[bptt_step]] += delta_t

				# update delta for next step
				delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step] ** 2)
		return [dLdU, dLdV, dLdW]



	def sgd_step(self, x, y, learning_rate):
		# Calculate the gradients
		dLdU, dLdV, dLdW = self.bptt(x, y);
		# Update parameters
		self.U -= learning_rate * dLdU
		self.V -= learning_rate * dLdV
		self.W -= learning_rate * dLdW

	def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
		bptt_gradients = self.bptt(x, y)

		model_parameters = ['U', 'V', 'W']

		# Gradient check for each parameter
		for pidx, pname in enumerate(model_parameters):
			# Get the actual parameter value from the model
			parameter = operator.attrgetter(pname)(self)
			print "Performing gradient check for parameter %s with size $d" %(pname, np.prod(parameter.shape))
			it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
			while not it.finished:
				ix = it.multi_index
				# Save the original value
				original_value = parameter[ix]
				# estimate the gradient using (f(x+h)-f(x-h))/(2*h)
				parameter[ix] = original_value + h
				gradplus = self.calculate_total_loss([x],[y])
				parameter[ix] = original_value - h
				gradminus = self.calculate_total_loss([x],[y])
				estimated_gradient = (gradplus - gradminus) / (2*h)
				# Reset parameter to original value
				parameter[ix] = original_value

				# gradient calculated using bptt
				backprop_gradient = bptt_gradients[pidx][ix]

				# calculate the relative error by : (|x-y| / (|x| + |y|))
				relative_error = np.abs(backprop_gradient - estimated_gradient) / (np.abs(backprop_gradient) + np.abs(estimated_gradient))

				# if error is too large, fail
				if(relative_error > error_threshold):
					print "Gradient check ERROR : parameter=%s ix=%s" %(pname, ix)
					print "+h Loss : %f" %(gradplus)
					print "-h Loss : %f" %(gradminus)
					print "Estimated_gradient : %f" %(estimated_gradient)
					print "Backpropagation gradient : %f" %(backprop_gradient)
					print "Relative Error : %f" %(relative_error)
				it.iternext()
			print "Gradient check for parameter %s passed." %(pname)







