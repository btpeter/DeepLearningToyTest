import numpy as np
import theano as theano
import theano.tensor as T
from util import *
import operator


class RNNTheano:

	def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):

		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate

		U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
		V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
		W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

		# Created Theano shared variables
		self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
		self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
		self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))

		# Store Theano graph

		self.theano = {}
		self.__theao_build__()

	def __theao_build__(self):
		U, V, W = self.U, self.V, self.W
		x = T.ivector('x')
		y = T.ivector('y')
		def forward_prop_step(x_t, s_t_prev, U, V, W):
			s_t = T.tanh(U[:,x_t]) + W.dot(s_t_prev)
			o_t = T.nnet.softmax(V.dot(s_t))
			# [ATTANTION]!! since Theano always returns a matrix, not a vector, 
			# 				so we just converting the matrix with one row it into 
			#				a VECTOR (which has length word_dim) 
			return [o_t[0], s_t]
		[o, s], updates = theano.scan(
			forward_prop_step,
			sequences=x,
			outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
			non_sequences=[U, V, W],
			truncate_gradient=self.bptt_truncate,
			strict=True)
		prediction = T.argmax(o, axis=1)
		o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

		# gradient 
		dU = T.grad(o_error, U)
		dV = T.grad(o_error, V)
		dW = T.grad(o_error, W)


		# Assohm funcations
		self.forward_propagation = theano.function([x], o)
		self.predict = theano.function([x], prediction)
		self.ce_error = theano.function([x, y], o_error)
		self.bptt = theano.function([x, y], [dU, dV, dW])

		# SGD
		learning_rate = T.scalar('learning_rate')
		self.sgd_step = theano.function([x, y, learning_rate], [], 
						updates=[(self.U, self.U - learning_rate * dU),
								(self.V, self.V - learning_rate * dV),
								(self.W, self.W - learning_rate * dW)])

	def calculate_total_loss(self, X, Y):
		return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])

	def calculate_loss(self, X, Y):
		num_words = np.sum([len(y) for y in Y])
		return self.calculate_total_loss(X, Y) / float(num_words)
