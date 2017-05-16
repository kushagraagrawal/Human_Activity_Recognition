import numpy as np
import warnings

from itertools import cycle,izip

from sklearn.utils import gen_even_slices
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer

'''
	softmax, tanh, dtanh
'''

def softmax(x):
	np.exp(x,x)
	x /= np.sum(x,axis=1)[:, np.newaxis]

def tanh(x):
	np.tanh(x,x)

def dtanh(x):
	'''derivative of tanh as a function of tanh '''
	x *= -x
	x+=1
	
class BaseMLP(BaseEstimator):
	def __init__(self,n_hidden,lr,l2decay,loss,output_layer,batch_size,verbose=0):
		self.n_hidden = n_hidden
		self.lr = lr
		self.l2decay = l2decay
		self.loss = loss
		self.batch_size = batch_size
		self.verbose = verbose
		
		if output_layer=='softmax' and loss!='cross_entropy':
            raise ValueError('Softmax output is only supported with cross entropy loss function.')
        if output_layer!='softmax' and loss=='cross_entropy':
            raise ValueError('Cross-entropy loss is only supported with softmax output layer.')
		
		if output_layer =='linear':
			self.output_func = id
		elif output_layer == 'softmax':
			self.output_func = softmax
		elif output_layer == 'tanh':
			self.output_func = tanh
		else:
			raise ValueError("output layer don't satisfy")
		
		if not loss in ['cross_entropy','square','crammer_singer']:
			raise ValueError("loss don't match")
			self.loss = loss
	
	def fit(self,X,y,max_epochs,shuffle_data,verbose = 0):
		n_samples, n_features = X.shape
		if y.shape[0] != n_samples:
			raise ValueError("shapes of x and y don't fit")
		self.n_outs = y.shape[1]
		n_batches = n_samples/self.batch_size
		if n_samples % self.batch_size != 0:
			warnings.warn("discarding samples")
		n_iterations = int(max_epochs * n_batches)
		
		if shuffle_data:
			X, y = shuffle(X,y)
		batch_slices = list(gen_even_slices(n_batches * self.batch_size, n_batches))
		
		#weights and bias
		self.weights1 = np.random.uniform(size = (n_features,self.n_hidden))/np.sqrt(n_features)
		self.bias1 = np.zeros(self.n_hidden)
		self.weights2 = np.random.uniform(size = (self.n_hidden,self.n_outs))/np.sqrt(self.n_hidden)
		self.bias2 = np.zeros(self.n_outs)
		# empty datasets for hidden and output layer
		x_hidden = np.empty((self.batch_size, self.n_hidden))
		delta_h = np.empty((self.batch_size, self.n_hidden))
		x_output = np.empty((self.batch_size, self.n_outs))
		delta_o = np.empty((self.batch_size, self.n_outs))
		
		#forward and backward propagation
		for i,batch_slice in izip(xrange(self.n_iterations),cycle(batch_slices)):
			self.forward(i,X,batch_slice,x_hidden,x_output)
			self.backward(i,X,y,batch_slice,x_hidden,x_output,delta_o,delta_h)
		return self
		
	#forward propagation
	def forward(i,X,batch_slice,x_hidden,x_output):
		x_hidden[:]= np.dot(X[batch_slice],self.weights1)
		x_hidden += self.bias1
		np.tanh(x_hidden,x_hidden)
		x_output[:] = np.dot(x_hidden,self.weights2)
		x_output += self.bias2
		
		self.output_func(x_output)
	
	def predict(self,X):
		n_samples = X.shape[0]
		x_hidden = np.empty((n_samples, self.n_hidden))
		x_output = np.empty((n_samples, self.n_outs))
		self.forward(None, X,slice(0,n_samples),x_hidden,x_output)
		return x_output
	#backward propagation
	def backward(i,X,y,batch_slice,x_hidden,x_output,delta_o,delta_h):
		
		if self.loss in ['cross_entropy'] or (self.loss == 'square' and self.output_func == id):
			delta_o[:] = y[batch_slice] - x_output
		elif self.loss == 'crammer_singer':
			raise ValueError("not implemented yet")
			delta_o[:] = 0
			delta_o[y[batch_slice],np.ogrid[len(batch_slice)]] -=1
			delta_o[np.argmax(x_output - np.ones((1))[y[batch_slice], np.opgrid[len(batch_slice)]],axis=1),np.ogrid[len(batch_slice)]] += 1
 		elif self.loss == 'square' and self.output_func == _tanh:
            delta_o[:] = (y[batch_slice] - x_output) * _dtanh(x_output)
        else:
            raise ValueError("Unknown combination of output function and error.")

		if self.verbose > 0:
			print(np.linalg.norm(delta_o / self.batch_size))
		delta_h[:] = np.dot(delta_o, self.weights2_.T)

        # update weights
		self.weights2_ += self.lr / self.batch_size * np.dot(x_hidden.T, delta_o)
		self.bias2_ += self.lr * np.mean(delta_o, axis=0)
		self.weights1_ += self.lr / self.batch_size * np.dot(X[batch_slice].T, delta_h)
		self.bias1_ += self.lr * np.mean(delta_h, axis=0)

class MLPClassifier(BaseMLP, ClassifierMixin):
	def __init__(self,n_hidden=200,lr=0.1,l2decay = 0, loss = 'cross_entropy',output_layer = 'softmax', batch_size = 100,verbose = 0):
		super(MLPClassifier,self).__init__(n_hidden,lr,l2decay,loss,output_layer,batch_size,verbose)
	
	def fit(self,X,y,max_epochs = 10, shuffle_data = False):
		self.lb = LabelBinarizer()
		one_hot_labels = self.lb.fit_transform(y)
		super(MLPClassifier,self).fit(X,one_hot_labels,max_epochs,shuffle_data)
		return self
	def predict(self,X):
		prediction = super(MLPClassifier,self).predict(X)
		return self.lb.inverse_transform(prediction)

def test_classification():
	from sklearn.datasets import load_digits
	digits = load_digits()
	X,y = digits.data, digits.target
	mlp = MLPClassifier()
	mlp.fit(X,y)
	training_score = mlp.score(X,y)
	print training_score

if __name__ == "__main__":
	test_classification()
