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
            raise ValueError('Softmax output is only supported '+
                'with cross entropy loss function.')
        if output_layer!='softmax' and loss=='cross_entropy':
            raise ValueError('Cross-entropy loss is only ' +
				'supported with softmax output layer.')
		
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
		
		
		self
