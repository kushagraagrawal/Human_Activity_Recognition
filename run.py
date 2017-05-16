import numpy as np
from MLP import *
import utility

def MLP_onFullDataset():
	XFull = utility.parseFile('X_train.txt')
	YFull = utility.parseFile('y_train.txt')
	
	XFullTest = utility.parseFile('X_test.txt')
	YFullTest = utility.parseFile('y_test.txt')
	
	clf = MLPClassifier()
	clf.fit(XFull,YFull.flatten())
	
	precision,recall,fscore = utility.checkAccuracy(clf.predict(XFullTest),YFullTest,[1,2,3,4,5,6])
	print precision
	print recall
	print fscore
	
if __name__ == '__main__':
	MLP_onFullDataset()
