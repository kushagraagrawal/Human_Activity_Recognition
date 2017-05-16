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
	#print precision
	#print recall
	print fscore

def MLP_onNonDynamicData():
	XFull = utility.parseFile('X_train.txt')
	YFull = utility.parseFile('y_train.txt')
	
	XFullTest = utility.parseFile('X_test.txt')
	YFullTest = utility.parseFile('y_test.txt')
	
	X_NonDynamic,Y_NonDynamic = utility.getDataSubset(XFull,YFull.flatten(),[4,5,6])
	X_NonDynamicTest,Y_NonDynamicTest = utility.getDataSubset(XFullTest,YFullTest.flatten(),[4,5,6])
	
	clf = MLPClassifier()
	clf.fit(X_NonDynamic, Y_NonDynamic.flatten())
	
	precision, recall, fscore = utility.checkAccuracy(clf.predict(X_NonDynamicTest),Y_NonDynamicTest,[4,5,6])
	utility.createConfusionMatrix(clf.predict(X_NonDynamicTest).flatten(),Y_NonDynamicTest.flatten(),[4,5,6])
	print fscore
	
	X_Dynamic, Y_Dynamic = utility.getDataSubset(XFull,YFull.flatten(),[1,2,3])
	X_DynamicTest, Y_DynamicTest = utility.getDataSubset(XFullTest,YFullTest.flatten(),[1,2,3])
	print len(X_DynamicTest),len(Y_DynamicTest)
	
	clf = MLPClassifier()
	clf.fit(X_Dynamic, Y_Dynamic.flatten())
	
	precision, recall, fscore = utility.checkAccuracy(clf.predict(X_DynamicTest),Y_DynamicTest,[1,2,3])
	utility.createConfusionMatrix(clf.predict(X_DynamicTest).flatten(),Y_DynamicTest.flatten(),[1,2,3])
	
	print fscore
	
	
	
	
if __name__ == '__main__':
	MLP_onFullDataset()
	MLP_onNonDynamicData()
