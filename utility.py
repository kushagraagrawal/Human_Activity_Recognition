import numpy as np
from sklearn import neighbors, datasets
from collections import defaultdict
import scipy

def parseFile(filename):
	f = open(filename)
	featureArray = []
	lines = f.readlines()
	for line in lines:
		feature_length = len(line.split(" "))
		raw_feature = line.split(" ")
		feature = []
		for index in xrange(feature_length):
			try:
				feature.append(float(raw_feature[index]))
			except:
				continue
		featureArray.append(feature)
	return np.asarray(featureArray)
	
def checkAccuracy(original,predicted,labels):
	TP = defaultdict(list)
	FP = dafaultdict(list)
	FN = defaultdict(list)
	
	precision = []
	recall = []
	f_score = []
	
	for i in xrange(len(original)):
		if original[i] == predicted[i]:
			TP[str(int(original[i]))].append(1)
		elif original[i] != predicted[i]:
			FP[str(int(predicted[i]))].append(1)
			FN[str(int(original[i]))].append(1)
	
	for label in labels:
		p = float(len(TP[str(label)])) / (len(TP[str(label)]) + len(FP[str(label)]))
		precision.append(p)
		
		r = float(len(TP[str(label)])) / (len(TP[str(label)]) + len(FN[str(label)]))
		recall.append(r)
		
		fs = float(2*p*r)/ (p+r)
		f_score.append(fs)
	return precision,recall,f_score

def getDataSubset(inputData, inputLabels, RequiredLabels):
	subData = []
	subLabels = []
	for loopVar in range(len(inputLabels)):
		if inputLabels[loopVar] in RequiredLabels:
			subData.append(inputData[loopVar])
			subLabels.append(inputLabels[loopVar])
	return np.asarray(subData), np.asarray(subLabels)
