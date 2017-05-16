from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import utility

print "Parsing"

X_train = utility.parseFile('X_train.txt')
Y_train = utility.parseFile('y_train.txt')
Y_train = Y_train.flatten()
X_train,Y_train = utility.getDataSubset(X_train,Y_train,[4,5,6])

X_test = utility.parseFile('X_test.txt')
Y_test = utility.parseFile('y_test.txt')
Y_test = Y_test.flatten()
X_test, Y_test = utility.getDataSubset(X_test,Y_test,[4,5,6])

print "Done"

clf = RandomForestClassifier(n_estimators = 50)
clf = clf.fit(X_train, Y_train)

print "Predicting"

predicted = []

for x_test in X_test:
	predicted.append(clf.predict(x_test)[0])

print "Done, now check accuracy"

precision, recall, f_score = utility.checkAccuracy(predicted, Y_test, [4,5,6])

print f_score
