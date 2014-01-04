
from sklearn import datasets

def load_iris_data() :

	iris = datasets.load_iris()

	return (iris.data, iris.target, iris.target_names)


from sklearn.neighbors import KNeighborsClassifier

def knn(X_train, y_train, k_neighbors = 3 ) :
	# metho returns a KNN object with methods
	#score(X_test, y_test) -- > to score the model using a test set
	#predict (X_classify, y_test) --> to predict a result using the trained model

	clf = KNeighborsClassifier(k_neighbors)
	clf.fit(X_train, y_train)

	return clf


from sklearn.cross_validation import KFold

def cross_validation(XX, yy, classifier, k_fold) :
	k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True, shuffle=True)

	k_score_total = 0
	for train_slice, test_slice in k_fold_indices :
		model=classifier(XX[[ train_slice  ]],
				yy[[ test_slice  ]])

		k_score = model.score(XX[[ test_slice  ]], 
					yy[[ test_slice ]])

		k_score_total += k_score

	return k_score_total*1.0/k_fold

from sklearn.naive_bayes import GaussianNB

def nb(X_train,y_train) : 

	gnb = GaussianNB()
	clf = gnb.fit(X_train, y_train)
	return clf


