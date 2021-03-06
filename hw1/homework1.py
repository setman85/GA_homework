OBimport argparse
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold

parser = argparse.ArgumentParser()
parser.add_argument('--knn', help='Runs the knn classifier', action="store_true")
parser.add_argument('--gaussian', help='Runs the gaussian classifier', action="store_true")
args = parser.parse_args()

def load_iris_data() :
	iris = datasets.load_iris()
	return (iris.data, iris.target, iris.target_names)

def knn(X_train, y_train, k_neighbors = 3) :
	clf = KNeighborsClassifier(k_neighbors)
	clf.fit(X_train, y_train)

	return clf

def nb(X_train, y_train) :
	gnb = GaussianNB()
	clf = gnb.fit(X_train, y_train)

	return clf

def cross_validate(XX, yy, classifier, k_fold) :
	#define random testing and training indices
	k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True, shuffle=True, random_state=0)
	k_score_total = 0

	for train_slice, test_slice in k_fold_indices :
		
		model = classifier(XX[[ train_slice  ]],
				yy[[  train_slice  ]])

		k_score = model.score(XX[[  test_slice  ]],
				yy[[  test_slice  ]])

		
		k_score_total += k_score

	return k_score_total/k_fold

(XX, yy, y) = load_iris_data()

if args.knn:
	print "Due to command line argument, running the knn classifier"
	classifiers_to_cv=[("knn",knn)]

	if args.gaussian:
		print "Due to command line argument, running Gaussian classifier also"
		classifiers_to_cv=[("knn",knn),("Naive Bayes",nb)]

elif args.gaussian:
	print "Due to command line argument, running Gaussian classifier"
	classifiers_to_cv=[("Naive Bayes",nb)]

else:
	print "No command line argument specified, using both classifiers"
	classifiers_to_cv=[("knn",knn),("Naive Bayes",nb)]

for (c_label, classifier) in classifiers_to_cv :
	print "---->  %s  <---- " %c_label
	best_k=0
	best_cv_a=0
	for k_f in [2,3,5,10,15,30,75,150] :
		cv_a = cross_validate(XX, yy, classifier, k_fold=k_f)
		if cv_a > best_cv_a :
			best_cv_a=cv_a
			best_k=k_f

		print "fold <<%s>>  : : acc <<%s>>" %(k_f, cv_a)

	print "Highest accuracy: fold <<%s>>  : :  <<%s>>>" %(best_k, best_cv_a)

