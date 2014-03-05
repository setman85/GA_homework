from logistic import *
import argparse
from numpy import arange

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--knn', help='Runs the knn classifier', action="store_true")
group.add_argument('--nb', help='Runs the naive bayes classifier', action="store_true")
group.add_argument('--logistic', help='Runs the logistic regression', action="store_true")
args = parser.parse_args()

flag =0

if args.knn:
	classfiers_to_cv=[("kNN",knn)]

elif args.nb:
	classfiers_to_cv=[("Naive Bayes",nb)]

elif args.logistic:
	classfiers_to_cv=[("Logistic",logistic_regression)]
	flag = 1
else:
	print "No argument passed at command line, run all methods"
	classfiers_to_cv=[("kNN",knn),("Naive Bayes",nb),("Logistic",logistic_regression)]
	flag = 1
data = load_titanic_data()

print "Titanic Data loaded"

X, y = organizeData(data)

print "Data sorted into numpy arrays"

stats(X,y)


for (c_label, classifer) in classfiers_to_cv :

    print
    print "---> %s <---" % c_label

    best_k=0
    best_cv_a=0
    for k_f in [2,3,5,10,15,30,50,75,150] :
       cv_a = cross_validate(X, y, classifer, k_fold=k_f, cvalue=1.0)
       if cv_a >  best_cv_a :
            best_cv_a=cv_a
            best_k=k_f

       print "fold <<%s>> :: acc <<%s>>" % (k_f, cv_a)


    print "Highest Accuracy: fold <<%s>> :: <<%s>>" % (best_k, best_cv_a)

#find optimal c value for logistic classifier, using the optimal fold number identified earlier
if flag == 1:
	print "Now optimizing C value for logistic regression method, using optimal number of folds"
	best_C=0
	best_cv_a=0

	for c in arange(0.1,1.0,0.1):
		cv_a = cross_validate(X, y, logistic_regression, k_fold=best_k, cvalue=c)
		if cv_a >  best_cv_a :
			best_cv_a=cv_a
			best_C=c

		print "C <<%s>> :: acc <<%s>>" % (c, cv_a)
	
	print "Highest Accuracy: C <<%s>> :: <<%s>>" % (best_C, best_cv_a)
