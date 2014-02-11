from randforest import *
import argparse
from numpy import arange

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--knn', help='Runs the knn classifier', action="store_true")
group.add_argument('--nb', help='Runs the naive bayes classifier', action="store_true")
group.add_argument('--logistic', help='Runs the logistic regression', action="store_true")
group.add_argument('--randforest', help='Runs the random forest', action="store_true")
args = parser.parse_args()

if args.knn:
	classfiers_to_cv=[("kNN",knn)]

elif args.nb:
	classfiers_to_cv=[("Naive Bayes",nb)]

elif args.logistic:
	classfiers_to_cv=[("Logistic",logistic_regression)]

elif args.randforest:
	classfiers_to_cv=[("Random Forest",Random_Forest)]
		
else:
	print "No argument passed at command line, run all methods"
	classfiers_to_cv=[("kNN",knn),("Naive Bayes",nb),("Logistic",logistic_regression),("Random Forest",Random_Forest)]

data = load_data()

print "Data loaded"

X, y = organizeData(data)

print "Data sorted into numpy arrays"

stats(X,y)

for (c_label, classifer) in classfiers_to_cv :

    print "---> %s <---" % c_label

    k_f = 3
    score, auc = cross_validate(X, y, classifer, c_label, k_fold=k_f, cvalue=1.0)
    print "Average score is %d" % score
    print "Average AUC is %d" % auc