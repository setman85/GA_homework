from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn import linear_model
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib
matplotlib.use("AGG")
from matplotlib import pyplot as plt

def load_data() :
    print "Starting to analyze data" 
    file = 'cmc.data'
    data = open(file).read()
    return data

def organizeData(data) :
    data = data.split('\n')
    X = []
    y = []

    for d in data:
        tmp = []
        d = d.split(',')
        #model based on 9 features, which are the first 9 elements of the array
        for i in range(0,9):
			try:
				tmp.append(float(d[i]))
			except:
				tmp.append(float(0))
        X.append(tmp)
		#The 10th element in the array is the one we are classifying on
        try:
			y.append(float(d[9]))
        except:
		    pass
    return np.array(X), np.array(y)

def stats(X, y):
    print "We are analyzing contraceptive use among women based on a variety of factors"
    print "Number of women in data is %d" %len(y)  
    print "Number of women with no contraceptive use is %d" %(y==1).sum()
    print "Number of women with long term use is %d" %(y==2).sum()
    print "Number of women with short term use is %d" %(y==3).sum()
    mean = np.mean(X, axis=0)
    print "Average age of women is %d" %mean[0]
    print "*****"
    print "Preparing histograms to visualize data"
    num_bins = 20
	
    class_one = y==1.0
    features_one = X[class_one,:]
    class_two = y==2
    class_three = y==3
    features_two = X[class_two,:]
    features_three = X[class_three,:]
	
    plt.figure(1)
    plt.autoscale(enable=True,axis='both',tight=None)
    plt.hist(features_one[:,0], 
        bins=num_bins,
        #range=(1,3),
        facecolor='red',
        stacked='false',
        label='Class 1 (No-Use)',
        alpha=.5)
    plt.hist(features_two[:,0], 
        bins=num_bins,
        #range=(1,3),
        facecolor='blue',
        stacked='false',
        label='Class 2 (Long-Term)',
        alpha=.5)
    plt.hist(features_three[:,0], 
        bins=num_bins,
        #range=(1,3),
        facecolor='green',
        stacked='false',
        label='Class 3 (Short-term)',
        alpha=.5)
    plt.xlabel('Age')
    plt.ylabel('Number of Women')
    plt.legend(loc="upper right")
    plt.savefig('Histogram.png')
    print "Histogram completed. Copy to local machine to view"
	
def knn(X_train, y_train, k_neighbors = 3, *args) :
    clf = KNeighborsClassifier(k_neighbors)
    clf.fit(X_train, y_train)

    return clf

def nb(X_train, y_train, *args) :
    gnb = GaussianNB()
    clf = gnb.fit(X_train, y_train)

    return clf

def logistic_regression(X_train, y_train, C) :
    # this function fits a logistic regression model
    clf = linear_model.LogisticRegression(C=C)
    clf.fit(X_train, y_train)

    return clf

def Random_Forest(x_train, Y_train,n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, 
                  random_state=None, verbose=0, min_density=None, compute_importances=None, *args):
    clf = RandomForestClassifier()
    clf.fit(x_train,Y_train)
    
    return clf
	
# generic cross validation function
def cross_validate(XX, yy, classifier, c_label, k_fold, cvalue=1.0) :
    # derive a set of (random) training and testing indices
    k_fold_indices = KFold((len(XX)-1), n_folds=k_fold, indices=True, shuffle=True, random_state=0)
    k_score_total = 0
    auc_total = 0
    flag = 1
    # for each training and testing slices run the classifier, score the results, and calculate AUC
    for train_slice, test_slice in k_fold_indices :
        model = classifier(XX[train_slice], yy[train_slice], cvalue)

        k_score = model.score(XX[test_slice], yy[test_slice])

        k_score_total += k_score
       
       
       # Can't calculate the ROC because data is not binary, but can use this code for the next time
       # test_predict = model.predict_proba(XX[test_slice])
       # fpr, tpr, threshold = roc_curve(yy[test_slice],test_predict[:,1],pos_label=None)
       # auc_total += auc(fpr,tpr)
		
	# for one of the folds in each classifier, plot the ROC. 
    #However, can't do this because the data is not binary. Will use this for the next time. 
        if (flag == 0):
            pl.clf()
            pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            pl.plot([0, 1], [0, 1], 'k--')
            pl.xlim([0.0, 1.0])
            pl.ylim([0.0, 1.0])
            pl.xlabel('False Positive Rate')
            pl.ylabel('True Positive Rate')
            pl.title('Receiver operating characteristic example')
            pl.legend(loc="lower right")
            plt.savefig('ROC %s.png' %c_label)
            flag = 1
    # return the average accuracy, AUC
    return k_score_total/k_fold, auc_total/k_fold
