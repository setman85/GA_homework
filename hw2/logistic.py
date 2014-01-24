from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn import linear_model
import numpy as np

def load_titanic_data() :
    
    file = '/home/GA4/GA_homework/hw2/train.csv'
    data = open(file).read()
    return data

def organizeData(data) :
    data = data.split('\n')
    X = []
    y = []

    for d in data[1:-1]:
        tmp = []
        d = d.split(',')
        #model based on passenger class, age, gender

        #Check for valid class value, default to 2nd class)
        pclass = float(d[2])

        if (d[2] in (1.0,2.0,3.0)):
            tmp.append(pclass)
        else:
            tmp.append(float(2))

        #Check for valid age, default to 30 if no data
        try:
            age = float(d[6])

            if (d[6]>0):
                tmp.append(age)
            else:
                tmp.append(float(30))
        except:
            tmp.append(float(30))

        #For gender, assign 1 to male, 0 to female. Default is male
        gender = d[5]

        if (d[5] == 'male'):
            tmp.append(float(1))

        elif (d[5] == 'female'):
            tmp.append(float(0))

        else:
            tmp.append(float(1))

        X.append(tmp)
        y.append(float(d[1]))
    
    return np.array(X), np.array(y)

def stats(X, y):
    print y
    print "Number of passengers is %d" %len(y)  
    
    print "Number of survivors is %d" %y.sum()
    mean = np.mean(X, axis=0)
    print "Average age of passengers is %d" %mean[1]
    genders = X[:,2]
    print "Number men in dataset is %d" %genders.sum()

def knn(X_train, y_train, Cvalue, k_neighbors = 3 ) :
    # function returns a kNN object
    #  useful methods of this object for this exercise:
    #   fit(X_train, y_train) --&gt; fit the model using a training set
    #   predict(X_classify) --&gt; to predict a result using the trained model
    #   score(X_test, y_test) --&gt; to score the model using a test set

    clf = KNeighborsClassifier(k_neighbors)
    clf.fit(X_train, y_train)

    return clf


def nb(X_train, y_train, Cvalue) :
    # this function returns a Naive Bayes object
    #  useful methods of this object for this exercise:
    #   fit(X_train, y_train) --&gt; fit the model using a training set
    #   predict(X_classify) --&gt; to predict a result using the trained model
    #   score(X_test, y_test) --&gt; to score the model using a test set

    gnb = GaussianNB()
    clf = gnb.fit(X_train, y_train)

    return clf

def logistic_regression(X_train, y_train, Cvalue) :
    # this function fits a logistic regression model

    clf = linear_model.LogisticRegression(C=Cvalue)
    clf.fit(X_train, y_train)

    return clf

# generic cross validation function
def cross_validate(XX, yy, classifier, k_fold, cvalue) :

    # derive a set of (random) training and testing indices
    k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True, shuffle=True, random_state=0)

    k_score_total = 0
    # for each training and testing slices run the classifier, and score the results
    for train_slice, test_slice in k_fold_indices :

        model = classifier(XX[[ train_slice  ]],
                         yy[[ train_slice  ]], Cvalue=cvalue)

        k_score = model.score(XX[[ test_slice ]],
                              yy[[ test_slice ]])

        k_score_total += k_score

    # return the average accuracy
    return k_score_total/k_fold
