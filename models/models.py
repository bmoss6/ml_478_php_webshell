import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load

from sklearn import neighbors
import unittest

def test_svm():
    df = pd.read_csv('../datasets/backdoor_webshells_features.csv')
    X, y = np.split(df, [9], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    all_names, X = np.split(X, [1], axis=1)
    svm_clf = SVC(gamma='scale')
    scores = cross_val_score(svm_clf, X, y, cv=10)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))

def test_nb():
    df = pd.read_csv('../datasets/backdoor_webshells_features.csv')
    X, y = np.split(df, [9], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    all_names, X = np.split(X, [1], axis=1)
    clf = MultinomialNB()
    scores = cross_val_score(clf, X, y, cv=10)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))

def test_gradient_boosting():
    df = pd.read_csv('../datasets/backdoor_webshells_features.csv')
    X, y = np.split(df, [9], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    all_names, X = np.split(X, [1], axis=1)

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X, y, cv=10)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))
    dump(clf, '../saved_models/gradient_boosting.joblib')

def test_bagging():
    df = pd.read_csv('../datasets/backdoor_webshells_features.csv')
    X, y = np.split(df, [9], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    all_names, X = np.split(X, [1], axis=1)
    clf = BaggingClassifier(base_estimator=SVC(gamma='scale'), n_estimators=10)
    scores = cross_val_score(clf, X, y, cv=10)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))
def test_knn():
    df = pd.read_csv('../datasets/backdoor_webshells_features.csv')
    X, y = np.split(df, [9], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    all_names, X = np.split(X, [1], axis=1)
    clf = neighbors.KNeighborsClassifier(3, weights='distance')
    clf.fit(X, y)
    scores = cross_val_score(clf, X, y, cv=10)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))

def test_random_forest():
    df = pd.read_csv('../datasets/backdoor_webshells_features.csv')
    X,y = np.split(df, [9], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    clf = RandomForestClassifier(n_estimators=20)
    all_names, X = np.split(X, [1], axis=1)
    scores = cross_val_score(clf, X,y, cv=10)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores)/len(scores)))
    clf.fit(X_train,y_train)
    print("\n Feature Importance")
    for f in range(len(list(X))):
        print("{}:{}".format(list(X)[f], clf.feature_importances_[f]))

    prob = clf.predict_proba(X_test)
    for x in range(len(X_test)):
        print("{}:{}".format(test_names.values[x], prob[x][1]))

def test_random_forest_full():
    df = pd.read_csv('../datasets/full_features.csv')
    X,y = np.split(df, [19], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    clf = RandomForestClassifier(n_estimators=20)
    all_names, X = np.split(X, [1], axis=1)
    scores = cross_val_score(clf, X,y, cv=20)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores)/len(scores)))
    dump(clf, '../saved_models/random_forest_v2.joblib')
    clf.fit(X_train,y_train)
    print("\n Feature Importance")
    for f in range(len(list(X))):
        print("{}:{}".format(list(X)[f], clf.feature_importances_[f]))
    dump(clf, '../saved_models/random_forest_v2.joblib')
   # prob = clf.predict_proba(X_test)

if __name__== "__main__":
    print("SVM")
    test_svm()
    print("------------------------------------")
    print("RANDOM FOREST")
    test_random_forest_full()
    print("-----------------------------------")
    print("3 NEAREST")
    test_knn()
    print("-----------------------------------")
    print("Multinomial NB")
    test_nb()
    print("------------------------")
    print("Bagging SVM")
    test_bagging()
    print("------------------------------")
    print ("Gradient Boosting")
    test_gradient_boosting()
    print("-----------------------------")
