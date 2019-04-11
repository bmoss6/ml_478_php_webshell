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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, recall_score, precision_score
from joblib import dump, load
import unittest

from sklearn import neighbors
import unittest


def test_svm():
    df = pd.read_csv('../datasets/full_features.csv')
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

    svm_clf.fit(X_train, y_train)
    y_score = svm_clf.predict(X_test)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


def test_nb():
    df = pd.read_csv('../datasets/full_features.csv')
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

    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


def test_gradient_boosting_initial():
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
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


def test_gradient_boosting_full():
    df = pd.read_csv('../datasets/full_features.csv')
    X, y = np.split(df, [19], axis=1)
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
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


def test_gradient_boosting_pca():
    df = pd.read_csv('../datasets/pca.csv')
    X, y = np.split(df, [4], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    # test_names, X_test = np.split(X_test, [1], axis=1)
    # train_names, X_train = np.split(X_train, [1], axis=1)
    # all_names, X = np.split(X, [1], axis=1)
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X, y, cv=10)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


def test_random_forest_initial():
    df = pd.read_csv('../datasets/backdoor_webshells_features.csv')
    X, y = np.split(df, [9], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    clf = RandomForestClassifier(n_estimators=20)
    all_names, X = np.split(X, [1], axis=1)
    scores = cross_val_score(clf, X, y, cv=20)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))
    dump(clf, '../saved_models/random_forest_v2.joblib')
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)

    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


#   print("\n Feature Importance")
#   for f in range(len(list(X))):
#       print("{}:{}".format(list(X)[f], clf.feature_importances_[f]))


def test_random_forest_full():
    df = pd.read_csv('../datasets/full_features.csv')
    X, y = np.split(df, [19], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    clf = RandomForestClassifier(n_estimators=20)
    all_names, X = np.split(X, [1], axis=1)
    scores = cross_val_score(clf, X, y, cv=20)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))
    dump(clf, '../saved_models/random_forest_v2.joblib')
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


#  print("\n Feature Importance")
#  for f in range(len(list(X))):
#      print("{}:{}".format(list(X)[f], clf.feature_importances_[f]))


def test_random_forest_pca():
    df = pd.read_csv('../datasets/pca.csv')
    X, y = np.split(df, [4], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    clf = RandomForestClassifier(n_estimators=5)
    all_names, X = np.split(X, [1], axis=1)
    scores = cross_val_score(clf, X, y, cv=20)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))
    dump(clf, '../saved_models/random_forest_v2.joblib')
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


#   print("\n Feature Importance")
#   for f in range(len(list(X))):
#       print("{}:{}".format(list(X)[f], clf.feature_importances_[f]))
#   dump(clf, '../saved_models/random_forest_v2.joblib')
# prob = clf.predict_proba(X_test)
# for x in range(len(X_test)):
#     print("{}:{}".format(test_names.values[x], prob[x][1]))


def test_gradient_boosting_initial_improv():
    parameters = {'min_samples_split': [2, 4, 8, 10, 20, 50, 100], 'n_estimators': [20, 50, 100],
                  'max_features': [4, 8], 'max_depth': [3, 4, 5]}
    df = pd.read_csv('../datasets/backdoor_webshells_features.csv')
    X, y = np.split(df, [9], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    all_names, X = np.split(X, [1], axis=1)
    clf = GradientBoostingClassifier(max_depth=5, max_features=4, min_samples_split=10, n_estimators=50)
    # clf = GridSearchCV(clf,parameters, cv=10)
    # clf.fit(X,y)
    # print(clf.best_params_)

    scores = cross_val_score(clf, X, y, cv=10)
    clf.fit(X_train, y_train)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


def test_gradient_boosting_full_improv():
    #    parameters = {'min_samples_split': [2,4,8,10,20,50,100], 'n_estimators': [20,50,100], 'max_features':[4,8], 'max_depth':[3,4,5]}
    parameters = {'max_depth': 5, 'max_features': 4, 'min_samples_split': 8, 'n_estimators': 100}
    df = pd.read_csv('../datasets/full_features.csv')
    X, y = np.split(df, [19], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    all_names, X = np.split(X, [1], axis=1)
    clf = GradientBoostingClassifier(max_depth=5, max_features=4, min_samples_split=8, n_estimators=100)
    #  clf = GridSearchCV(clf,parameters, cv=10)
    #  clf.fit(X,y)
    #  print(clf.best_params_)

    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X, y, cv=10)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


def test_gradient_boosting_pca_improv():
    df = pd.read_csv('../datasets/pca.csv')
    X, y = np.split(df, [4], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    # test_names, X_test = np.split(X_test, [1], axis=1)
    # train_names, X_train = np.split(X_train, [1], axis=1)
    # all_names, X = np.split(X, [1], axis=1)
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X, y, cv=10)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


def test_random_forest_initial_improv():
    df = pd.read_csv('../datasets/backdoor_webshells_features.csv')
    X, y = np.split(df, [9], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    clf = RandomForestClassifier(n_estimators=20)
    all_names, X = np.split(X, [1], axis=1)
    scores = cross_val_score(clf, X, y, cv=20)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))
    dump(clf, '../saved_models/random_forest_v2.joblib')
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)

    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


#   print("\n Feature Importance")
#   for f in range(len(list(X))):
#       print("{}:{}".format(list(X)[f], clf.feature_importances_[f]))


def test_random_forest_full_improv():
    df = pd.read_csv('../datasets/full_features.csv')
    X, y = np.split(df, [19], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    clf = RandomForestClassifier(n_estimators=20)
    all_names, X = np.split(X, [1], axis=1)
    scores = cross_val_score(clf, X, y, cv=20)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))
    dump(clf, '../saved_models/random_forest_v2.joblib')
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


#  print("\n Feature Importance")
#  for f in range(len(list(X))):
#      print("{}:{}".format(list(X)[f], clf.feature_importances_[f]))


def test_random_forest_pca_improv():
    df = pd.read_csv('../datasets/pca.csv')
    X, y = np.split(df, [4], axis=1)
    y = np.asarray(y.values, dtype='int')
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    test_names, X_test = np.split(X_test, [1], axis=1)
    train_names, X_train = np.split(X_train, [1], axis=1)
    clf = RandomForestClassifier(n_estimators=5)
    all_names, X = np.split(X, [1], axis=1)
    scores = cross_val_score(clf, X, y, cv=20)
    for x in range(len(scores)):
        print("Accuracy for CV #{}: {}".format(x, scores[x]))
    print("AVERAGE:{}".format(sum(scores) / len(scores)))
    dump(clf, '../saved_models/random_forest_v2.joblib')
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)
    f1 = f1_score(y_test, y_score)
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall {}".format(recall))
    return f1, recall, precision, (sum(scores) / len(scores))


def get_average(func):
    f1_ = 0
    rc = 0
    pr = 0
    acc = 0
    for x in range(10):
        res = func()
        f1_ += res[0]
        rc += res[1]
        pr += res[2]
        acc += res[3]
    return f1_ / 10, rc / 10, pr / 10, acc / 10


"""
res = get_average(test_gradient_boosting_initial)
print("AVERAGE F1: {}".format(res[0]))
print("AVERAGE RECALL: {}".format(res[1]))
print("AVERAGE PRECISION:{}".format(res[2]))
print("AVERAGE ACCURACY:{}".format(res[3]))
print("---------------------------------")
print("----------------------------------")
res = get_average(test_gradient_boosting_initial_improv)
print("AVERAGE F1: {}".format(res[0]))
print("AVERAGE RECALL: {}".format(res[1]))
print("AVERAGE PRECISION:{}".format(res[2]))
print("AVERAGE ACCURACY:{}".format(res[3]))


res = get_average(test_gradient_boosting_full)
print("AVERAGE F1: {}".format(res[0]))
print("AVERAGE RECALL: {}".format(res[1]))
print("AVERAGE PRECISION:{}".format(res[2]))
print("AVERAGE ACCURACY:{}".format(res[3]))
print("---------------------------------")
print("----------------------------------")
res = get_average(test_gradient_boosting_full_improv)
print("AVERAGE F1: {}".format(res[0]))
print("AVERAGE RECALL: {}".format(res[1]))
print("AVERAGE PRECISION:{}".format(res[2]))
print("AVERAGE ACCURACY:{}".format(res[3]))

"""
if __name__ == "__main__":
    """
    print ("Gradient Boosting Initial")
    res = get_average(test_gradient_boosting_initial)
    print("AVERAGE F1: {}".format(res[0]))
    print("AVERAGE RECALL: {}".format(res[1]))
    print("AVERAGE PRECISION:{}".format(res[2]))
    print("AVERAGE ACCURACY:{}".format(res[3]))
    print("-----------------------------------")
    print ("Gradient Boosting Full")
    res = get_average(test_gradient_boosting_full)
    print("AVERAGE F1: {}".format(res[0]))
    print("AVERAGE RECALL: {}".format(res[1]))
    print("AVERAGE PRECISION:{}".format(res[2]))
    print("AVERAGE ACCURACY:{}".format(res[3]))
    print("-----------------------------------")
    print("Gradient Boosting PCA")
    res = get_average(test_gradient_boosting_pca)
    print("AVERAGE F1: {}".format(res[0]))
    print("AVERAGE RECALL: {}".format(res[1]))
    print("AVERAGE PRECISION:{}".format(res[2]))
    print("AVERAGE ACCURACY:{}".format(res[3]))
    print("-----------------------------------")
    print("Random Forest Initial")
    res = get_average(test_random_forest_initial)
    print("AVERAGE F1: {}".format(res[0]))
    print("AVERAGE RECALL: {}".format(res[1]))
    print("AVERAGE PRECISION:{}".format(res[2]))
    print("AVERAGE ACCURACY:{}".format(res[3]))
    print("-----------------------------------")
    """
    print("Random Forest Full")
    res = get_average(test_random_forest_full)
    print("AVERAGE F1: {}".format(res[0]))
    print("AVERAGE RECALL: {}".format(res[1]))
    print("AVERAGE PRECISION:{}".format(res[2]))
    print("AVERAGE ACCURACY:{}".format(res[3]))
    print("--------------------------------")

"""  
AVERAGE F1: 0.9617101069697529
AVERAGE RECALL: 0.9460006048747633
AVERAGE PRECISION:0.9780351286618952
AVERAGE ACCURACY:0.9880016226340167
  
  
  
AVERAGE F1: 0.968658345719245
AVERAGE RECALL: 0.9567400731333855
AVERAGE PRECISION:0.981126317535361
AVERAGE ACCURACY:0.99017422316634

"""
