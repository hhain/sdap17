## evaluation helpers
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, f1_score, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
def dt_eval(data, target, depth=4, class_weight="balanced"):
    """
    evals a single data set against itself
    target is the class variable, all other variables are used for training

    """
    class_var = target
    features = data.columns.values[~data.columns.str.contains(class_var)]

    clf = DecisionTreeClassifier(random_state=5, max_depth=depth, splitter="best",
                                 criterion="gini", class_weight=class_weight)
    clf.fit(data[features], data[class_var])
    #y_pred = clf.predict(data[features])
    #clf_rep = classification_report(data[class_var], y_pred)
    #print(clf_rep)
    return clf

from sklearn.naive_bayes import GaussianNB
def bay_eval(data, target):
    """
    evals a single data set against itself
    target is the class variable, all other variables are used for training

    """
    class_var = target
    features = data.columns.values[~data.columns.str.contains(class_var)]

    clf = GaussianNB()
    clf.fit(data[features], data[class_var])
    #y_pred = clf.predict(data[features])
    #clf_rep = classification_report(data[class_var], y_pred)
    #print(clf_rep)
    return clf

from sklearn.ensemble import RandomForestClassifier
def rf_eval(data, target, estimators=10, random_state=None):
    """
    evals a single data set against itself
    target is the class variable, all other variables are used for training
    
    """
    class_var = target
    features = data.columns.values[~data.columns.str.contains(class_var)]
    
    clf = RandomForestClassifier(n_estimators=estimators, class_weight="balanced", random_state=random_state)
    clf.fit(data[features], data[class_var])
    #y_pred = clf.predict(data[features])
    #clf_rep = classification_report(data[class_var], y_pred)
    #print(clf_rep)
    return clf

from sklearn import neighbors
def knn_eval(data, target, k):
    
    class_var = target
    features = data.columns.values[~data.columns.str.contains(class_var)]
    clf = neighbors.KNeighborsClassifier(k) #, weights=weights)
    clf.fit(data[features], data[class_var])
    
    return clf
#dt_eval(ds1_rv, target="target")
#ds1_rv.dropna(inplace=True)
#clf1=dt_eval(ds1_mf_rv, 'target', depth=4, class_weight=None)
#clf2=dt_eval(ds2_mf_rv, 'target', depth=4, class_weight=None)
#clf3=dt_eval(ds3_mf_rv, 'target', depth=4, class_weight=None)



# hold out cross validation
def hold_out_val(data, target, include_self=True, class_weight=None,
                 features=None, cl='rf', verbose=False, random_state=None):
    """ 
    performs simple hold-out validation
    :param data: list of datasets to evaluate
    :param verbose: if true print confusion matrix and classification report for each evaluation
    """

    f1_lst = []

    for d_idx, d in enumerate(data):
        d.dropna(inplace=True)
        class_var = target
        if features is None:
            features = d.columns.values[~d.columns.str.contains(class_var)]

        if cl == 'rf':
            clf = rf_eval(d, 'target', estimators=100, random_state=random_state)
        elif cl == 'dt':
            clf = dt_eval(d, 'target', depth=4)#, class_weight=class_weight)
        elif cl == 'nb':
            clf = bay_eval(d, 'target')
        else:
            clf = knn_eval(d, 'target', k=41)
            
        for e_idx, e in enumerate(data):
            if e_idx == d_idx and not include_self:
                continue
            e.dropna(inplace=True)
            if e_idx == d_idx and verbose:
                print("Self Evaluation", e_idx, "vs", d_idx)
            elif verbose:
                print("CV Evaluation", e_idx, "vs", d_idx)
                
            y_pred = clf.predict(e[features])
            f1 = np.mean(f1_score(y_true = e[class_var], y_pred=y_pred, average=None))
            f1_lst.append(f1)
            
            if verbose:
                clf_rep = classification_report(y_true = e[class_var], y_pred = y_pred)
                print(clf_rep)
                print("Confusion Matrix")
                print(confusion_matrix(y_true = e[class_var], y_pred = y_pred))
            
    f1_avg = np.mean(np.array(f1_lst))
    f1_std = np.std(np.array(f1_lst))
    if verbose:
        print("f1 average +/- f1 std:",f1_avg, "+/-", f1_std)
    
    return f1_avg, f1_std
    