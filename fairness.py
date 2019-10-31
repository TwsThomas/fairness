
import numpy as np
import pandas as pd
import copy

np.set_printoptions(precision=4)

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def fairness(y_pred, y_test, G_test, verbose=False):
    
    G_test = np.array(G_test, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)
    y_test = np.array(y_test, dtype=bool)
    if verbose:
        for name, mask in zip(['G', '~G', 'all'], [G_test, ~G_test, [True]*len(y_test)]):
            print('---------', name, '------------')
            # print('Accuraccy:', accuracy_score(y_test[mask], y_pred[mask]))
            print('Error (1-Acc.):', round(1-accuracy_score(y_test[mask], y_pred[mask]),4))
            print(confusion_matrix(y_test[mask], y_pred[mask]) / sum(mask))
            print('% of positive preds =', y_pred[mask].sum() / len(y_pred[mask]))

        print('.\n')
    
    Zg_test = sum(G_test) / len(y_test)
    
    P = sum(y_pred) # Positive prediction
    P_G = sum(y_pred * G_test) # Positive prediction on G
    P_test = sum(y_test) # Positive label (in y_test)
    P_test_G = sum(y_test * G_test) # Positive label on G (in y_test)
    TP = sum(y_pred * y_test) # True Positif 
    TP_G = sum(y_pred * G_test * y_test) # True Positif on G
    
    fair_dp = P_G / Zg_test - P
    fair_dp /= len(y_pred)
    if verbose:
        print('fair Demo. Parity =\t', round(fair_dp,5))
    

    fair_eop = TP_G / P_test_G - TP / P_test
    fair_eop /= len(y_pred)
    if verbose:
        print('fair Equal Opp.   =\t', round(fair_eop,5))
    
    return fair_dp, fair_eop

def show_score(y_pred, y_test, G, verbose=False):
    #  return ['err_G', 'err', 'fair_dp', 'fair_eop'], 

    G_test = np.array(G, dtype=bool)
    
    if verbose:
        print('Error (1-Accuraccy):')
        print('on G \t on X')
    score = []
    for _, mask in zip(['G', 'all'], [G_test, [True]*len(y_test)]):
        score.append(1-accuracy_score(y_test[mask], y_pred[mask]))
        if verbose:
            print(round(score[-1],4), end='\t')

    fair_dp, fair_eop = fairness(y_pred, y_test, G, verbose=False)
    if verbose:
        print()
        print('fair_dp \t fair_eop')
        print(round(fair_dp,4),'\t', round(fair_eop,4))
    
    #  return ['err_G', 'err', 'fair_dp', 'fair_eop'], 
    return score + [fair_dp, fair_eop]


def get_w(lambda_, Gk, y_train):
    y_train = np.array(y_train, dtype=bool)
    re_weight = np.array([lambda_[k] * Gk[k] for k in range(len(Gk))])
    w_tilde = np.exp(np.sum(re_weight, axis = 0))  # n-vectors
    assert w_tilde.shape[0] == (len(y_train))
    w = np.array(w_tilde / (1 + w_tilde) * y_train +\
        1 / (1 + w_tilde) * (1 - y_train))
    return w


def fairness_error(lr, X, y_true, Gk, notion = 'dp'):
    
    y_pred = np.array(lr.predict(X), dtype=bool)
    y_true = np.array(y_true, dtype=bool)
    
    l_dp = []
    l_eop = []
        
    for G in Gk:
        G = np.array(G, dtype=bool)
    
        Zg_test = sum(G) / len(y_true)

        P = sum(y_pred) # Positive prediction
        P_G = sum(y_pred * G) # Positive prediction on G
        P_test = sum(y_true) # Positive label (in y_true)
        P_test_G = sum(y_true * G) # Positive label on G (in y_true)
        TP = sum(y_pred * y_true) # True Positif 
        TP_G = sum(y_pred * y_true * G) # True Positif on G

        fair_dp = abs(P_G / Zg_test - P)
        fair_dp /= len(y_pred)

        fair_eop = abs(TP_G / P_test_G - TP / P_test)
        fair_eop /= len(y_pred)

        l_dp.append(fair_dp)
        l_eop.append(fair_eop)

    if notion == 'dp':
        return np.array(l_dp)
    elif notion == 'eop':
        return np.array(l_eop)
    else:
        return None