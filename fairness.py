
import numpy as np
import pandas as pd

np.set_printoptions(precision=4)

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from joblib import Memory, Parallel, delayed
location = './cachedir'
memory = Memory(location, verbose=0)




def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
                #    vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def fairness_violation(y_pred, y_test, protected_group,
                       notion='demographic_parity'):
    """ Compute the fairness violation w.r.t. the protected group

    notion is either 'demographic_parity' or 'equal_opportunity'
    """

    if notion=='demographic_parity':
        # demographic_parity compare the positive prediction rate 
        
        # Cardinality on each group
        nb_pg = sum(protected_group)  
        nb_X = len(y_pred)
        # Positive prediction
        P_X = sum(y_pred)   
        P_pg = sum(y_pred * protected_group)
        
        demographic_parity_violation = P_pg / nb_pg - P_X / nb_X
        return demographic_parity_violation

    elif notion=='equal_opportunity':
        # equal_opportunity compare true positive rates

        # Positive true label
        P_true_X = sum(y_test) 
        P_true_pg = sum(y_test * protected_group) 
        # True Positif
        TP_X = sum(y_pred * y_test) 
        TP_pg = sum(y_pred * protected_group * y_test)    

        equal_opportunity_violation = TP_pg / P_true_pg - TP_X / P_true_X
        return equal_opportunity_violation


def reweigth_samples(lambda_, protected_group, y_bias):
    # Compute the reweighting w.r.t the protected group
    
    w_tilde = np.exp(lambda_ * protected_group)
    w = np.array(w_tilde / (1 + w_tilde) * y_bias +\
                 1 / (1 + w_tilde) * (1 - y_bias))
    return w