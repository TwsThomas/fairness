
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


"""
 1. **Demographic parity** [Dwork et al., 2012]() A fair classifier should make positive predictions on protected groups at the same rate as on all of X

Also the **positive predictions** are **higher** in X all than on the protected group. That violate *the Demographic parity* fairness.
"""
    
#     print(f"lambda: {lambda_:.6f}, "
#           f"train violation: {train_violation:.6f}, "
#           f"test violation: {test_violation:.6f}")



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
                       notion='equal_opportunity'):
    """ Compute the fairness violation w.r.t. the protected group

    notion is either 'demographic_parity' or 'equal_opportunity'
    """

    notion = 'equal_opportunity'
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
        P_true_pg = sum(y_test & protected_group) 
        # True Positif
        TP_X = sum(y_pred & y_test) 
        TP_pg = sum(y_pred & protected_group & y_test)    

        equal_opportunity_violation = TP_pg / P_true_pg - TP_X / P_true_X
        return equal_opportunity_violation


    


def get_generated_bias_data(n_samples=20000, lambda0 = 1.7, random_state=1):
    # generate y_bias according to P.1. 
    
    np.random.seed(random_state)
    X, y_true = make_classification(n_samples=n_samples,
                                    random_state=random_state, n_features=20) 
    protected_group = X[:,0] > 0 
    ratio_pg = np.sum(protected_group) / len(y_true)
    
    y_bias = y_true.copy()
    y_bias = y_true * np.exp(-lambda0 * (y_true * (protected_group/ ratio_pg - 1)))
    y_bias /= max(y_bias)
    
    y_obs = np.random.binomial(1, y_bias, len(y_bias))
    return train_test_split(X, y_true, y_obs,
                            protected_group,
                            random_state=random_state)

# X = np.random.uniform(-1, 1, size=(500,2))
# y_true = X[:,1] > 0
# pg = X[:,0] < -.7
# y_bias = y_true.copy()
# for i in range(len(X)):
#     if pg[i]:
#         if y_true[i]:
#             y_bias[i] = np.random.binomial(1, .7)
            