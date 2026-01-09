import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def data_preprocess(yes_list, faces):
    rows = []
    for person, encs in faces.items():
        row = {'name': person}

        for i, value in enumerate(encs):
            row[f'f{i}'] = value

        rows.append(row)

    df = pd.DataFrame(rows)


    df['target'] = df['name'].isin(yes_list).astype(bool)

    X = df.filter(regex="^f").values 
    X = normalize(X, norm='l2')
 
    # binary target 
    y = df['target'].astype(int).to_numpy() # 1 = True, 0 = False 

    # group by indentity to avoid leakage 
    groups = df['name'].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
 
    return X_train, X_test, y_train, y_test 


