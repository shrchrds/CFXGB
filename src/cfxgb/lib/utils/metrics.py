# -*- coding:utf-8 -*-
import numpy as np



def accuracy(y_true, y_pred):
    return 1.0 * np.sum(y_true == y_pred) / len(y_true)

def accuracy_pb(y_true, y_proba):
    y_true = y_true.reshape(-1)
    y_pred = np.argmax(y_proba.reshape((-1, y_proba.shape[-1])), 1)
    return 1.0 * np.sum(y_true == y_pred) / len(y_true)

