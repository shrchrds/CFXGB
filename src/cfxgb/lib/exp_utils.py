
import numpy as np
from scipy.sparse import issparse

from cfxgb.lib.utils.log_utils import get_logger

LOGGER = get_logger('CFXGB.exp_utils')

def load_model_config(model_path, log_name=None):
    import json
    from .utils.config_utils import load_json
    config = load_json(model_path)
    if log_name is not None:
        logger = get_logger(log_name)
        logger.info(log_name)
        logger.info("\n" + json.dumps(config, sort_keys=True, indent=4, separators=(',', ':')))
    return config


def concat_datas(datas):
    if type(datas) != list:
        return datas
    for i, data in enumerate(datas):
        datas[i] = data.reshape((data.shape[0], -1))
    return np.concatenate(datas, axis=1)

def data_norm(X_train, X_test):
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    return X_mean, X_std

def append_origin(X, X_origin):
    return np.hstack(( X.reshape((X.shape[0]), -1), X_origin.reshape((X_origin.shape[0], -1)) ))

def prec_ets(n_trees, X_train, y_train, X_test, y_test, random_state=None):
    """
    ExtraTrees
    """
    from sklearn.ensemble import ExtraTreesClassifier
    if not issparse(X_train):
        X_train = X_train.reshape((X_train.shape[0], -1))
    if not issparse(X_test):
        X_test = X_test.reshape((X_test.shape[0], -1))
    LOGGER.info('start predict: n_trees={},X_train.shape={},y_train.shape={},X_test.shape={},y_test.shape={}'.format(
        n_trees, X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    clf = ExtraTreesClassifier(n_estimators=n_trees, max_depth=None, n_jobs=-1, verbose=1, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    prec = float(np.sum(y_pred == y_test)) / len(y_test)
    LOGGER.info('prec_ets{}={:.6f}%'.format(n_trees, prec*100.0))
    return clf, y_pred

def prec_rf(n_trees, X_train, y_train, X_test, y_test):
    """
    ExtraTrees
    """
    from sklearn.ensemble import RandomForestClassifier
    if not issparse(X_train):
        X_train = X_train.reshape((X_train.shape[0], -1))
    if not issparse(X_test):
        X_test = X_test.reshape((X_test.shape[0], -1))
    LOGGER.info('start predict: n_trees={},X_train.shape={},y_train.shape={},X_test.shape={},y_test.shape={}'.format(
        n_trees, X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=None, n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    prec = float(np.sum(y_pred == y_test)) / len(y_test)
    LOGGER.info('prec_rf{}={:.6f}%'.format(n_trees, prec*100.0))
    return clf, y_pred


def prec_log(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    if not issparse(X_train):
        X_train = X_train.reshape((X_train.shape[0], -1))
    if not issparse(X_test):
        X_test = X_test.reshape((X_test.shape[0], -1))
    LOGGER.info('start predict: X_train.shape={},y_train.shape={},X_test.shape={},y_test.shape={}'.format(
        X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    clf = LogisticRegression(solver='sag', n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    prec = float(np.sum(y_pred == y_test)) / len(y_test)
    LOGGER.info('prec_log={:.6f}%'.format(prec*100.0))
    return clf, y_pred

def plot_forest_all_proba(y_proba_all, y_gt):
    from matplotlib import pylab
    N = len(y_gt)
    num_tree = len(y_proba_all)
    pylab.clf()
    mat = np.zeros((num_tree, N))
    LOGGER.info('mat.shape={}'.format(mat.shape))
    for i in range(num_tree):
        mat[i,:] = y_proba_all[i][(range(N), y_gt)]
    pylab.matshow(mat, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    pylab.grid(False)
    pylab.show()

def plot_confusion_matrix(cm, label_list, title='Confusion matrix', cmap=None):
    from matplotlib import pylab
    cm = np.asarray(cm, dtype=np.float32)
    for i, row in enumerate(cm):
        cm[i] = cm[i] / np.sum(cm[i])
    #import matplotlib.pyplot as plt
    #plt.ion()
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(label_list)))
    ax.set_xticklabels(label_list, rotation='vertical')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(range(len(label_list)))
    ax.set_yticklabels(label_list)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(False)
    pylab.savefig('test.jpg')
    pylab.show()
