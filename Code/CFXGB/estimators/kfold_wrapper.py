
import os, os.path as osp
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from ..utils.log_utils import get_logger
from ..utils.cache_utils import name2path
from xgboost import XGBClassifier

LOGGER = get_logger("gcforest.estimators.kfold_wrapper")

class KFoldWrapper(object):
    """
    K-Fold Wrapper
    """
    def __init__(self, name, n_folds, est_class, est_args, args, random_state=None):
        """
        Parameters
        ----------
        n_folds (int):
            Number of folds.
            If n_folds=1, means no K-Fold
        est_class (class):
            Class of estimator
        args:
            args
        est_args (dict):
            Arguments of estimator
        random_state (int):
            random_state used for KFolds split and Estimator
        """
        self.name = name
        self.n_folds = n_folds
        self.est_class = est_class
        self.est_args = est_args
        self.random_state = random_state
        self.estimator1d = [None for k in range(self.n_folds)]
        self.estimatorec = [None for k in range(self.n_folds)]
        self.args = args


    def _init_estimator(self, k):
        est_args = self.est_args.copy()
        est_name = "{}/{}".format(self.name, k)
        est_args["random_state"] = self.random_state
        return self.est_class(est_name, est_args)

    def fit_transform(self, X, y, y_stratify, cache_dir=None, test_sets=None, eval_metrics=None, keep_model_in_mem=True):
        if cache_dir is not None:
            cache_dir = osp.join(cache_dir, name2path(self.name))
        assert 2 <= len(X.shape) <= 3, "X.shape should be n x k or n x n2 x k"
        assert len(X.shape) == len(y.shape) + 1
        assert X.shape[0] == len(y_stratify)
        test_sets = test_sets if test_sets is not None else []
        eval_metrics = eval_metrics if eval_metrics is not None else []
        # K-Fold split
        n_stratify = X.shape[0]
        if self.n_folds == 1:
            cv = [(range(len(X)), range(len(X)))]
        else:
            if y_stratify is None:
                skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                cv = [(t, v) for (t, v) in skf.split(len(n_stratify))]
            else:
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify), y_stratify)]

        #print("cv",cv[0])

        # Fit
        y_probas = []
        n_dims = X.shape[-1]
        n_datas = X.size / n_dims
        inverse = False
        n_classes = len(np.unique(y))

        if(self.args.ParentCols):
            par_col_est = np.zeros((n_stratify,n_classes*self.args.ParentCols), dtype=np.float32)


        for k in range(self.n_folds):
            est = self._init_estimator(k)
            # print(est)
            if not inverse:
                train_idx, val_idx = cv[k]
            else:
                val_idx, train_idx = cv[k]

            # fit on k-fold train
            esti = est.fit(X[train_idx].reshape((-1, n_dims)), y[train_idx].reshape(-1),cache_dir=cache_dir)


            # predict on k-fold validation
            y_proba = est.predict_proba(X[val_idx].reshape((-1, n_dims)))
            self.log_eval_metrics(self.name, y[val_idx], y_proba, eval_metrics, "train_{}".format(k))

            if(self.args.ParentCols):

                self.estimatorec[k] = esti
                if(not isinstance(esti,XGBClassifier)):
                    par_col_est[val_idx] = self.extracols(esti,data=X[val_idx].reshape((-1, n_dims)),n_classes=n_classes,num_cols=self.args.ParentCols,Train=True)


            # merging result
            if k == 0:
                if len(X.shape) == 2:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1]), dtype=np.float32)
                #print(2)

                y_probas.append(y_proba_cv)


            y_probas[0][val_idx, :] += y_proba

            if keep_model_in_mem:
                self.estimator1d[k] = est


            # test
            for vi, (prefix, X_test, y_test) in enumerate(test_sets):
                y_proba = est.predict_proba(X_test.reshape((-1, n_dims)), cache_dir=cache_dir)
                if len(X.shape) == 3:
                    y_proba = y_proba.reshape((X_test.shape[0], X_test.shape[1], y_proba.shape[-1]))
                if k == 0:
                    y_probas.append(y_proba)
                else:
                    y_probas[vi + 1] += y_proba


        if inverse and self.n_folds > 1:
            y_probas[0] /= (self.n_folds - 1)
        for y_proba in y_probas[1:]:
            y_proba /= self.n_folds


        # log
        self.log_eval_metrics(self.name, y, y_probas[0], eval_metrics, "train_cv")

        for vi, (test_name, X_test, y_test) in enumerate(test_sets):
            if y_test is not None:
                self.log_eval_metrics(self.name, y_test, y_probas[vi + 1], eval_metrics, test_name)

        if(self.args.ParentCols):
            if(isinstance(esti,XGBClassifier)):
                return y_probas,None, False

            return y_probas,par_col_est, True
        else:
            return y_probas,None, False


    def predict_proba(self, X_test, n_classes):
        assert 2 <= len(X_test.shape) <= 3, "X_test.shape should be n x k or n x n2 x k"
        # K-Fold split
        n_dims = X_test.shape[-1]
        n_datas = X_test.size // n_dims
        if(self.args.ParentCols):
            par_col_est = np.zeros((n_datas,n_classes*self.args.ParentCols), dtype=np.float32)
            esti = None

        for k in range(self.n_folds):
            if(self.args.ParentCols):
                esti = self.estimatorec[k]
                if(not isinstance(esti,XGBClassifier)):

                    par_col_est+= self.extracols(esti,data=X_test.reshape((-1, n_dims)),n_classes=n_classes,num_cols=self.args.ParentCols,Train=True)



            est = self.estimator1d[k]
            y_proba = est.predict_proba(X_test.reshape((-1, n_dims)), cache_dir=None)

            if k == 0:
                y_proba_kfolds = y_proba
            else:
                y_proba_kfolds += y_proba


        y_proba_kfolds /= self.n_folds
        if(self.args.ParentCols and esti!=None):
            par_col_est/=self.n_folds


        if(self.args.ParentCols):
            if(isinstance(esti,XGBClassifier)):
                return y_proba_kfolds,None,False

            return y_proba_kfolds,par_col_est,True
        else:
            return y_proba_kfolds,None,False


    def extracols(self,esti,data,n_classes,num_cols,Train=True):


        l1 = len(data)
        cols = np.zeros((l1,num_cols*n_classes))
        l=0

        for j, tree in enumerate(esti.estimators_):
            value = tree.tree_.value
            node_indicator = tree.decision_path(data)
            for i in range(len(data)):
                node_index = node_indicator.indices[node_indicator.indptr[i]:node_indicator.indptr[i + 1]]
                if len(node_index)<num_cols+1: # Need to exclude leaf node
                    add_list = [node_index[-1]]*(num_cols-len(node_index)+1)  # Concatenating the leaf node so it's distribution will be used here.
                    node_index = np.concatenate((add_list,node_index),axis=0)

                node_len = len(node_index)
                node_index = node_index[::-1] # To reverse the list
                list = node_index[1:num_cols+1]

                # For each level in the tree, we obtain parent node and get class distributions
                for n in range(num_cols):
                    cols[i][n*n_classes:n*n_classes+n_classes]+= value[list[n]][0]/np.sum(value[list[n]][0])

            l+=1

        cols/=l
        return cols


    def log_eval_metrics(self, est_name, y_true, y_proba, eval_metrics, y_name):
        """
            y_true (ndarray): n or n1 x n2
            y_proba (ndarray): n x n_classes or n1 x n2 x n_classes
            """
        if eval_metrics is None:
            return
        for (eval_name, eval_metric) in eval_metrics:
            accuracy = eval_metric(y_true, y_proba)
            LOGGER.info("Accuracy({}.{}.{})={:.2f}%".format(est_name, y_name, eval_name, accuracy * 100.))
