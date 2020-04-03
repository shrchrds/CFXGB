# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import os, os.path as osp
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from joblib import Parallel, delayed
#from numba import njit
from ..utils.log_utils import get_logger
from ..utils.cache_utils import name2path
from .sklearn_estimators import GCXGBClassifier
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
            For Parallelisation
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
        if(self.args.ParentCols):
            #For Parallelisation - Train
            self.esti = None
            self.X = None
            self.n_dims= None
            self.val_idx = None
        
            #For Parallelisation - Test
            self.n_datas= None
            self.l0 = None
            self.l1 = None
        

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
        # Fit
        y_probas = []
        n_dims = X.shape[-1]
        n_datas = X.size / n_dims
        inverse = False

        if(self.args.ParentCols):
            _, val_idx = cv[0]
            l0 = np.zeros((n_stratify, ), dtype=np.float32)
            l1 = np.zeros((n_stratify, ), dtype=np.float32)
            esti = None
        
        
        for k in range(self.n_folds):
            est = self._init_estimator(k)
            if not inverse:
                train_idx, val_idx = cv[k]
            else:
                val_idx, train_idx = cv[k]
            # fit on k-fold train
            est.fit(X[train_idx].reshape((-1, n_dims)), y[train_idx].reshape(-1), cache_dir=cache_dir)
            
            if(self.args.ParentCols):
                esti = est._init_estimator()   #Need RFClassifier not GCRFClassifier
                
                esti.fit(X[train_idx].reshape((-1, n_dims)), y[train_idx].reshape(-1))  #Without this there is no estimators_
                self.estimatorec[k] = esti
                #print("Fit - ",k)
                if(not isinstance(esti,XGBClassifier)):
                    
                    #s0,s1=self.extracolsTrain(esti,X,n_dims,val_idx)
                    #s0 = Parallel(n_jobs=-1, verbose=0, prefer = "threads", backend="threading")(delayed(self.extracolsTrain)(esti,X,n_dims,val_idx)
                    self.esti = esti
                    self.X = X
                    self.n_dims = n_dims
                    self.val_idx = val_idx
                    ls = Parallel(n_jobs=self.args.Cores, verbose=0, prefer = "threads", backend="threading")(delayed(self.extracolsTrain)(i) for i in range(len(X[val_idx].reshape((-1, n_dims)))))

    
                    s0 = [i for i,j in ls]
                    s1 = [j for i,j in ls]
                

                    l0[val_idx]=s0
                    l1[val_idx] =s1
                

            # predict on k-fold validation
            y_proba = est.predict_proba(X[val_idx].reshape((-1, n_dims)), cache_dir=cache_dir)
            self.log_eval_metrics(self.name, y[val_idx], y_proba, eval_metrics, "train_{}".format(k))

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
                return y_probas,None,None

            return y_probas,l0,l1
        else:
            return y_probas,None,None
            
    def extracolsTrain(self,i):

        
        retls = []
        sum_0 = [0] * (len(self.X[self.val_idx].reshape((-1, self.n_dims))))
        sum_1 = [0] * (len(self.X[self.val_idx].reshape((-1, self.n_dims))))
        l=0
        
        for j, tree in enumerate(self.esti.estimators_):
            value = tree.tree_.value
            node_indicator = tree.decision_path(self.X[self.val_idx].reshape((-1, self.n_dims)))
            node_index = node_indicator.indices[node_indicator.indptr[i]:node_indicator.indptr[i + 1]]
            #print(value[node_index[-2]])
            l+=1
            sum_0[i]+=(value[node_index[-2]][0][0]/(value[node_index[-2]][0][0]+value[node_index[-2]][0][1]))
            sum_1[i]+=(value[node_index[-2]][0][1]/(value[node_index[-2]][0][0]+value[node_index[-2]][0][1]))
        
        
        sum_0[i]/=l
        sum_1[i]/=l
        retls = [sum_0[i],sum_1[i]]
        return retls


    def predict_proba(self, X_test):
        assert 2 <= len(X_test.shape) <= 3, "X_test.shape should be n x k or n x n2 x k"
        # K-Fold split
        n_dims = X_test.shape[-1]
        n_datas = X_test.size // n_dims
        if(self.args.ParentCols):
            l0 = np.zeros((n_datas, ), dtype=np.float32)
            l1 = np.zeros((n_datas, ), dtype=np.float32)
            esti = None
        for k in range(self.n_folds):
            
            
            ######################################################################################
            #TESTING - ADDITION OF COLUMNS
            if(self.args.ParentCols):
                esti = self.estimatorec[k]
                if(not isinstance(esti,XGBClassifier)):
                #self.extracolsTest(esti,X_test,n_datas,n_dims,l0,l1)
                    self.esti = esti
                    self.X=X_test
                    self.n_datas = n_datas
                    self.n_dims = n_dims
                    self.l0 = l0
                    self.l1 = l1
                    Parallel(n_jobs=self.args.Cores, verbose=0, prefer = "threads", backend="threading")(delayed(self.extracolsTest)(i) for i in range(n_datas))
                
                else:
                    esti = None
            
            
            ######################################################################################

            est = self.estimator1d[k]
            y_proba = est.predict_proba(X_test.reshape((-1, n_dims)), cache_dir=None)
#            if len(X_test.shape) == 3:
#                y_proba = y_proba.reshape((X_test.shape[0], X_test.shape[1], y_proba.shape[-1]))
            if k == 0:
                y_proba_kfolds = y_proba
            else:
                y_proba_kfolds += y_proba
                
                
        y_proba_kfolds /= self.n_folds
        if(self.args.ParentCols):
            l0 = self.l0
            l1  = self.l1
            try:
                l0/=(self.n_folds)
                l1/=(self.n_folds)
            except:
                pass
        

        if(self.args.ParentCols):
            if(esti == None):
                return y_proba_kfolds,None,None
            
            return y_proba_kfolds,l0,l1
        else:
            return y_proba_kfolds,None,None

    
    def extracolsTest(self,i):
        #esti.predict(X_test.reshape((-1, n_dims)))
        
        
        esti = self.esti
        X_test = self.X
        n_datas = self.n_datas
        n_dims = self.n_dims
        
        

        sum_0=0
        sum_1 = 0
        l=0
        for j, tree in enumerate(esti.estimators_):
            value = tree.tree_.value
            node_indicator = tree.decision_path(X_test.reshape((-1, n_dims)))
            node_index = node_indicator.indices[node_indicator.indptr[i]:node_indicator.indptr[i + 1]]
            #print(value[node_index[-2]])
            l+=1
            sum_0+=(value[node_index[-2]][0][0]/(value[node_index[-2]][0][0]+value[node_index[-2]][0][1]))
            sum_1+=(value[node_index[-2]][0][1]/(value[node_index[-2]][0][0]+value[node_index[-2]][0][1]))
        
        self.l0[i]+=sum_0/l
        self.l1[i]+=sum_1/l

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
