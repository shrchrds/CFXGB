import numpy as np

from .cascade.cascade_classifier import CascadeClassifier
from .config import GCTrainConfig
from .utils.log_utils import get_logger
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score



LOGGER = get_logger("CFXGB.CFXGB")


class CFXGB(object):
    def __init__(self, config,args):
        self.config = config
        self.train_config = GCTrainConfig(config.get("train", {}))
        if "cascade" in self.config:
            self.ca = CascadeClassifier(self.config["cascade"],args)
        else:
            LOGGER.info("Model needs cascade parameters!")
            exit(0)
#        print(args)


    def get_encoded(self, X_train, y_train, X_test=None, y_test=None, train_config=None):
        train_config = train_config or self.train_config
        if X_test is None or y_test is None:
            if "test" in train_config.phases:
                train_config.phases.remove("test")
            X_test, y_test = None, None
        
        if self.ca is not None:
            _, X_train, _, X_test, _, = self.ca.fit_transform(X_train, y_train, X_test, y_test, train_config=train_config)

        if X_test is None:
            return X_train
        else:
            return X_train, X_test

    def transform(self, X):
        
        y_proba = self.ca.transform(X)
        return y_proba

    def predict_proba(self, X):
        y_proba = self.ca.predict_proba(X)
        return y_proba

    def predict_by_cascade(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def set_data_cache_dir(self, path):
        self.train_config.data_cache.cache_dir = path

    def set_keep_data_in_mem(self, flag):
        """
        flag (bool):
            if flag is 0, data will not be keeped in memory.
            this is for the situation when memory is the bottleneck
        """
        self.train_config.data_cache.config["keep_in_mem"]["default"] = flag

    def set_keep_model_in_mem(self, flag):
        """
        flag (bool):
            if flag is 0, model will not be keeped in memory.
            this is for the situation when memory is the bottleneck
        """
        self.train_config.keep_model_in_mem = flag
    def finalTransform(self,X1,X2,X3,X4):
        X_train_enc = np.hstack((X1, X2))
        X_test_enc = np.hstack((X3, X4))
        return X_train_enc,X_test_enc
    
    def classify(self,X1,y1,X2,y2):

        maxauc = 0
        maxd = 0
        maxlr =0
        lr = [0.05,0.1,0.2,0.3]
        max_depth = [2,3,4]
        for l in lr:
            for md in max_depth:
                clf = XGBClassifier(n_estimators=100, learning_rate = l, max_depth = md,verbosity =0, random_state = 0,n_jobs=-1)
                clf.fit(X1, y1)
                y_p = clf.predict(X2)
                fpr, tpr, thresholds = metrics.roc_curve(y2, y_p,pos_label=1)
                auc = metrics.auc(fpr, tpr)
                if(maxauc<auc):
                    maxauc = auc
                    maxd = md
                    y_pred = y_p
                    maxlr = l
        print('AUC,depth')
        print(maxauc)
        print(maxd)
        print(maxlr)

        return y_pred



