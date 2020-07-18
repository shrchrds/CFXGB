##########################################################################################
##########################################################################################
##########################################################################################

#Contact these authors for further related questions

#Authors : Surya Dheeshjith <surya.dheeshjith@gmail.com>
#        : Thejas Gubbi Sadashiva <tgs001@fiu.edu>


##########################################################################################
##########################################################################################
##########################################################################################

import argparse
import sys
import numpy as np
import sys
import pickle
import json
import os.path as osp
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,roc_auc_score
import pandas as pd
import random
from CFXGB.CFXGB import CFXGB
from CFXGB.utils.config_utils import load_json
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from imblearn.under_sampling import RandomUnderSampler
from CFXGB.utils.log_utils import get_logger
import logging
import time
t = time.time()



################################################################################################################
#ARGUMENT FUNCTION
################################################################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='All parameters usage')
    parser.add_argument('-p',"--parameters", dest="parameters", type=str, default='DefaultParameters.json', help="Add a JSON file with parameters. Refer DefaultParameters.json. Default = DefaultParameters.json")
    parser.add_argument('-d',"--dataset", dest="Dataset", type=str, default=None, help="Dataset (csv file). Refer the datasets in Dataset Folder.")
    parser.add_argument('-i',"--ignore",action="store_true", dest="ignore", default=False, help="If dataset was saved using pandas, Use this parameter to ignore first column (Redundant column). Default = False")
    parser.add_argument('-r',"--randomsamp", action="store_true",dest="RandomSamp", default=False, help="If dataset is imbalanced, Random sampling will balance the dataset. Default = False")
    parser.add_argument('-v',"--parentvaluecols", default=0,type = int,dest="ParentCols", help="Number of levels of parent node values to consider. Use this for larger columned datasets. RUN AT YOUR OWN RISK. (BETA). Default = 0")
    parser.add_argument('-f',"--featureselect", action="store_true",dest="featureSelect", default=False, help="Initial Feature Selection. Default = False")
    parser.add_argument('-s',"--sample", dest="sample", default=False,type = int, help="Sample instances")
    args = parser.parse_args()
    return args


################################################################################################################
#MAIN
################################################################################################################

if __name__ == "__main__":
    
    #Logging
    logger = get_logger("CFXGB.CFXGB")
    
    #PARSING ARGUMENTS
    args = parse_args()

################################################################################################################
#ARGUMENT CHECK
################################################################################################################

    if args.Dataset is None:
        logger.error("Dataset required")
        exit(0)

    if args.ParentCols<0:
        logger.error("Enter valid levels")
        exit(0)

    if args.parameters is None:
        logger.error("Model Parameters required")
        exit(0)
    else:
        config = load_json(args.parameters)
    logger.info ("Loaded JSON")

    logger.info ("JSON ----------------------------------------------------------------------------------")
    json1 = json.dumps(config,indent=4, separators=(". ", " = "))
    logger.info(json1)
    logger.info ("END OF JSON----------------------------------------------------------------------------")



################################################################################################################
#DATASET
################################################################################################################

    full_path = osp.join('Datasets',args.Dataset+'.csv')
    if not osp.exists(full_path):
        logger.error("Enter valid Dataset")
        exit(0)


    logger.info(args.Dataset + " used")
    data = pd.read_csv(full_path)
    if(args.ignore):
        logger.info ("First column ignored")
        data = data.iloc[:,1:]


    logger.info("Data Read Complete")
################################################################################################################


################################################################################################################
#Extra Columns
################################################################################################################

    if(args.ParentCols):
        logger.info("{} level(s) of parent nodes will be added. ".format(args.ParentCols))

    else:
        logger.info("Parent nodes not considered")
################################################################################################################

################################################################################################################
#Sample
################################################################################################################

    if(args.sample):
        weights=data.groupby(data.columns[-1])[data.columns[-1]].transform('count')
        if(len(np.unique(weights))==1):
            logging.info("Equal weights already.")
            data = data.sample(n=args.sample,random_state=0)
        else:
            sum = np.sum(np.unique(weights))
            weights = sum - weights
            data = data.sample(n=args.sample,weights=weights,random_state=0)
        logger.info("Distribution after sampling : \n{}".format(data.iloc[:,-1].value_counts()))
        

################################################################################################################

################################################################################################################
# X,y
################################################################################################################

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

################################################################################################################


################################################################################################################
#Feature Selection (Initial)
################################################################################################################

    if(args.featureSelect):
        logger.info("Feature Selection - Initial")
        clf = XGBClassifier(n_estimators=100, learning_rate = 0.3, max_depth = 4,verbosity =0, random_state = 0,n_jobs=-1)
        rfe = RFECV(clf, step=1, cv=5, verbose=0)
        X = rfe.fit_transform(X,y)

################################################################################################################



################################################################################################################
#TRAIN TEST SPLIT
################################################################################################################

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)  #stratify = y
    logger.info("Train Test Split complete")


################################################################################################################

#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#
#TRAINING
#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#




#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#
#SAMPLING
#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#


    if(args.RandomSamp):
        rus = RandomUnderSampler(random_state=0)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        logger.info("Applied Random Under-Sampling")


    else:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        logger.info("No Random Under-Sampling")

    X_test = np.array(X_test)

#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#
#MODEL
#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#

    #CFXGB
    cfxgb = CFXGB(config,args)


#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#



#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#
#CASCADED FOREST AS TRANSFORMER
#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#



    X_train_enc = cfxgb.get_encoded(X_train, y_train)
    X_test_enc = cfxgb.transform(X_test)


    #Final Transformation
    X_train_enc,X_test_enc = cfxgb.finalTransform(X_train,X_train_enc,X_test,X_test_enc)
#    X_train_enc = pd.DataFrame(X_train_enc)
#    X_train_enc.to_csv("X_train_enc.csv")
#    X_test_enc = pd.DataFrame(X_train_enc)
#    X_test_enc.to_csv("X_test_enc.csv")
    logger.info("X_train_enc.shape={}, X_test_enc.shape={}".format(X_train_enc.shape, X_test_enc.shape))



#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#
#XGBOOST
#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#


    y_pred = cfxgb.classify(X_train_enc,y_train,X_test_enc,y_test)



    logger.info("Confusion Matrix - \n{}".format(confusion_matrix(y_test,y_pred)))
    logger.info("\nClassification Report - \n{}".format(classification_report(y_test,y_pred)))
    logger.info("Accuracy - {}\n".format(accuracy_score(y_test, y_pred)))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    logger.info("AUC ")
    auc = metrics.auc(fpr, tpr)
    logger.info(auc)
    logger.info("Time - {}".format(time.time()-t))
    logger.info("Arguments used in this run : {}".format(str(sys.argv)))
    
    logging.shutdown()

#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#
