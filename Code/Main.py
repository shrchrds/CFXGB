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
import logging
import time
t = time.time()
################################################################################################################
#LOGGING INTO COMMAND_ERRORS
################################################################################################################
logger = logging.getLogger("CFXGB")
DEFAULT_LOGGING_DIR = "logs"
if not osp.exists(DEFAULT_LOGGING_DIR): os.makedirs(DEFAULT_LOGGING_DIR)
logging_path = osp.join(DEFAULT_LOGGING_DIR,"Command_Errors.log")
fh = logging.FileHandler(logging_path)
logger.addHandler(fh)

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
    parser.add_argument('-v',"--parentvaluecols", action="store_true",dest="ParentCols", default=False, help="Addition of more columns based on parent node values. Use this for larger columned datasets. RUN AT YOUR OWN RISK. (BETA). Default = False")
    parser.add_argument('-c',"--cores",dest="Cores", default=-1,type = int, help="Cores to be used during addition of more columns. RUN AT YOUR OWN RISK. (BETA). Default = -1 (All cores)")
    parser.add_argument('-f',"--featureselect", action="store_true",dest="featureSelect", default=False, help="Initial Feature Selection. Default = False")
    parser.add_argument('-s',"--sample1000", action="store_true",dest="sample1000", default=False, help="Sample 1000 instances. Default = False")
    args = parser.parse_args()
    return args


################################################################################################################
#MAIN
################################################################################################################

if __name__ == "__main__":

    
    #PARSING ARGUMENTS
    args = parse_args()
    
################################################################################################################
#ARGUMENT CHECK
################################################################################################################

    if args.Dataset is None:
        logger.error("Dataset required")
        exit(0)
    if args.Cores != -1 and args.ParentCols == False:
        logger.error("Parameter Error")
        exit(0)
    if args.parameters is None:
        logger.error("Model Parameters required")
        exit(0)
    else:
        config = load_json(args.parameters)
    print ("Loaded JSON")

    print ("JSON ----------------------------------------------------------------------------------")
    json1 = json.dumps(config,indent=4, separators=(". ", " = "))
    print(json1)
    print ("END OF JSON----------------------------------------------------------------------------")



################################################################################################################
#DATASET
################################################################################################################

    full_path = osp.join('Datasets',args.Dataset+'.csv')
    if not osp.exists(full_path):
        logger.error("Enter valid Dataset")
        exit(0)

        
    print(args.Dataset + " used")
    data = pd.read_csv(full_path)
    if(args.ignore):
        print ("First column ignored")
        data = data.iloc[:,1:]
     
    
    print ("Data Read Complete")
################################################################################################################


################################################################################################################
#Extra Columns
################################################################################################################

    if(args.ParentCols):
        print ("Columns based on parent nodes will be added. Cores to be used = ",args.Cores)


################################################################################################################

################################################################################################################
#Sample 10000
################################################################################################################

    if(args.sample1000):
        data = data.sample(n=1000)
        print("Sampled 1000 rows.....Current shape : "+str(data.shape))

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
        print("Feature Selection - Initial")
        clf = XGBClassifier(n_estimators=100, learning_rate = 0.3, max_depth = 4,verbosity =0, random_state = 0,n_jobs=-1)
        rfe = RFECV(clf, step=1, cv=5, verbose=0)
        X = rfe.fit_transform(X,y)

################################################################################################################



################################################################################################################
#TRAIN TEST SPLIT
################################################################################################################
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)  #stratify = y
    print ("Train Test Split complete")

    
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
        print ("Applied Random Under-Sampling")
        
    
    else:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        print ("No SAMPLING")

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
    print("X_train_enc.shape={}, X_test_enc.shape={}".format(X_train_enc.shape, X_test_enc.shape))

    

#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#
#XGBOOST
#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#


    y_pred = cfxgb.classify(X_train_enc,y_train,X_test_enc,y_test)



    print("Confusion Matrix - \n")
    print(confusion_matrix(y_test,y_pred))
    print("\nClassification - \n")
    print(classification_report(y_test,y_pred))
    print("Accuracy - \n")
    accurscor = accuracy_score(y_test, y_pred)
    print(accurscor)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    print("AUC ")
    auc = metrics.auc(fpr, tpr)
    print(auc)
    print("Time - ",time.time()-t)
    print(str(sys.argv))

#$#$#$#$#$#$#$$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$$#$$$#$#$#$#$$#$#$#$$#$#$#$#$#$#$#$#$#$#$#
