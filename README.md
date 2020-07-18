# CFXGB : Cascaded Forest and XGBoost Classifier

This is the official implementation for [CFXGB : Cascaded Forest and XGBoost Classifier](). CFXGB is a supervised machine learning model created by Surya Dheeshjith and Thejas Gubbi Sadashiva. The model is based on paper [1]. CFXGB is an extension of the model proposed in [2]. Implementation done in Python 3.

For more details, contact Surya Dheeshjith : surya.dheeshjith@gmail.com (or) Thejas Gubbi Sadashiva : tgs001@fiu.edu

### References
[1] CFXGB: An Optimized and Effective Learning Approach for Click Fraud Detection.  
[2]  Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks.
In IJCAI-2017. (https://arxiv.org/abs/1702.08835v2 )



## Details

We have included demo code for execution and a detailed explanation of how you can use your own dataset with custom model parameters.


![Pipeline](/images/Pipeline2.png)



### Requirements

* All required packages are in requirements.txt

```pip install -r requirements.txt```

*If facing issues downloading xgboost package, use this conda command*

```conda install py-xgboost```

* python==3.7.3
* numpy==1.16.4
* pandas==0.24.2
* argparse
* joblib==0.13.2
* psutil==5.7.0
* scikit-learn==0.21.2
* imblearn==0.5.0
* scipy==1.2.1
* simplejson
* xgboost==0.90

### Datasets Used    

  - TalkingData : https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection
  - Avazu : https://www.kaggle.com/c/avazu-ctr-prediction/data
  - Kad : https://www.kaggle.com/tbyrnes/advertising/data
  - UNSW : https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
  - CICID : https://www.unb.ca/cic/datasets/ids-2017.html

### Repository Structure

    .
    ├── Code
    │   ├── CFXGB
    │   │   ├── CFXGB.py
    │   │   ├── __init__.py
    │   │   ├── cascade
    │   │   │   ├── __init__.py
    │   │   │   ├── cascade_classifier.py
    │   │   ├── config.py
    │   │   ├── data_cache.py
    │   │   ├── estimators
    │   │   │   ├── __init__.py
    │   │   │   ├── base_estimator.py
    │   │   │   ├── kfold_wrapper.py
    │   │   │   ├── sklearn_estimators.py
    │   │   ├── exp_utils.py
    │   │   └── utils
    │   │       ├── __init__.py
    │   │       ├── cache_utils.py
    │   │       ├── config_utils.py
    │   │       ├── debug_utils.py
    │   │       ├── log_utils.py
    │   │       ├── metrics.py
    │   ├── DefaultParameters.json
    │   ├── Main.py
    │   ├── Parameters.json
    │   ├── ParametersTest.json
    ├── README.md
    ├── images
    └── requirements.txt



### Running demo code

To run the model, first change directories to the Code directory

```cd code```

Now to run a dataset like Kad with parameters in DefaultParameters.json, run this :

```python3 Main.py -d Kad -p DefaultParameters.json```

### List of Command-line Parameters

* -h --help : List the parameters and their use.

* -d --dataset : A dataset must be considered for learning. This parameter takes the dataset csv file name. This parameter **must** be passed.    

* -p --parameters : Model Parameters are passed using a json file. This parameter must be used to specify the name of json file. This parameter **must** be passed.  

* -i --ignore : Ignore the first column. (For some cases).  
                Default = False

* -r --randomsamp : Balance the dataset using random under sampling. (Use for imbalanced datasets).   
                    Default = False

* -v --parentvaluecols : Addition of columns based on class distributions of parents of leaf nodes in the decision tree.    
                                Default = False

* -c --cores : Number of cores to be used during addition of columns (When -v is True).    
                         Default = -1 (All cores)

### How to run code for different datasets and model parameters

```python3 Main.py -d <Dataset_Name> -p <Parameter_list>.json```




###### Coded by Surya Dheeshjith

###### Last updated : 18 July 2020
