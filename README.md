<!-- [![Build Status](https://travis-ci.com/suryadheeshjith/CFXGB.svg?token=knvpVbu96NR4wtBr8v1E&branch=master)](https://travis-ci.com/suryadheeshjith/CFXGB)
[![codecov](https://codecov.io/gh/suryadheeshjith/CFXGB/branch/master/graph/badge.svg?token=XSIRP3ODQK)](https://codecov.io/gh/suryadheeshjith/CFXGB) -->

# CFXGB : Cascaded Forest and XGBoost Classifier

This is the official implementation for [CFXGB : Cascaded Forest and XGBoost Classifier](). CFXGB is a supervised machine learning model created by Surya Dheeshjith and Thejas Gubbi Sadashiva. The model is based on paper [1]. CFXGB is an extension of the model proposed in [2]. Implementation done in Python 3.

For more details, contact Surya Dheeshjith : surya.dheeshjith@gmail.com (or) Thejas Gubbi Sadashiva : tgs001@fiu.edu


## Details

We have included demo code for execution and a detailed explanation of how you can use your own dataset with custom model parameters.


![Pipeline](/images/Pipeline2.png)

The implementation of using Parent nodes in the decision trees as stated in the paper has also been implemented

![Parent nodes](/images/DecisionTree4.png)

### Requirements


* python==3.6.10
* numpy
* pandas
* argparse
* joblib
* psutil
* scikit-learn
* imblearn
* scipy
* simplejson


### Installation

1. Clone the Repository

    ```git clone https://github.com/suryadheeshjith/CFXGB.git```

2. Install dependencies

    ```python setup.py install```

3. You will also need to download the xgboost package. Use this conda command to do so

    ```conda install py-xgboost```



### Datasets Used    

  - TalkingData : https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection
  - Avazu : https://www.kaggle.com/c/avazu-ctr-prediction/data
  - Kad : https://www.kaggle.com/tbyrnes/advertising/data



### Repository Structure


    .
    ├── MANIFEST.in
    ├── README.md
    ├── images
    ├── requirements.txt
    ├── setup.cfg
    ├── setup.py
    ├── src
    │   └── cfxgb
    │       ├── DefaultParameters.json
    │       ├── Main.py
    │       ├── Parameters.json
    │       ├── __init__.py
    │       └── lib
    │           ├── CFXGB.py
    │           ├── __init__.py
    │           ├── cascade
    │           │   ├── __init__.py
    │           │   └── cascade_classifier.py
    │           ├── config.py
    │           ├── data_cache.py
    │           ├── estimators
    │           │   ├── __init__.py
    │           │   ├── base_estimator.py
    │           │   ├── kfold_wrapper.py
    │           │   └── sklearn_estimators.py
    │           ├── exp_utils.py
    │           └── utils
    │               ├── __init__.py
    │               ├── cache_utils.py
    │               ├── config_utils.py
    │               ├── debug_utils.py
    │               ├── log_utils.py
    │               └── metrics.py
    └── test
        ├── Datasets
        │   └── sample_data.csv
        ├── __init__.py
        ├── sample_parameters.json
        ├── test_run.py
        └── unit
            └── __init__.py


### Running demo code

To run the model, first change directories to the src directory

```
cd src
cd cfxgb
```

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

### References
[1] CFXGB: An Optimized and Effective Learning Approach for Click Fraud Detection. (https://www.sciencedirect.com/science/article/pii/S2666827020300165)

[2]  Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks.
In IJCAI-2017. (https://arxiv.org/abs/1702.08835v2 )





###### Coded by Surya Dheeshjith

###### Last updated : 30 July 2020
