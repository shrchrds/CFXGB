import os.path as osp
from cfxgb.Main import main

parameters", type=str, default='DefaultParameters.json', help="Add a JSON file with parameters. Refer DefaultParameters.json. Default = DefaultParameters.json")
parser.add_argument('-d',"--dataset", dest="Dataset", type=str, default=None, help="Dataset (csv file). Refer the datasets in Dataset Folder.")
parser.add_argument('-i',"--ignore",action="store_true", dest="ignore", default=False, help="If dataset was saved using pandas, Use this parameter to ignore first column (Redundant column). Default = False")
parser.add_argument('-r',"--randomsamp", action="store_true",dest="RandomSamp", default=False, help="If dataset is imbalanced, Random sampling will balance the dataset. Default = False")
parser.add_argument('-v',"--parentvaluecols", default=0,type = int,dest="ParentCols", help="Number of levels of parent node values to consider. Use this for larger columned datasets. RUN AT YOUR OWN RISK. (BETA). Default = 0")
parser.add_argument('-f',"--featureselect", action="store_true",dest="featureSelect", default=False, help="Initial Feature Selection. Default = False")
parser.add_argument('-s',"--sample", dest="sample", default=False,type = int, help="Sample instances")

################################################################################
# DEFAULTS
################################################################################
data_fol = 'test/Data'
parameters_file = 'sample_parameters.json'
Dataset_file = 'sample_data.csv'
ignore = False
RandomSamp = True
ParentCols  = 0
featureSelect = False
sample = False
################################################################################

# UPDATE IF REQUIRED
class _patch_args_helper_:
    def __init__(self, parameters, Dataset, ignore, RandomSamp, ParentCols, featureSelect, sample):
        self.parameters = parameters
        self.Dataset  = Dataset
        self.ignore = ignore
        self.RandomSamp  = RandomSamp
        self.ParentCols = ParentCols
        self.featureSelect = featureSelect
        self.sample = sample


# UPDATE IF REQUIRED
def test_patch():

    parameters = osp.join(data_fol,parameters_file)
    Dataset = osp.join(data_fol,Dataset_file)
    args = _patch_args_helper_(parameters, Dataset, ignore, RandomSamp, ParentCols, featureSelect, sample)

    generate(args)
