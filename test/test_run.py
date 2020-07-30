import os.path as osp
from cfxgb.Main import main

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
