
import os.path as osp
from cfxgb.Main import main

################################################################################
# DEFAULTS
################################################################################
parameters = 'sample_Parameters.json'
Dataset = 'sample_data'
ignore = False
RandomSamp = True
ParentCols  = 0
featureSelect = False
sample = False
################################################################################

# UPDATE IF REQUIRED
class _args_helper_:
    def __init__(self, parameters, Dataset, ignore, RandomSamp, ParentCols, featureSelect, sample):
        self.parameters = parameters
        self.Dataset  = Dataset
        self.ignore = ignore
        self.RandomSamp  = RandomSamp
        self.ParentCols = ParentCols
        self.featureSelect = featureSelect
        self.sample = sample


# UPDATE IF REQUIRED
def test_run():

    args = _args_helper_(parameters, Dataset, ignore, RandomSamp, ParentCols, featureSelect, sample)

    main(args)
