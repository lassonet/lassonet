import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from dataset_collection import DatasetCollection

class D1Dataset(DatasetCollection):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, max_npoints=2*1024, **kwargs)
        self.eval_return = [0, 2]