import numpy
import os
import pickle, cPickle
import gzip
import random
from DataHelper import DatasetHelper
from FileHelper import *

class MNistNewDataHelper(DatasetHelper):
    def _convert_type(self,
                      _variables,
                      _types):
        variables = []
        for _variable, _type in zip(_variables, _types):
            variables.append(_variable.astype(_type))
        return variables

    def __init__(self,
                 _dataset_path = None):
        DatasetHelper.__init__(self)

        # Check parameters
        check_not_none(_dataset_path, 'datasetPath');

        # Set parameters
        self.dataset_path  = _dataset_path

        # Load the dataset
        with gzip.open(self.dataset_path, 'rb') as _file:
            try:
                _train_set, _valid_set, _test_set = pickle.load(_file, encoding='latin1')
            except:
                _train_set, _valid_set, _test_set = pickle.load(_file)

        self.train_known_data_x, \
        self.train_known_data_y, \
        self.train_unknown_data_x, \
        self.train_unknown_data_y, = _train_set

        self.valid_known_data_x, \
        self.valid_known_data_y, \
        self.valid_unknown_data_x, \
        self.valid_unknown_data_y, = _valid_set

        self.test_known_data_x, \
        self.test_known_data_y, \
        self.test_unknown_data_x, \
        self.test_unknown_data_y, = _test_set

        # Reshape data to num_samples, num_channels, width, height
        self.train_known_data_x   = self._reshape_to_image(self.train_known_data_x)
        self.train_unknown_data_x = self._reshape_to_image(self.train_unknown_data_x)
        self.valid_known_data_x   = self._reshape_to_image(self.valid_known_data_x)
        self.valid_unknown_data_x = self._reshape_to_image(self.valid_unknown_data_x)
        self.test_known_data_x    = self._reshape_to_image(self.test_known_data_x)
        self.test_unknown_data_x  = self._reshape_to_image(self.test_unknown_data_x)

    def _reshape_to_image(self,
                          _data):
        data = _data.reshape((len(_data), 1, 28, 28))
        return data
