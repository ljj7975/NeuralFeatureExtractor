from abc import ABC, abstractmethod
from utils import ensure_dir
import copy
import csv
import warnings
from pprint import pprint

# Must be modified to support per class saving

class FileHandler(ABC):
    def __init__(self, dir_path):
        super(FileHandler, self).__init__()
        self.meta = None

        self.features = []
        self.labels = []
        self.dir_path = dir_path

        self.feature_flushed_count = 0
        self.label_flushed_count = 0

        ensure_dir(self.dir_path)
        self._prepare_file(self.dir_path)

    def __del__(self):
        if self.meta is not None:
            if 'total' in self.meta:
                assert self.feature_flushed_count == self.meta['total'], \
                    "number of flushed feature count is not equal to total count"
                assert self.label_flushed_count == self.meta['total'], \
                    "number of flushed label count is not equal to total count"
        else:
            warnings.warn("meta file is not generated")

    @abstractmethod
    def _prepare_file(self, dir_path):
        pass

    def add_sample(self, feature, label):
        self.features = self.features + feature
        self.labels = self.labels + label

    def generate_meta_file(self, meta):
        '''
        recommanded to store
            - feature_size: feature tensor size
            - total: total number of samples
        '''

        meta_writer = csv.writer(open(self.dir_path + "/meta.csv", "w"))
        for key, val in meta.items():
            meta_writer.writerow([key, val])

        self.meta = meta

    @abstractmethod
    def flush_sample(self):
        pass

    def flush(self):
        self.feature_flushed_count += len(self.features)
        self.label_flushed_count += len(self.labels)

        for feature, label in zip(self.features, self.labels):
            self.flush_sample(feature, label)

        self.features = []
        self.labels = []