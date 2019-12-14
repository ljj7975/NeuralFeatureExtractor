from abc import ABC, abstractmethod
from utils import ensure_dir
import copy
import csv
import warnings
from pprint import pprint

class FileHandler(ABC):
    def __init__(self, dir_path):
        super(FileHandler, self).__init__()
        self.meta_file_generated = False

        self.features = []
        self.labels = []
        self.dir_path = dir_path

        self.feature_flushed_count = 0
        self.label_flushed_count = 0

        ensure_dir(self.dir_path)
        self._prepare_file(self.dir_path)

    def __del__(self):
        if not self.meta_file_generated:
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

        print('storing meta file')
        pprint(meta)

        meta_writer = csv.writer(open(self.dir_path + "/meta.csv", "w"))
        for key, val in meta.items():
            meta_writer.writerow([key, val])

        self.meta_file_generated = True

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