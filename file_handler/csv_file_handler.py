from .file_handler import FileHandler
import csv

class CsvFileHandler(FileHandler):
    def __init__(self, dir_path):
        super(CsvFileHandler, self).__init__(dir_path)

    def _prepare_file(self, dir_path):
        self.feature_file_writer = csv.writer(open(dir_path + "/feature.csv", mode='w'))
        self.label_file_writer = csv.writer(open(dir_path + "/label.csv", mode='w'))

    def flush_sample(self, feature, label):
        self.feature_file_writer.writerow(feature)
        self.label_file_writer.writerow(label)
