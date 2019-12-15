import os
import torch
import numpy as np
import librosa

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class GSCDataset(Dataset):
    def __init__(self, config):
        super(GSCDataset, self).__init__()

        for class_name in os.listdir(config["data_dir"]):
            dir_path = config["data_dir"] + "" + class_name
            if not os.path.isdir(dir_path):
                continue
            for file_name in os.listdir(dir_path):
                if "wav" not in file_name:
                    continue
                file_path = dir_path + "/" + file_name
                data = librosa.core.load(file_path, sr=16000)[0]
                print(class_name, file_name, len(data))
                break

    # def __len__(self):
    #     return self.len

    # def __getitem__(self, idx):
    #     noise_ind = np.random.randint(self.num_noise_sample)
    #     entry = self.data[idx] * self.noises[noise_ind]
    #     return torch.tensor(entry, dtype=torch.float), torch.tensor(self.labels[idx])

class GSCDataLoader(DataLoader):
    def __init__(self, config):
        self.dataset = GSCDataset(config)

        super(GSCDataLoader, self).__init__(
                    dataset=self.dataset,
                    batch_size=config["batch_size"],
                    shuffle=False)