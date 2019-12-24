from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import shutil
import numpy as np
from tqdm import tqdm

class FERGDataLoader(DataLoader):
    def __init__(self, config):

        processed_dir = os.path.join(config["data_dir"], 'processed')
        if not os.path.exists(processed_dir):
            print("preprocess the raw data")
            os.mkdir(processed_dir)

            characters = ["aia", "bonnie", "jules", "malcolm", "mery", "ray"]
            for character in tqdm(characters):
                char_folder = os.path.join(config["data_dir"], character)

                for emotion_folder in os.listdir(char_folder):
                    source_folder = os.path.join(char_folder, emotion_folder)
                    if not os.path.isdir(source_folder):
                        continue
                    emotion = emotion_folder.split('_')[1]

                    if emotion in ["neutral", "surprise"]:
                        continue

                    target_folder = os.path.join(processed_dir, emotion)
                    if not os.path.exists(target_folder):
                        os.mkdir(target_folder)

                    for file_name in os.listdir(source_folder):
                        source_file = os.path.join(source_folder, file_name)
                        target_file = os.path.join(target_folder, file_name)

                        shutil.copy(source_file, target_file)

        trsfm = transforms.Compose([
            transforms.Resize([28, 28]),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.dataset = datasets.ImageFolder(
            root=processed_dir,
            transform=trsfm
        )

        dataset_size = len(self.dataset)

        indices = list(range(dataset_size))
        split = int(np.floor(config["dataset_ratio"] * dataset_size))
        np.random.shuffle(indices)

        if config["train"]:
            indices = indices[:split]
        else:
            indices = indices[-split:]

        sampler = SubsetRandomSampler(indices)

        super(FERGDataLoader, self).__init__(
                    dataset=self.dataset,
                    batch_size=config["batch_size"],
                    sampler=sampler)
