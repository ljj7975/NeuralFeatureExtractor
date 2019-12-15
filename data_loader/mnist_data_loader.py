from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class MNISTDataLoader(DataLoader):
    def __init__(self, config):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.dataset = MNIST(config["data_dir"], 
                             train=config["train"],
                             transform=trsfm,
                             download=True)

        super(MNISTDataLoader, self).__init__(
                    dataset=self.dataset,
                    batch_size=config["batch_size"],
                    shuffle=False)