from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class MnistDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size=64):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        print(data_dir)

        self.dataset = MNIST(data_dir, 
                             train=False, 
                             transform=trsfm, 
                             download=True)

        super(MnistDataLoader, self).__init__(
                    dataset=self.dataset,
                    batch_size=batch_size,
                    shuffle=False)