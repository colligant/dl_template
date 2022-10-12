from typing import Any

from src.datasets import DataModule
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class MNISTDataset(DataModule):

    def __init__(self, training, mnist_data_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if training:
            self.dataset = MNIST(mnist_data_path,
                                 train=True,
                                 download=True,
                                 transform=transforms.ToTensor())
        else:
            self.dataset = MNIST(mnist_data_path,
                                 train=False,
                                 download=True,
                                 transform=transforms.ToTensor())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    def collate_fn(self):
        """Return None for the default collate_fn."""
        return None



