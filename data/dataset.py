import torchvision
import torchvision.transforms as transforms


class Dataset:
    def __init__(self, download=False, path="./data/Mnist/"):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.trainset = torchvision.datasets.MNIST(
            root=path, train=True, download=download, transform=transform
        )
        self.testset = torchvision.datasets.MNIST(
            root=path, train=False, download=download, transform=transform
        )
