import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import GarmentClassifier


# Set seed
random.seed(42)
torch.manual_seed(42)

# Define dataset
batch_size = 2048
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
training_set = torchvision.datasets.FashionMNIST('./data',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)
validation_set = torchvision.datasets.FashionMNIST('./data',
                                                   train=False,
                                                   transform=transform,
                                                   download=True)
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)


# Define model
model = GarmentClassifier()

# Define loss
loss_fn = torch.nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
