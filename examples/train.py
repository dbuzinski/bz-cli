import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import GarmentClassifier
from bz.metrics import Accuracy, Precision
from bz import load_config

# Set seed
torch.manual_seed(42)
# Dataloader generator
g = torch.Generator()
g.manual_seed(2048)

hyperparameters = load_config()

# Define dataset
batch_size = hyperparameters.get("batch_size", 2048)
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
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, generator=g)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

# Metrics
metrics = [Accuracy(), Precision()]

# Define model
model = GarmentClassifier()

# Define loss
loss_fn = torch.nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.get("lr", 0.001))
