import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import GarmentClassifier
from bz import get_config

# Set seed for reproducibility
torch.manual_seed(42)
g = torch.Generator()
g.manual_seed(2048)

# Load configuration
config = get_config()

# Define dataset and transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
training_set = torchvision.datasets.FashionMNIST(
    './data', train=True, transform=transform, download=True
)
validation_set = torchvision.datasets.FashionMNIST(
    './data', train=False, transform=transform, download=True
)

# Create data loaders
training_loader = DataLoader(
    training_set, 
    batch_size=config.hyperparameters.get("batch_size", 64), 
    shuffle=True, 
    generator=g
)
validation_loader = DataLoader(
    validation_set, 
    batch_size=config.hyperparameters.get("batch_size", 64), 
    shuffle=False
)

# Define model, loss function, and optimizer
model = GarmentClassifier()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config.hyperparameters.get("lr", 0.001)
)

# Set the Python objects in the config
config.model = model
config.loss_fn = loss_fn
config.optimizer = optimizer
config.training_loader = training_loader
config.validation_loader = validation_loader
config.training_set = training_set
config.validation_set = validation_set

# Define metrics (will be loaded from config by the framework)
metrics = []

# Define hyperparameters
hyperparameters = {
    "lr": config.hyperparameters.get("lr", 0.001),
    "batch_size": config.hyperparameters.get("batch_size", 64)
}
