# bz CLI

A powerful and extensible tool for reproducible and efficient training of AI models with a plugin-based architecture.

## 🚀 Features

- **🔄 Extensible Plugin System**: Hook into training lifecycle with custom plugins
- **📊 Rich Metrics**: Modular metrics system with classification and regression metrics
- **⚙️ Unified Configuration**: Type-safe configuration system with validation
- **🛡️ Error Handling**: Robust error handling and recovery
- **📈 Monitoring**: Integration with TensorBoard, WandB, and other tools
- **💾 Checkpointing**: Automatic model checkpointing and resumption
- **🎯 CLI Commands**: Intuitive command-line interface
- **🔧 Type Safety**: Full MyPy type checking support

## 📦 Installation

```bash
pip install bz-cli
```

## 🏃‍♂️ Quick Start

### 1. Initialize a Project

```bash
bz init
```

This creates a basic project structure with:
- `train.py` - Training script template
- `model.py` - Model definition template  
- `bz_config.json` - Configuration file
- `README.md` - Project documentation

### 2. Train a Model

```bash
bz train
```

### 3. Train with Custom Settings

```bash
# Train with custom epochs
bz train --epochs 10 --config my_config.json
```

## 📋 Basic Usage

Create a `train.py` file in your project directory:

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import MyModel
from bz.metrics import Accuracy
from bz.config import get_config_manager

# Load configuration
config_manager = get_config_manager()
training_config = config_manager.get_training_config()

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
    batch_size=training_config["batch_size"], 
    shuffle=True
)
validation_loader = DataLoader(
    validation_set, 
    batch_size=training_config["batch_size"], 
    shuffle=False
)

# Define model, loss function, and optimizer
model = MyModel()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=training_config["learning_rate"]
)

# Define metrics
metrics = [Accuracy()]
```

## 🎯 CLI Commands

| Command | Description |
|---------|-------------|
| `bz train` | Train a model with current configuration |
| `bz init` | Initialize a new project with templates |
| `bz validate` | Validate a trained model (coming soon) |

## 🔧 Configuration

The `bz` CLI uses a unified configuration system. See the [Configuration Guide](usage.md#configuration) for details.

## 🔌 Plugins

Extend functionality with plugins. See the [Plugin System](usage.md#plugin-system) for details.

## 📊 Metrics

Track model performance with modular metrics system including classification and regression metrics. See the [Metrics System](usage.md#metrics-system) for details.

## 📚 Examples

Check out the `examples/` directory for complete working examples:

- **Fashion MNIST**: Basic image classification
- **Configuration examples**: Different config formats  
- **Plugin examples**: Custom plugin implementations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

