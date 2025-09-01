# BZ CLI - Enhanced Machine Learning Training Framework

A powerful, extensible Python package for training machine learning models with a plugin-based architecture that makes the training loop extensible and customizable.

## 🚀 Features

- **🔄 Extensible Plugin System**: Hook into any part of the training lifecycle
- **📊 Modular Metrics System**: Classification and regression metrics with easy registration
- **⚙️ Unified Configuration**: Type-safe configuration with validation and environment support
- **🛡️ Advanced Error Handling**: Graceful plugin failures and recovery
- **💾 Checkpoint Management**: Automatic checkpointing and resuming
- **🔧 Type Safety**: Full MyPy type checking support
- **📈 Monitoring**: Integration with TensorBoard, WandB, and other tools

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.7.0 or higher
- CUDA (optional, for GPU acceleration)

### Quick Install
```bash
pip install bz-cli
```

### Plugin Installation
```bash
# Install specific plugins
pip install bz-cli[optuna]      # Hyperparameter optimization
pip install bz-cli[wandb]       # Experiment tracking
pip install bz-cli[tensorboard] # Logging and visualization
pip install bz-cli[profiler]    # Performance monitoring

# Install all plugins
pip install bz-cli[all]
```

### Development Install
```bash
git clone https://github.com/dbuzinski/bz-cli.git
cd bz-cli
pip install -e ".[dev,all]"
```

## 🎯 Quick Start

### 1. Initialize a Project

```bash
bz init
```

This creates:
- `train.py` - Your training script
- `bz_config.json` - Configuration file
- `model.py` - Model definition
- `README.md` - Project documentation

### 2. Configure Your Training

Edit `bz_config.json`:

```json
{
  "training": {
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "device": "auto",
    "compile": true,
    "checkpoint_interval": 5
  },
  "plugins": {
    "console_out": {"enabled": true},
    "tensorboard": {
      "enabled": true,
      "config": {"log_dir": "runs/experiment"}
    },
    "early_stopping": {
      "enabled": true,
      "config": {
        "patience": 3,
        "min_delta": 0.001
      }
    }
  },
  "metrics": {
    "metrics": ["accuracy", "precision", "recall"]
  }
}
```

### 3. Define Your Training

Edit `train.py`:

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import MyModel
from bz.metrics import Accuracy, Precision, Recall
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
metrics = [Accuracy(), Precision(), Recall()]
```

### 4. Start Training

```bash
# Basic training
bz train
```

## 🔌 Plugin System

### Core Plugins (Included)
- **console_out**: Formatted console output with tqdm progress bars
- **early_stopping**: Advanced early stopping with multiple strategies

### Optional Plugins
- **optuna**: Hyperparameter optimization with Optuna
- **wandb**: Weights & Biases integration for experiment tracking
- **tensorboard**: TensorBoard logging and visualization
- **profiler**: Performance monitoring and profiling

### Plugin Discovery
Plugins are automatically discovered using Python entry points. The framework searches for plugins in:
1. Built-in plugins (console_out, early_stopping)
2. Installed packages with `bz.plugins` entry points
3. User-defined plugins in the current environment

### Plugin Configuration
Plugins are configured in `bzconfig.json`:

```json
{
  "plugins": [
    "console_out",
    {
      "tensorboard": {
        "enabled": true,
        "log_dir": "runs/experiment"
      }
    }
  ]
}
```

## 📊 Built-in Metrics

### Classification Metrics
- `Accuracy` - Classification accuracy
- `Precision` - Classification precision
- `Recall` - Classification recall
- `F1Score` - F1 score
- `TopKAccuracy` - Top-K accuracy

### Regression Metrics
- `MeanSquaredError` - Regression MSE
- `MeanAbsoluteError` - Regression MAE

## 🔧 Command Line Options

```bash
# Basic training
bz train

# Custom epochs and device
bz train --epochs 20 --device cuda

# Custom config file
bz train --config my_config.json



# Initialize project
bz init
```

## 📚 Documentation

📖 **Full Documentation**: [https://dbuzinski.github.io/bz-cli/](https://dbuzinski.github.io/bz-cli/)

The documentation includes:
- **Usage Guide**: Complete usage instructions and examples
- **API Reference**: Comprehensive API documentation
- **Examples**: Working examples for all features
- **Plugin System**: How to create and use plugins
- **Metrics System**: How to use and create custom metrics
- **Configuration**: Advanced configuration options

## 🧪 Testing

Run the comprehensive test suite:

```bash
uv run pytest
```

Lint and format code:

```bash
uv run ruff check src tests
uv run mypy src tests
uv run black src tests
```

## 🛠️ Development

### Project Structure

This project uses a monorepo structure with separate packages for plugins:

```
bz-cli/
├── src/
│   ├── bz/                 # Core framework
│   │   ├── __init__.py     # Main trainer and core classes
│   │   ├── cli.py          # Command-line interface
│   │   ├── config.py       # Configuration management
│   │   ├── health.py       # System health checks
│   │   ├── metrics/        # Modular metrics system
│   │   └── plugins/        # Core plugin system
│   ├── bz_optuna/          # Optuna plugin package
│   ├── bz_wandb/           # WandB plugin package
│   ├── bz_tensorboard/     # TensorBoard plugin package
│   └── bz_profiler/        # Profiler plugin package
├── tests/                  # Comprehensive test suite
├── examples/               # Example projects
│   ├── fashion-mnist/      # Image classification example
│   └── custom-plugin/      # Plugin development example
└── docs/                   # Documentation
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request