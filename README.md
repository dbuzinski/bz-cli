# BZ CLI - Enhanced Machine Learning Training Framework

A powerful, extensible Python package for training machine learning models with a plugin-based architecture that makes the training loop extensible and customizable.

## 🚀 Features

### Core Features
- **Extensible Plugin System**: Hook into any part of the training lifecycle
- **Comprehensive Metrics System**: Built-in and custom metrics with easy registration
- **Robust Configuration Management**: Environment-based configs with validation
- **Advanced Error Handling**: Graceful plugin failures and recovery
- **Checkpoint Management**: Automatic checkpointing and resuming
- **Early Stopping**: Configurable early stopping with patience and minimum delta

### Plugin System
- **Plugin Registry**: Dynamic plugin discovery and loading
- **Plugin Dependencies**: Manage dependencies between plugins
- **Plugin Configuration**: JSON/YAML configuration for plugins
- **Built-in Plugins**:
  - **Console Output**: Formatted training progress with tqdm
  - **TensorBoard**: Integration with TensorBoard for visualization
  - **Weights & Biases**: Experiment tracking and model versioning

### Configuration Management
- **Environment Support**: Different configs for dev/staging/prod
- **Schema Validation**: Type-safe configuration with validation
- **Deep Merging**: Intelligent config merging with defaults
- **Plugin Dependencies**: Automatic dependency resolution

## 📦 Installation

```bash
pip install bz-cli
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
    "checkpoint_interval": 5,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.001
  },
  "plugins": {
    "console_out": {
      "enabled": true,
      "config": {},
      "dependencies": []
    },
    "tensorboard": {
      "enabled": true,
      "config": {
        "log_dir": "runs/experiment"
      },
      "dependencies": []
    },
    "wandb": {
      "enabled": false,
      "config": {
        "project_name": "my-experiment",
        "entity": "my-username"
      },
      "dependencies": []
    }
  },
  "metrics": {
    "metrics": ["accuracy", "precision", "recall"]
  },
  "hyperparameters": {
    "lr": 0.001,
    "batch_size": 64
  },
  "environment": "development"
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
hyperparameters = config_manager.get_hyperparameters()
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
bz train
```

## 🔧 Advanced Usage

### Environment-Based Configuration

Create environment-specific configs:

```bash
# Development
bz_config.development.json

# Staging  
bz_config.staging.json

# Production
bz_config.production.json
```

Set the environment:

```bash
export BZ_ENV=production
bz train
```

### Custom Plugins

Create your own plugin:

```python
from bz.plugins import Plugin, PluginContext

class MyCustomPlugin(Plugin):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.custom_value = config.get("custom_value", "default")
    
    def start_training_session(self, context: PluginContext) -> None:
        print(f"Starting training with custom value: {self.custom_value}")
    
    def end_epoch(self, context: PluginContext) -> None:
        print(f"Epoch {context.epoch} completed!")

# Register your plugin
from bz.plugins import register_plugin
register_plugin("my_custom", MyCustomPlugin, {"custom_value": "hello"})
```

### Custom Metrics

Create custom metrics:

```python
from bz.metrics import Metric
import torch
from torch import Tensor

class CustomMetric(Metric):
    def __init__(self, name: str = None):
        super().__init__(name)
        self.total_value = 0.0
        self.count = 0
    
    def reset(self) -> None:
        self.total_value = 0.0
        self.count = 0
    
    def update(self, preds: Tensor, targets: Tensor) -> None:
        # Your custom metric calculation
        self.total_value += torch.sum(preds).item()
        self.count += preds.numel()
    
    def compute(self) -> float:
        return self.total_value / self.count if self.count > 0 else 0.0
```

### Command Line Options

```bash
# Basic training
bz train

# Custom epochs and device
bz train --epochs 20 --device cuda

# Custom config file
bz train --config my_config.json

# Early stopping
bz train --early-stopping-patience 5 --early-stopping-min-delta 0.01

# List available plugins
bz list-plugins

# List available metrics
bz list-metrics
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
pytest tests/
```

The test suite covers:
- Plugin system functionality
- Configuration management
- CLI functionality
- Trainer components
- Metrics system
- Error handling

## 📚 API Reference

### Core Classes

#### `Trainer`
Main training class with plugin support and error handling.

#### `Plugin`
Base class for all plugins with lifecycle hooks.

#### `PluginContext`
Context object passed to plugins during training.

#### `ConfigManager`
Manages configuration loading, validation, and environment support.

### Plugin Lifecycle Hooks

- `start_training_session()` - Called at training start
- `load_checkpoint()` - Called when loading checkpoints
- `start_epoch()` - Called at each epoch start
- `start_training_loop()` - Called at training loop start
- `start_training_batch()` - Called at each training batch
- `end_training_batch()` - Called at each training batch end
- `end_training_loop()` - Called at training loop end
- `start_validation_loop()` - Called at validation loop start
- `start_validation_batch()` - Called at each validation batch
- `end_validation_batch()` - Called at each validation batch end
- `end_validation_loop()` - Called at validation loop end
- `save_checkpoint()` - Called when saving checkpoints
- `end_epoch()` - Called at each epoch end
- `end_training_session()` - Called at training end

## 🔌 Built-in Plugins

### Console Output Plugin
Provides formatted console output with progress bars.

**Configuration:**
```json
{
  "console_out": {
    "enabled": true,
    "config": {
      "update_interval": 1
    }
  }
}
```

### TensorBoard Plugin
Integrates with TensorBoard for training visualization.

**Configuration:**
```json
{
  "tensorboard": {
    "enabled": true,
    "config": {
      "log_dir": "runs/experiment"
    }
  }
}
```

### Weights & Biases Plugin
Integrates with Weights & Biases for experiment tracking.

**Configuration:**
```json
{
  "wandb": {
    "enabled": true,
    "config": {
      "project_name": "my-experiment",
      "entity": "my-username"
    }
  }
}
```

## 📊 Built-in Metrics

- `Accuracy` - Classification accuracy
- `Precision` - Classification precision
- `Recall` - Classification recall
- `F1Score` - F1 score
- `MeanSquaredError` - Regression MSE
- `MeanAbsoluteError` - Regression MAE
- `TopKAccuracy` - Top-K accuracy

## 🛠️ Development

### Project Structure

```
bz-cli/
├── src/bz/
│   ├── __init__.py          # Main trainer and core classes
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── metrics/            # Metrics system
│   │   └── __init__.py
│   └── plugins/            # Plugin system
│       ├── __init__.py
│       ├── plugin.py       # Base plugin class
│       ├── console_out.py  # Console output plugin
│       ├── tensorboard.py  # TensorBoard plugin
│       └── wandb.py        # WandB plugin
├── tests/                  # Test suite
├── examples/               # Example projects
└── docs/                   # Documentation
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Acknowledgments

- Built on PyTorch for deep learning capabilities
- Inspired by modern ML training frameworks
- Community contributions and feedback

