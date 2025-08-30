# bz CLI

A powerful and extensible tool for reproducible and efficient training of AI models with a plugin-based architecture.

## Features

- **🔄 Extensible Plugin System**: Hook into training lifecycle with custom plugins
- **📊 Rich Metrics**: Built-in and custom metrics for model evaluation
- **⚙️ Unified Configuration**: Type-safe configuration system with validation
- **🛡️ Error Handling**: Robust error handling and recovery
- **📈 Monitoring**: Integration with TensorBoard, WandB, and other tools
- **💾 Checkpointing**: Automatic model checkpointing and resumption
- **🚀 Early Stopping**: Built-in early stopping with configurable patience
- **🎯 CLI Commands**: Intuitive command-line interface

## Quick Start

### Installation

```bash
pip install bz-cli
```

### Basic Usage

1. **Initialize a project**:
   ```bash
   bz init
   ```

2. **Train a model**:
   ```bash
   bz train
   ```

3. **Train with custom config**:
   ```bash
   bz train --config my_config.json --epochs 10
   ```

## Configuration

The `bz` CLI uses a unified configuration system. Create a `bz_config.json` file:

```json
{
  "training": {
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "device": "auto",
    "compile": true,
    "checkpoint_interval": 5,
    "early_stopping_patience": 3
  },
  "plugins": {
    "console_out": {
      "enabled": true,
      "config": {}
    },
    "tensorboard": {
      "enabled": true,
      "config": {
        "log_dir": "runs/experiment"
      }
    }
  },
  "metrics": {
    "metrics": ["accuracy", "precision", "recall"]
  },
  "hyperparameters": {
    "lr": 0.001,
    "batch_size": 64
  }
}
```

## Plugin System

### Built-in Plugins

- **ConsoleOutPlugin**: Progress bars and training summaries
- **TensorBoardPlugin**: Logging to TensorBoard
- **WandBPlugin**: Integration with Weights & Biases (coming soon)

### Creating Custom Plugins

```python
from bz.plugins import Plugin

class MyCustomPlugin(Plugin):
    def start_training_session(self, context):
        print("Training started!")
    
    def end_epoch(self, context):
        print(f"Epoch {context.epoch} completed")
        print(f"Training loss: {context.training_loss_total / context.training_batch_count}")
```

## Metrics System

### Built-in Metrics

- **Accuracy**: Classification accuracy
- **Precision**: Classification precision
- **Recall**: Classification recall
- **F1Score**: F1 score for classification
- **MeanSquaredError**: MSE for regression
- **MeanAbsoluteError**: MAE for regression
- **TopKAccuracy**: Top-K accuracy for classification

### Using Metrics

```python
from bz.metrics import Accuracy, Precision, get_metric

# Direct instantiation
metrics = [Accuracy(), Precision()]

# Using registry
accuracy = get_metric("accuracy")
precision = get_metric("precision", average="macro")
```

## Training Script

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
metrics = [Accuracy()]
```

## CLI Commands

### `bz train`

Train a model with the current configuration.

**Options:**
- `--epochs`: Number of training epochs
- `--checkpoint-interval`: Checkpoint save interval
- `--no-compile`: Disable model compilation
- `--config`: Path to configuration file
- `--device`: Device to use (auto/cpu/cuda)

### `bz init`

Initialize a new project with templates.

**Options:**
- `--template`: Template to use (basic/advanced)

### `bz validate`

Validate a trained model (coming soon).

## Examples

Check out the `examples/` directory for complete working examples:

- **Fashion MNIST**: Basic image classification
- **Configuration examples**: Different config formats
- **Plugin examples**: Custom plugin implementations

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

