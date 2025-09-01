# User Guide

This guide covers the essential aspects of using and extending the `bz` CLI tool.

## CLI Commands

The `bz` CLI provides a simple interface for training machine learning models.

### `bz init`

Initialize a new project with the required structure.

```bash
bz init
```

This creates:
* `train.py` - Your training script
* `bzconfig.json` - Configuration file

### `bz train`

Train your model using the current configuration.

```bash
# Basic training
bz train

# Train with custom config
bz train --config my_config.json

# Train for specific epochs
bz train --epochs 100

# Use specific device
bz train --device cuda
```

**Options:**
- `--config`: Path to configuration file (default: `bzconfig.json`)
- `--epochs`: Number of training epochs (overrides config)
- `--device`: Device to use (`cpu`, `cuda`, or `auto`)

## Configuration Schema

The `bzconfig.json` file controls all aspects of training. Here's the complete schema:

```json
{
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "device": "auto",
    "compile": true,
    "checkpoint_interval": 10,
    "validation_split": 0.2
  },
  "plugins": [
    "console_out",
    {
      "tensorboard": {
        "log_dir": "runs/experiment",
        "flush_interval": 10
      }
    },
    {
      "early_stopping": {
        "patience": 10,
        "monitor": "validation_loss",
        "min_delta": 0.001
      }
    }
  ],
  "metrics": [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
  ]
}
```

### Training Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epochs` | int | 1 | Number of training epochs |
| `batch_size` | int | 32 | Training batch size |
| `learning_rate` | float | 0.001 | Learning rate |
| `device` | string | "auto" | Device selection |
| `compile` | bool | true | Enable model compilation |
| `checkpoint_interval` | int | 0 | Save checkpoint every N epochs |
| `validation_split` | float | 0.2 | Validation data fraction |

### Plugin Configuration

Plugins can be specified as simple strings (using defaults) or as objects with custom configuration:

```json
{
  "plugins": [
    "console_out",                    // Simple string
    "early_stopping",                 // Simple string
    {                                 // Object with config
      "tensorboard": {
        "log_dir": "custom/path"
      }
    }
  ]
}
```

### Available Plugins

- **console_out**: Progress bars and training output
- **early_stopping**: Automatic training termination
- **tensorboard**: TensorBoard logging
- **wandb**: Weights & Biases integration
- **profiler**: Performance monitoring

## Training Script Structure

The `train.py` file is where you define your model, data, and training logic.

### Required Components

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# 1. Model definition
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 2. Loss function
loss_fn = nn.CrossEntropyLoss()

# 3. Optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# 4. Data loaders
training_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32)
```

### How It Works

1. **Discovery**: `bz` automatically discovers your `train.py` file
2. **Import**: The file is imported and variables are extracted
3. **Validation**: Required variables are checked for existence
4. **Training**: The training loop is executed with your components

### Variable Requirements

| Variable | Required | Type | Description |
|----------|----------|------|-------------|
| `model` | Yes | `nn.Module` | Your PyTorch model |
| `loss_fn` | Yes | Callable | Loss function |
| `optimizer` | Yes | Optimizer | PyTorch optimizer |
| `training_loader` | Yes | DataLoader | Training data |
| `validation_loader` | No | DataLoader | Validation data |
| `metrics` | No | List | Custom metrics to track |

### Example Complete Script

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=32)

# Model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

# Training components
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
```

## Custom Plugins

Plugins allow you to hook into the training lifecycle and add custom functionality.

### Plugin Base Class

```python
from bz.plugins import Plugin

class MyCustomPlugin(Plugin):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
    
    def start_training_session(self, context):
        """Called when training begins"""
        print(f"Starting training for {context.epochs} epochs")
    
    def end_epoch(self, context):
        """Called at the end of each epoch"""
        print(f"Epoch {context.epoch}: Loss = {context.training_loss:.4f}")
    
    def end_training_session(self, context):
        """Called when training completes"""
        print("Training completed successfully!")
```

### Available Hooks

| Hook | Description | Context Available |
|------|-------------|-------------------|
| `start_training_session` | Training begins | Epochs, device, config |
| `start_epoch` | Epoch begins | Current epoch number |
| `end_epoch` | Epoch ends | Metrics, loss, epoch info |
| `end_training_session` | Training ends | Final summary |
| `save_checkpoint` | Checkpoint saved | Checkpoint path |
| `load_checkpoint` | Checkpoint loaded | Checkpoint info |

### Plugin Configuration

```python
# In your train.py
from my_plugins import MyCustomPlugin

# Create plugin with config
my_plugin = MyCustomPlugin({
    "log_file": "training.log",
    "verbose": True
})

# Add to plugins list
plugins = [my_plugin]
```

### Plugin Discovery

Place your plugin files in a `plugins/` directory or install them as Python packages. The framework automatically discovers plugins in the `plugins` list.

## Custom Metrics

Create custom metrics to track domain-specific performance measures.

### Metric Base Class

```python
from bz.metrics import Metric
import torch

class CustomAccuracy(Metric):
    def __init__(self, threshold=0.5, name=None):
        super().__init__(name or "custom_accuracy")
        self.threshold = threshold
        self.correct = 0
        self.total = 0
    
    def reset(self):
        """Reset metric state between epochs"""
        self.correct = 0
        self.total = 0
    
    def update(self, predictions, targets):
        """Update metric with batch results"""
        # Apply threshold for binary classification
        pred_labels = (predictions > self.threshold).long()
        self.correct += (pred_labels == targets).sum().item()
        self.total += targets.size(0)
    
    def compute(self):
        """Compute final metric value"""
        return self.correct / self.total if self.total > 0 else 0.0
```

### Using Custom Metrics

```python
# In your train.py
from my_metrics import CustomAccuracy

# Create metrics list
metrics = [
    CustomAccuracy(threshold=0.5),
    CustomAccuracy(threshold=0.7, name="strict_accuracy")
]

# The framework automatically tracks these during training
```

### Metric Requirements

- **Inherit from `Metric`**: Use the base class for consistency
- **Implement required methods**: `reset()`, `update()`, `compute()`
- **Handle edge cases**: Check for division by zero, empty tensors
- **Use descriptive names**: Names appear in logs and TensorBoard

### Advanced Metrics

For complex metrics that require multiple passes or external data:

```python
class AdvancedMetric(Metric):
    def __init__(self):
        super().__init__("advanced_metric")
        self.predictions = []
        self.targets = []
    
    def update(self, predictions, targets):
        """Store predictions and targets for later computation"""
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        """Compute metric using all collected data"""
        # Complex computation here
        return complex_calculation(self.predictions, self.targets)
```

### Metric Registration

Custom metrics are automatically available when imported. No additional registration is required - just include them in your `metrics` list.
