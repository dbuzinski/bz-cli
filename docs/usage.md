# Usage Guide

This guide covers all aspects of using the `bz` CLI for machine learning model training.

## CLI Commands

### `bz train`

Train a model with the current configuration.

**Options:**
- `--epochs`: Number of training epochs
- `--checkpoint-interval`: Checkpoint save interval
- `--no-compile`: Disable model compilation
- `--config`: Path to configuration file
- `--device`: Device to use (auto/cpu/cuda)
- `--early-stopping-patience`: Enable early stopping with specified patience
- `--early-stopping-min-delta`: Minimum improvement for early stopping (default: 0.001)

**Examples:**
```bash
# Basic training
bz train

# Train for 10 epochs
bz train --epochs 10

# Use custom configuration
bz train --config my_config.json

# Train on CPU
bz train --device cpu

# Disable model compilation
bz train --no-compile

# Enable early stopping with 5 epochs patience
bz train --early-stopping-patience 5

# Enable early stopping with custom minimum delta
bz train --early-stopping-patience 10 --early-stopping-min-delta 0.0001
```

### `bz init`

Initialize a new project with templates.

**Options:**
- `--template`: Template to use (basic/advanced)

**Examples:**
```bash
# Initialize with basic template
bz init

# Initialize with advanced template
bz init --template advanced
```

### `bz validate`

Validate a trained model (coming soon).

**Options:**
- `--model-path`: Path to model checkpoint
- `--config`: Path to configuration file

## Configuration

The `bz` CLI uses a unified configuration system that supports both simple and advanced configurations.

### Configuration File Format

Create a `bz_config.json` file in your project directory:

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
      "config": {
        "update_interval": 1,
        "show_progress": true
      }
    },
    "tensorboard": {
      "enabled": true,
      "config": {
        "log_dir": "runs/experiment",
        "flush_interval": 10
      }
    }
  },
  "metrics": {
    "metrics": ["accuracy", "precision", "recall"],
    "custom_metrics": {
      "f1_score": {
        "type": "f1",
        "average": "macro"
      }
    }
  },
  "hyperparameters": {
    "lr": 0.001,
    "batch_size": 64,
    "optimizer": "adam",
    "weight_decay": 0.0001
  }
}
```

### Configuration Options

#### Training Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `epochs` | int | 1 | Number of training epochs |
| `batch_size` | int | 32 | Training batch size |
| `learning_rate` | float | 0.001 | Learning rate |
| `device` | string | "auto" | Device to use (auto/cpu/cuda) |
| `compile` | bool | true | Enable model compilation |
| `checkpoint_interval` | int | 0 | Checkpoint save interval |
| `early_stopping_patience` | int | null | Early stopping patience |
| `early_stopping_min_delta` | float | 0.001 | Minimum improvement for early stopping |

#### Plugin Configuration

Each plugin can be enabled/disabled and configured:

```json
{
  "plugins": {
    "plugin_name": {
      "enabled": true,
      "config": {
        "option1": "value1",
        "option2": "value2"
      }
    }
  }
}
```

#### Metrics Configuration

```json
{
  "metrics": {
    "metrics": ["accuracy", "precision", "recall"],
    "custom_metrics": {
      "metric_name": {
        "type": "metric_type",
        "parameters": {}
      }
    }
  }
}
```

### Configuration File Discovery

The CLI looks for configuration files in this order:

1. File specified by `--config` argument
2. File specified by `BZ_CONFIG` environment variable
3. `bz_config.json` in current directory
4. `config.json` in current directory
5. Default configuration

### Environment Variables

- `BZ_CONFIG`: Path to configuration file

## Plugin System

The `bz` CLI uses a plugin-based architecture that allows you to extend functionality by hooking into the training lifecycle.

### Built-in Plugins

#### ConsoleOutPlugin

Provides progress bars and training summaries.

**Configuration:**
```json
{
  "console_out": {
    "enabled": true,
    "config": {
      "update_interval": 1,
      "show_progress": true
    }
  }
}
```

#### TensorBoardPlugin

Logs training metrics to TensorBoard.

**Configuration:**
```json
{
  "tensorboard": {
    "enabled": true,
    "config": {
      "log_dir": "runs/experiment",
      "flush_interval": 10
    }
  }
}
```

#### EarlyStoppingPlugin

Automatically stops training when the monitored metric stops improving.

**Configuration:**
```json
{
  "early_stopping": {
    "enabled": true,
    "config": {
      "patience": 10,
      "min_delta": 0.001,
      "monitor": "validation_loss",
      "mode": "min",
      "strategy": "patience",
      "restore_best_weights": true,
      "verbose": true,
      "min_epochs": 0
    }
  }
}
```

**Parameters:**
- `patience`: Number of epochs to wait for improvement
- `min_delta`: Minimum change to qualify as improvement
- `monitor`: Metric to monitor ("validation_loss", "training_loss")
- `mode`: "min" (lower is better) or "max" (higher is better)
- `strategy`: "patience" (simple patience), "plateau" (plateau detection), "custom"
- `restore_best_weights`: Whether to restore best model weights
- `verbose`: Enable verbose logging
- `min_epochs`: Minimum epochs before early stopping can trigger

**CLI Usage:**
```bash
# Enable early stopping with 5 epochs patience
bz train --early-stopping-patience 5

# Enable early stopping with custom minimum delta
bz train --early-stopping-patience 10 --early-stopping-min-delta 0.0001
```

### Creating Custom Plugins

Create a custom plugin by inheriting from the `Plugin` base class:

```python
from bz.plugins import Plugin

class MyCustomPlugin(Plugin):
    def __init__(self, config=None):
        self.config = config or {}
    
    def start_training_session(self, context):
        """Called at the start of training."""
        print("Training started!")
        print(f"Training for {context.hyperparameters.get('epochs', 'unknown')} epochs")
    
    def start_epoch(self, context):
        """Called at the start of each epoch."""
        print(f"Starting epoch {context.epoch + 1}")
    
    def end_epoch(self, context):
        """Called at the end of each epoch."""
        if context.training_batch_count > 0:
            avg_loss = context.training_loss_total / context.training_batch_count
            print(f"Epoch {context.epoch} completed")
            print(f"Training loss: {avg_loss:.4f}")
            
            # Print metrics
            for name, value in context.training_metrics.items():
                print(f"Training {name}: {value:.4f}")
    
    def end_training_session(self, context):
        """Called at the end of training."""
        print("Training completed!")
```

### Plugin Lifecycle Hooks

| Hook | Description | Context Available |
|------|-------------|-------------------|
| `start_training_session` | Training session begins | Basic context |
| `load_checkpoint` | Checkpoint loaded | Checkpoint info |
| `start_epoch` | Epoch begins | Epoch number |
| `start_training_loop` | Training loop begins | Training setup |
| `start_training_batch` | Training batch begins | Batch info |
| `end_training_batch` | Training batch ends | Batch metrics |
| `end_training_loop` | Training loop ends | Epoch metrics |
| `start_validation_loop` | Validation begins | Validation setup |
| `start_validation_batch` | Validation batch begins | Batch info |
| `end_validation_batch` | Validation batch ends | Batch metrics |
| `end_validation_loop` | Validation ends | Validation metrics |
| `save_checkpoint` | Checkpoint saved | Checkpoint path |
| `end_epoch` | Epoch ends | Full epoch context |
| `end_training_session` | Training session ends | Final summary |

### Using Plugins

#### In Configuration

Enable plugins in your configuration file:

```json
{
  "plugins": {
    "console_out": {"enabled": true},
    "tensorboard": {"enabled": true},
    "my_custom_plugin": {
      "enabled": true,
      "config": {"option": "value"}
    }
  }
}
```

#### In Training Script

Load plugins manually in your training script:

```python
from bz import Trainer
from bz.plugins import ConsoleOutPlugin, TensorBoardPlugin
from my_plugins import MyCustomPlugin

trainer = Trainer()

# Add plugins
trainer.add_plugin(ConsoleOutPlugin.init(training_spec))
trainer.add_plugin(TensorBoardPlugin.init(training_spec, "runs/experiment"))
trainer.add_plugin(MyCustomPlugin({"option": "value"}))

# Train
trainer.train(model, optimizer, loss_fn, training_loader, ...)
```

## Metrics System

The `bz` CLI provides a comprehensive metrics system for tracking model performance.

### Built-in Metrics

The metrics system is organized into individual files for better maintainability:

#### Classification Metrics

- **Accuracy** (`accuracy.py`): Classification accuracy
- **Precision** (`precision.py`): Classification precision
- **Recall** (`recall.py`): Classification recall
- **F1Score** (`f1_score.py`): F1 score for classification
- **TopKAccuracy** (`top_k_accuracy.py`): Top-K accuracy for classification

#### Regression Metrics

- **MeanSquaredError** (`mean_squared_error.py`): MSE for regression
- **MeanAbsoluteError** (`mean_absolute_error.py`): MAE for regression

#### Base Classes

- **Metric** (`metric.py`): Abstract base class for all metrics

### Using Metrics

#### Direct Instantiation

```python
from bz.metrics import Accuracy, Precision, Recall, F1Score

metrics = [
    Accuracy(),
    Precision(average="macro"),
    Recall(),
    F1Score()
]
```

#### Using Metric Registry

```python
from bz.metrics import get_metric, list_available_metrics

# List all available metrics
print(list_available_metrics())
# Output: ['accuracy', 'precision', 'recall', 'f1_score', 'mse', 'mae', 'top5_accuracy']

# Create metrics using registry
metrics = [
    get_metric("accuracy"),
    get_metric("precision", average="macro"),
    get_metric("recall"),
    get_metric("f1_score"),
    get_metric("top5_accuracy")  # Creates TopKAccuracy with k=5
]
```

#### In Configuration

```json
{
  "metrics": {
    "metrics": ["accuracy", "precision", "recall", "f1_score"]
  }
}
```

### Creating Custom Metrics

Create custom metrics by inheriting from the `Metric` base class:

```python
from bz.metrics import Metric
import torch

class CustomAccuracy(Metric):
    def __init__(self, threshold=0.5, name=None):
        super().__init__(name)
        self.threshold = threshold
        self.correct = 0
        self.total = 0
    
    def reset(self):
        """Reset metric state."""
        self.correct = 0
        self.total = 0
    
    def update(self, preds, targets):
        """Update metric with new predictions and targets."""
        # Apply threshold for binary classification
        predictions = (preds > self.threshold).long()
        self.correct += (predictions == targets).sum().item()
        self.total += targets.size(0)
    
    def compute(self):
        """Compute final metric value."""
        return self.correct / self.total if self.total > 0 else 0.0
```

### Metric Configuration

Configure metrics in your configuration file:

```json
{
  "metrics": {
    "metrics": ["accuracy", "precision", "recall"],
    "custom_metrics": {
      "custom_accuracy": {
        "type": "CustomAccuracy",
        "threshold": 0.7
      }
    }
  }
}
```

## Training Script Structure

Your `train.py` file should define the following variables:

### Required Variables

- `model`: Your PyTorch model
- `loss_fn`: Loss function
- `optimizer`: Optimizer
- `training_loader`: Training data loader

### Optional Variables

- `validation_loader`: Validation data loader
- `metrics`: List of metrics to track
- `plugins`: List of custom plugins
- `hyperparameters`: Dictionary of hyperparameters

### Example Training Script

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import MyModel
from bz.metrics import Accuracy, Precision
from bz.config import get_config_manager

# Load configuration
config_manager = get_config_manager()
hyperparameters = config_manager.get_hyperparameters()
training_config = config_manager.get_training_config()

# Set seed for reproducibility
torch.manual_seed(42)
g = torch.Generator()
g.manual_seed(2048)

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
    shuffle=True, 
    generator=g
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
metrics = [Accuracy(), Precision()]

# Optional: Define custom hyperparameters
hyperparameters = {
    "lr": training_config["learning_rate"],
    "batch_size": training_config["batch_size"],
    "optimizer": "adam"
}
```

## Error Handling

The `bz` CLI includes robust error handling:

### Plugin Errors

If a plugin fails, the error is logged but training continues with other plugins.

### Training Errors

- **Batch errors**: Failed batches are skipped, training continues
- **Checkpoint errors**: Checkpoint failures are logged, training continues
- **Model compilation**: Compilation failures are logged, training continues without compilation

### Configuration Errors

- **Invalid values**: Configuration validation catches invalid values
- **Missing files**: Graceful fallback to defaults
- **JSON errors**: Clear error messages for malformed JSON

## Best Practices

### Configuration

1. **Use meaningful names**: Name your configuration files descriptively
2. **Version control**: Include configuration files in version control
3. **Environment-specific configs**: Use different configs for different environments
4. **Documentation**: Document custom configuration options

### Plugins

1. **Keep plugins focused**: Each plugin should have a single responsibility
2. **Handle errors gracefully**: Don't let plugin errors crash training
3. **Use configuration**: Make plugins configurable
4. **Test plugins**: Write tests for custom plugins

### Metrics

1. **Choose relevant metrics**: Select metrics appropriate for your task
2. **Monitor trends**: Watch for metric trends over time
3. **Custom metrics**: Create custom metrics for domain-specific needs
4. **Documentation**: Document custom metric behavior

### Training Scripts

1. **Reproducibility**: Set random seeds
2. **Configuration**: Use the configuration system
3. **Error handling**: Handle potential errors gracefully
4. **Documentation**: Document your training setup
