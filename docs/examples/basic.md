# Basic Example

This example demonstrates how to get started with the `bz` CLI using a simple image classification task.

## Prerequisites

Install the required dependencies:

```bash
pip install bz-cli torch torchvision
```

## Step 1: Initialize a Project

Create a new project directory and initialize it:

```bash
mkdir my-ml-project
cd my-ml-project
bz init
```

This creates the following files:
- `train.py` - Training script
- `model.py` - Model definition
- `bz_config.json` - Configuration file
- `README.md` - Project documentation

## Step 2: Define Your Model

Edit `model.py` to define your neural network:

```python
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## Step 3: Configure Training

Edit `bz_config.json` to configure your training:

```json
{
  "training": {
    "epochs": 5,
    "batch_size": 64,
    "learning_rate": 0.001,
    "device": "auto",
    "compile": true,
    "checkpoint_interval": 2
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
    },
    "early_stopping": {
      "enabled": true,
      "config": {
        "patience": 3,
        "min_delta": 0.001,
        "monitor": "validation_loss",
        "mode": "min"
      }
    }
  },
  "metrics": {
    "metrics": ["accuracy", "precision"]
  },
  "hyperparameters": {
    "lr": 0.001,
    "batch_size": 64
  }
}
```

## Step 4: Run Training

Start training your model:

```bash
# Basic training
bz train
```

You should see output similar to:

```
Starting training with configuration:
  Epochs: 5
  Batch size: 64
  Learning rate: 0.001
  Device: cuda
  Plugins: ['ConsoleOutPlugin', 'TensorBoardPlugin']
  Metrics: ['Accuracy', 'Precision']

Epoch 1:
Training:   100%|██████████| 938/938 [00:45<00:00, 20.67batch/s, loss=0.8234, accuracy=0.7123, precision=0.7234]
Validation: 100%|██████████| 157/157 [00:05<00:00, 28.45batch/s, loss=0.6543, accuracy=0.7891, precision=0.8012]

Epoch 2:
Training:   100%|██████████| 938/938 [00:44<00:00, 21.12batch/s, loss=0.6543, accuracy=0.7891, precision=0.8012]
Validation: 100%|██████████| 157/157 [00:05<00:00, 29.01batch/s, loss=0.5432, accuracy=0.8234, precision=0.8345]

...

========================================
        Training Complete
========================================

 Epochs Run     : 5
 Total Time     : 3m 45s
 Training Loss  : 0.2345
 Training Accuracy: 0.9234
 Training Precision: 0.9345
 Validation Loss: 0.2123
 Validation Accuracy: 0.9345
 Validation Precision: 0.9456

========================================
```

## Step 5: View Results

### TensorBoard Logs

View training progress in TensorBoard:

```bash
tensorboard --logdir runs/experiment
```

Open your browser to `http://localhost:6006` to see:
- Training and validation loss curves
- Accuracy and precision metrics
- Model graph visualization

### Checkpoints

Checkpoints are saved in `.bz/checkpoints/` directory. Each checkpoint contains:
- Model state
- Optimizer state
- Training epoch
- Metrics

## Step 6: Resume Training

To resume training from a checkpoint:

```bash
bz train --epochs 10
```

The CLI will automatically detect and load the latest checkpoint.

## Customization

### Add More Metrics

Edit your `train.py` to include additional metrics:

```python
from bz.metrics import Accuracy, Precision, Recall, F1Score

# Define metrics
metrics = [Accuracy(), Precision(), Recall(), F1Score()]
```

### Custom Configuration

Create a custom configuration file `my_config.json`:

```json
{
  "training": {
    "epochs": 10,
    "batch_size": 128,
    "learning_rate": 0.0005
  },
  "plugins": {
    "console_out": {"enabled": true},
    "tensorboard": {"enabled": true}
  },
  "metrics": {
    "metrics": ["accuracy", "precision", "recall", "f1_score"]
  }
}
```

Use it with:

```bash
bz train --config my_config.json
```

### Command Line Overrides

Override configuration values from the command line:

```bash
bz train --epochs 15 --device cpu --no-compile
```

## Next Steps

- Learn about [Custom Plugins](plugins.md)
- Check out the [API Reference](../reference.md) for detailed documentation
