import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass

from bz import Trainer
from bz.config import get_config_manager, ConfigManager


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            run_training(args)
        elif args.command == "validate":
            run_validation(args)
        elif args.command == "init":
            run_init(args)
        else:
            parser.print_help()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="A tool to help train machine learning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bz train                    # Train with default config
  bz train --epochs 10        # Train for 10 epochs
  bz train --config my_config.json  # Use custom config
  bz validate                 # Validate model
  bz init                     # Initialize project structure
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--checkpoint-interval", type=int, help="Checkpoint save interval")
    train_parser.add_argument("--no-compile", action="store_true", help="Disable model compilation")
    train_parser.add_argument("--config", type=str, help="Path to configuration file")
    train_parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], help="Device to use")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a trained model")
    validate_parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    validate_parser.add_argument("--config", type=str, help="Path to configuration file")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize project structure")
    init_parser.add_argument(
        "--template", type=str, choices=["basic", "advanced"], default="basic", help="Template to use"
    )

    return parser


def run_training(args):
    """Run the training process."""
    # Load configuration
    config_manager = get_config_manager()
    if args.config:
        config_manager = ConfigManager(config_path=args.config)

    training_config = config_manager.get_training_config()

    # Override config with command line arguments
    if args.epochs:
        training_config["epochs"] = args.epochs
    if args.checkpoint_interval is not None:
        training_config["checkpoint_interval"] = args.checkpoint_interval
    if args.no_compile:
        training_config["compile"] = False
    if args.device:
        training_config["device"] = args.device

    # Import train.py as module
    train_path = "train.py"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found in current directory")
        print("Please run this command from a directory containing train.py")
        sys.exit(1)

    train_dir = os.path.dirname(os.path.abspath(train_path))
    sys.path.insert(0, train_dir)

    try:
        spec = importlib.util.spec_from_file_location("bz_train", train_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except FileNotFoundError:
        print(f"Error: {train_path} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {train_path}: {e}")
        sys.exit(1)
    finally:
        sys.path.pop(0)

    # Load training specification
    try:
        training_spec = load_training_spec(module)
    except Exception as e:
        print(f"Error loading training specification: {e}")
        sys.exit(1)

    # Create trainer and load plugins
    trainer = Trainer()

    # Load plugins based on configuration
    plugins = load_plugins_from_config(config_manager, training_spec)
    trainer.plugins = plugins

    # Load metrics
    metrics = load_metrics_from_config(config_manager, module)

    # Start training
    print("Starting training with configuration:")
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Device: {training_config['device']}")
    print(f"  Plugins: {[p.__class__.__name__ for p in plugins]}")
    print(f"  Metrics: {[m.name for m in metrics]}")
    print()

    trainer.train(
        training_spec.model,
        training_spec.optimizer,
        training_spec.loss_fn,
        training_spec.training_loader,
        validation_loader=training_spec.validation_loader,
        epochs=training_config["epochs"],
        compile=training_config["compile"],
        checkpoint_interval=training_config["checkpoint_interval"],
        metrics=metrics,
        hyperparameters=config_manager.get_hyperparameters(),
    )


def run_validation(args):
    """Run model validation."""
    print("Validation functionality not yet implemented")
    # TODO: Implement validation logic


def run_init(args):
    """Initialize project structure."""
    print(f"Initializing project with {args.template} template...")

    # Create basic project structure
    files_to_create = {
        "train.py": get_train_template(),
        "bz_config.json": get_config_template(),
        "model.py": get_model_template(),
        "README.md": get_readme_template(),
    }

    for filename, content in files_to_create.items():
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write(content)
            print(f"✓ Created {filename}")
        else:
            print(f"⚠ {filename} already exists, skipping")

    print("\nProject initialized successfully!")
    print("Next steps:")
    print("1. Edit train.py to define your model, data, and training setup")
    print("2. Configure bz_config.json with your desired settings")
    print("3. Run 'bz train' to start training")


def load_plugins_from_config(config_manager: ConfigManager, training_spec) -> list:
    """Load plugins based on configuration."""
    plugins = []

    # Load enabled plugins
    if config_manager.is_plugin_enabled("console_out"):
        from bz.plugins import ConsoleOutPlugin

        plugin_config = config_manager.get_plugin_config("console_out")
        plugins.append(ConsoleOutPlugin.init(training_spec))

    if config_manager.is_plugin_enabled("tensorboard"):
        from bz.plugins import TensorBoardPlugin

        plugin_config = config_manager.get_plugin_config("tensorboard")
        if plugin_config:
            log_dir = plugin_config.get("config", {}).get("log_dir", "runs/experiment")
        else:
            log_dir = "runs/experiment"
        plugins.append(TensorBoardPlugin.init(training_spec, log_dir))

    return plugins


def load_metrics_from_config(config_manager: ConfigManager, module) -> list:
    """Load metrics based on configuration."""
    metrics = []
    metrics_config = config_manager.get_metrics_config()

    # Load built-in metrics
    metric_map = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1Score",
        "mse": "MeanSquaredError",
    }

    for metric_name in metrics_config.get("metrics", []):
        if metric_name in metric_map:
            import bz.metrics as metrics_module

            metric_class = getattr(metrics_module, metric_map[metric_name], None)
            if metric_class:
                metrics.append(metric_class())

    # Load custom metrics from module
    try:
        module_metrics = getattr(module, "metrics", [])
        metrics.extend(module_metrics)
    except AttributeError:
        pass

    return metrics


@dataclass
class TrainingSpecification:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        training_loader,
        validation_loader,
        hyperparameters,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.hyperparameters = hyperparameters


def load_training_spec(module):
    model = _load_required(module, "model")
    loss_fn = _load_required(module, "loss_fn")
    optimizer = _load_required(module, "optimizer")
    training_loader = _load_required(module, "training_loader")
    validation_loader = _load_optional(module, "validation_loader", None)
    hyperparameters = _load_optional(module, "hyperparameters", {})
    return TrainingSpecification(model, loss_fn, optimizer, training_loader, validation_loader, hyperparameters)


def _load_required(module, attr):
    try:
        val = getattr(module, attr)
        return val
    except AttributeError:
        raise Exception(f"{attr} must be specified in train.py")


def _load_optional(module, attr, default_val):
    try:
        val = getattr(module, attr)
        return val
    except AttributeError:
        return default_val


# Template functions for project initialization
def get_train_template() -> str:
    return """import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import MyModel
from bz.metrics import Accuracy
from bz.config import get_config_manager

# Set seed for reproducibility
torch.manual_seed(42)
g = torch.Generator()
g.manual_seed(2048)

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
metrics = [Accuracy()]
"""


def get_config_template() -> str:
    return """{
  "training": {
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "device": "auto",
    "compile": true,
    "checkpoint_interval": 5
  },
  "plugins": {
    "console_out": {
      "enabled": true,
      "config": {}
    },
    "tensorboard": {
      "enabled": false,
      "config": {
        "log_dir": "runs/experiment"
      }
    }
  },
  "metrics": {
    "metrics": ["accuracy"]
  },
  "hyperparameters": {
    "lr": 0.001,
    "batch_size": 64
  }
}
"""


def get_model_template() -> str:
    return """import torch.nn as nn
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
"""


def get_readme_template() -> str:
    return """# My ML Project

This project uses the bz CLI for training machine learning models.

## Setup

1. Install dependencies:
   ```bash
   pip install torch torchvision bz-cli
   ```

2. Run training:
   ```bash
   bz train
   ```

3. View logs:
   ```bash
   tensorboard --logdir runs/experiment
   ```

## Configuration

Edit `bz_config.json` to customize training parameters, enable/disable plugins, and configure metrics.
"""
