"""
Command-line interface for bz CLI.
Provides commands for training, validation, and project initialization.
"""

import argparse
import importlib.util
import os
import sys

from bz import Trainer, get_config, instantiate_plugins
from bz.plugins import get_plugin_registry
from bz.metrics import list_available_metrics
from bz.health import run_health_check as run_health_check_func, print_health_report


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
        elif args.command == "list-plugins":
            run_list_plugins(args)
        elif args.command == "list-metrics":
            run_list_metrics(args)
        elif args.command == "health":
            run_health_check(args)
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
  bz list-plugins             # List available plugins
  bz list-metrics             # List available metrics
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

    # Optuna hyperparameter optimization
    train_parser.add_argument("--optimize", action="store_true", help="Enable hyperparameter optimization with Optuna")
    train_parser.add_argument("--n-trials", type=int, default=10, help="Number of optimization trials")
    train_parser.add_argument("--study-name", default="bz_optimization", help="Optuna study name")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a trained model")
    validate_parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    validate_parser.add_argument("--config", type=str, help="Path to configuration file")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize project structure")
    init_parser.add_argument(
        "--template", type=str, choices=["basic", "advanced"], default="basic", help="Template to use"
    )

    # List plugins command
    list_plugins_parser = subparsers.add_parser("list-plugins", help="List available plugins")
    list_plugins_parser.add_argument("--config", type=str, help="Path to configuration file")

    # List metrics command
    subparsers.add_parser("list-metrics", help="List available metrics")

    # Health check command
    health_parser = subparsers.add_parser("health", help="Run system health check")
    health_parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    return parser


def run_training(args):
    """Run the training process."""
    # Set CLI config path if specified
    if args.config:
        from bz import set_cli_config_path

        set_cli_config_path(args.config)

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

    # Get the config from the train.py module
    try:
        config = module.config
    except AttributeError:
        print("Error: train.py must define a 'config' variable")
        print("Make sure you call 'config = get_config()' and set up your model, loss_fn, etc.")
        sys.exit(1)

    # Instantiate plugins now that data loaders are set up
    instantiate_plugins(config)

    # Override config with command line arguments
    if args.epochs:
        config.epochs = args.epochs

    # Create trainer and load plugins
    trainer = Trainer()

    # Set up trainer with the objects from config
    trainer.plugins = config.plugins

    # Start training
    print("Starting training with configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Device: {config.device}")
    print(f"  Model: {type(config.model).__name__}")
    print(f"  Loss function: {type(config.loss_fn).__name__}")
    print(f"  Optimizer: {type(config.optimizer).__name__}")
    print(f"  Plugins: {[p.name for p in config.plugins]}")
    print(f"  Metrics: {[m.name for m in config.metrics]}")
    print()

    trainer.train(config)


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
        "bzconfig.json": get_config_template(),
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


def run_list_plugins(args):
    """List available plugins."""
    print("Available plugins:")

    # Get registered plugins
    registry = get_plugin_registry()
    registered_plugins = registry.list_plugins()

    for plugin_name in registered_plugins:
        print(f"  - {plugin_name}")

    # Show configured plugins if config is provided
    if args.config:
        try:
            # Temporarily set environment variable to load the specified config
            original_env = os.environ.get("BZ_CONFIG")
            os.environ["BZ_CONFIG"] = args.config
            config = get_config()
            if original_env:
                os.environ["BZ_CONFIG"] = original_env
            elif "BZ_CONFIG" in os.environ:
                del os.environ["BZ_CONFIG"]

            print("\nConfigured plugins:")
            for plugin in config.plugins:
                if isinstance(plugin, str):
                    print(f"  - {plugin}: enabled")
                elif isinstance(plugin, dict):
                    plugin_name = list(plugin.keys())[0]
                    config_data = plugin[plugin_name]
                    status = "enabled" if config_data.get("enabled", True) else "disabled"
                    print(f"  - {plugin_name}: {status}")

        except Exception as e:
            print(f"Error loading config: {e}")


def run_list_metrics(args):
    """List available metrics."""
    print("Available metrics:")
    metrics = list_available_metrics()
    for metric_name in sorted(metrics):
        print(f"  - {metric_name}")


def run_health_check(args):
    """Run system health check."""
    if args.json:
        import json

        health_data = run_health_check_func()
        print(json.dumps(health_data, indent=2))
    else:
        print_health_report()


# Template functions for project initialization
def get_train_template() -> str:
    return """import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import MyModel
from bz.metrics import Accuracy
from bz.config import get_config

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
model = MyModel(num_layers=config.hyperparameters.get("num_layers", 3))
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

# Define metrics
metrics = [Accuracy()]
"""


def get_config_template() -> str:
    return """{
  "epochs": 10,
  "hyperparameters": {
    "lr": 0.001,
    "batch_size": 64
  },
  "metrics": ["accuracy"],
  "plugins": ["console_out"]
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
