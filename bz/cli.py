import argparse
import importlib.util
import os
import sys

from bz import Trainer
from bz.plugins import default_plugins


def main():
    parser = argparse.ArgumentParser(description="A tool to help train machine learning models")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    # Import train.py as module
    train_path = "train.py"
    train_dir = os.path.dirname(os.path.abspath(train_path))
    sys.path.insert(0, train_dir)

    spec = importlib.util.spec_from_file_location("bz_train", train_path)

    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except FileNotFoundError:
        print("The current directory does not contain 'train.py'")
        return
    except Exception as e:
        print(f"Error while loading training file: {e}")
        return
    sys.path.pop(0)

    # Load train module to training specification and train
    training_spec = TrainingSpecification.from_training_module(module)
    trainer = Trainer()
    trainer.plugins = _load_optional(module, "plugins", default_plugins(training_spec))
    metrics = _load_optional(module, "metrics", [])

    compile = not args.no_compile
    trainer.train(training_spec.model, training_spec.optimizer, training_spec.loss_fn, training_spec.training_loader, validation_loader=training_spec.validation_loader, epochs=args.epochs, compile=compile, checkpoint_interval=args.checkpoint_interval, metrics=metrics)


class TrainingSpecification:
    def __init__(self, model, loss_fn, optimizer, training_loader, validation_loader):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.training_loader = training_loader
        self.validation_loader = validation_loader

    @staticmethod
    def from_training_module(module):
        model = _load_required(module, "model")
        loss_fn = _load_required(module, "loss_fn")
        optimizer = _load_required(module, "optimizer")
        training_loader = _load_required(module, "training_loader")
        validation_loader = _load_optional(module, "validation_loader", None)
        return TrainingSpecification(model, loss_fn, optimizer,
                                     training_loader, validation_loader)


def _load_required(module, attr):
    try:
        val = getattr(module, attr)
        return val
    except AttributeError:
        raise Exception(f"{attr} must be specified")

def _load_optional(module, attr, default_val):
    try:
        val = getattr(module, attr)
        return val
    except AttributeError:
        return default_val

