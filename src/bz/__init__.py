import hashlib
import inspect
import json
import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Any


default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self):
        self.plugins = []

    def add_plugin(self, plugin):
        self.plugins.append(plugin)

    def train(self, model, optimizer, loss_fn, training_loader, validation_loader=None, device=default_device, epochs=1, compile=True, checkpoint_interval=0, metrics=[], hyperparameters={}):
        context = TrainingContext()
        context.hyperparameters = hyperparameters
        self.__run_stage("start_training_session", context)

        # Compute training signature and checkpoint directory
        training_signature = compute_training_signature(model, optimizer, loss_fn, context.hyperparameters)
        checkpoint_dir = os.path.join(".bz", "checkpoints", training_signature)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Try to resume from latest checkpoint
        latest_checkpoint_epoch = find_latest_checkpoint_epoch(checkpoint_dir)
        if latest_checkpoint_epoch:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{latest_checkpoint_epoch}.pth")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            loss_fn.load_state_dict(checkpoint["loss_fn_state"])
            training_loader.generator.set_state(checkpoint["generator_state"])
            context.epoch = latest_checkpoint_epoch
            context.extra["start_epoch"] = latest_checkpoint_epoch
            context.extra["checkpoint_path"] = checkpoint_path
            self.__run_stage("load_checkpoint", context)

        # compile model
        if compile:
            model.compile()
        os.makedirs(checkpoint_dir, exist_ok=True)
        # set model to training mode
        model.train()
        model.to(device)
        for epoch in range(context.epoch, epochs):
            self.__run_stage("start_epoch", context)
            self.__run_stage("start_training_loop", context)
            # reset metrics for training loop
            for metric in metrics:
                metric.reset()
                context.training_metrics[metric.name] = 0.
            context.training_loss_total = 0.
            context.training_batch_count = 0
            for (batch_data, batch_labels) in training_loader:
                self.__run_stage("start_training_batch", context)
                optimizer.zero_grad(set_to_none=True)
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                preds = model(batch_data)
                loss = loss_fn(preds, batch_labels)
                loss.backward()
                optimizer.step()
                # update loss and metrics
                with torch.no_grad():
                    for metric in metrics:
                        metric.update(preds.detach().cpu(), batch_labels.detach().cpu())
                        context.training_metrics[metric.name] = metric.compute()
                    context.training_loss_total += loss.item()
                    context.training_batch_count += 1
                self.__run_stage("end_training_batch", context)
            self.__run_stage("end_training_loop", context)
            if validation_loader:
                model.eval()
                with torch.no_grad():
                    # reset metrics for validation loop
                    for metric in metrics:
                        metric.reset()
                        context.validation_metrics[metric.name] = 0.
                    context.validation_loss_total = 0.
                    context.validation_batch_count = 0
                    self.__run_stage("start_validation_loop", context)
                    for (batch_inputs, batch_labels) in validation_loader:
                        self.__run_stage("start_validation_batch", context)
                        batch_inputs = batch_inputs.to(device)
                        batch_labels = batch_labels.to(device)
                        preds = model(batch_inputs)
                        loss = loss_fn(preds, batch_labels)
                        # update loss and metrics
                        for metric in metrics:
                            metric.update(preds.detach().cpu(), batch_labels.detach().cpu())
                            context.validation_metrics[metric.name] = metric.compute()
                        context.validation_loss_total += loss.item()
                        context.validation_batch_count += 1
                        self.__run_stage("end_validation_batch", context)
                    self.__run_stage("end_validation_loop", context)
            # Save the model checkpoint
            if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth")
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss_fn_state": loss_fn.state_dict(),
                    "generator_state": training_loader.generator.get_state()
                }, checkpoint_path)
                context.extra["checkpoint_path"] = checkpoint_path
                self.__run_stage("save_checkpoint", context)

            self.__run_stage("end_epoch", context)
            # Update context fields
            context.epoch += 1
        self.__run_stage("end_training_session", context)

    def __run_stage(self, stage_name, context):
        for plugin in self.plugins:
            plugin.run_stage(stage_name, context)


@dataclass(slots=True)
class TrainingContext:
    epoch: int = 0
    training_loss_total: float = 0.
    validation_loss_total: float = 0.
    training_batch_count: int = 0
    validation_batch_count: int = 0
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


def find_latest_checkpoint_epoch(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_epoch") and f.endswith(".pth")]
    epochs = []
    for f in files:
        try:
            epoch_num = int(f.replace("model_epoch", "").replace(".pth", ""))
            epochs.append(epoch_num)
        except ValueError:
            continue
    return max(epochs) if epochs else 0


def compute_training_signature(model, optimizer, loss_fn, config):
    payload = config.copy()
    payload["__model"] = type(model).__name__
    payload["__optimizer"] = type(optimizer).__name__
    payload["__optimizer_params"] = optimizer.param_groups
    payload["__loss_fn"] = type(loss_fn).__name__
    # __loss_fn_params might throw.
    # Leaving it like this because I want to know if someone encounters this situation.
    payload["__loss_fn_params"] = loss_fn.__dict__ or inspect.signature(loss_fn)
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def load_config(path=None):
    # Load file if provided a path
    if path:
        with open(path, 'r') as f:
            return json.load(f)

    # If BZ_CONFIG environment variable is set, load
    # the file it points to
    env_path = os.environ.get("BZ_CONFIG")
    if env_path and os.path.isfile(env_path):
        with open(env_path, 'r') as f:
            return json.load(f)

    # Otherwise default to config.json in current folder
    default_path = "config.json"
    if os.path.isfile(default_path):
        with open(default_path, 'r') as f:
            return json.load(f)

    return {}
