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

    def train(self, model, optimizer, loss_fn, training_loader, validation_loader=None, device=default_device, epochs=1, compile=True, checkpoint_interval=0, metrics=[]):
        context = TrainingContext()
        self.__run_stage("start_training_session", context)
        # compile model
        if compile:
            model.compile()
        checkpoint_dir = ".bz/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        # set model to training mode
        model.train()
        model.to(device)
        for epoch in range(epochs):
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
                torch.save(model.state_dict(), checkpoint_path)
                print(f"checkpoint saved to {checkpoint_path}")
                self.__run_stage("checkpoint", context)
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
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
