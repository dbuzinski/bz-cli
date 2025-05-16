import torch


default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self):
        self.plugins = []
        self.metrics = []

    def add_plugin(self, plugin):
        self.plugins.append(plugin)

    def add_metric(self, metric):
        self.metrics.append(metric)

    def train(self, model, optimizer, loss_fn, training_loader, validation_loader=None, device=default_device, epochs=1, compile=True):
        # compile model
        if compile:
            model.compile()
        # set model to training mode and
        model.train()
        model.to(device)
        for epoch in range(epochs):
            # reset metrics for next epoch
            for metric in self.metrics:
                metric.reset()
            self.__run_stage("start_epoch", epoch)
            self.__run_stage("start_training_loop")
            for (batch_data, batch_labels) in training_loader:
                self.__run_stage("start_training_batch")
                optimizer.zero_grad(set_to_none=True)
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                preds = model(batch_data)
                loss = loss_fn(preds, batch_labels)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for metric in self.metrics:
                        metric.update(preds.detach().cpu(), batch_labels.detach().cpu())
                    for plugin in self.plugins:
                        if hasattr(plugin, "update_loss"):
                            plugin.update_loss(loss.item())
                self.__run_stage("end_training_batch")
            self.__run_stage("end_training_loop")
            if validation_loader:
                model.eval()
                with torch.no_grad():
                    self.__run_stage("start_validation_loop")
                    for (batch_inputs, batch_labels) in validation_loader:
                        self.__run_stage("start_validation_batch")
                        batch_inputs = batch_inputs.to(device)
                        batch_labels = batch_labels.to(device)
                        preds = model(batch_inputs)
                        self.__run_stage("end_validation_batch")
                    self.__run_stage("end_validation_loop")
            self.__run_stage("end_epoch", epoch)

    def __run_stage(self, stage_name, *args, **kwargs):
        for plugin in self.plugins:
            plugin.run_stage(stage_name, *args, **kwargs)



