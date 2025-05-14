import torch
from tqdm import tqdm


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
                self.__run_stage("end_training_batch")
            self.__run_stage("end_epoch", epoch)
            if validation_loader:
                model.eval()
                with torch.no_grad():
                    for (batch_inputs, batch_labels) in validation_loader:
                        self.__run_stage("start_validation_batch")
                        batch_inputs = batch_inputs.to(device)
                        batch_labels = batch_labels.to(device)
                        preds = model(batch_inputs)
                        self.__run_stage("end_validation_batch")

    def __run_stage(self, stage_name, *args, **kwargs):
        for plugin in self.plugins:
            plugin.run_stage(stage_name, *args, **kwargs)


class Plugin:
    def start_epoch(self, epoch):
        pass

    def end_epoch(self, epoch):
        pass

    def start_training_batch(self):
        pass

    def end_training_batch(self):
        pass

    def start_validation_batch(self):
        pass

    def end_validation_batch(self):
        pass

    def run_stage(self, stage, *args, **kwargs):
        getattr(self, stage)(*args, **kwargs)


class TrainingProgressPlugin(Plugin):
    def __init__(self, training_data_len, metrics=[]):
        super().__init__()
        self.training_data_len = training_data_len
        self.metrics = metrics
        self._disp_bar = None
        self._disp_iter = None

    def start_epoch(self, epoch):
        disp_bar = tqdm(range(self.training_data_len - 1))
        disp_bar.set_description(f"Epoch {epoch+1}:")
        self._disp_bar = disp_bar
        self._disp_iter = iter(disp_bar)

    def end_training_batch(self):
        if self._disp_iter and self._disp_bar:
            next(self._disp_iter, None)
            metrics_dict = {}
            for metric in self.metrics:
                metrics_dict.update({metric.name: metric.result()})
            self._disp_bar.set_postfix(metrics_dict)

    @staticmethod
    def init(spec):
        training_data_len = len(spec.training_loader)
        return TrainingProgressPlugin(training_data_len)
