from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Plugin:
    def start_epoch(self, epoch):
        pass

    def start_training_loop(self):
        pass

    def start_training_batch(self):
        pass

    def end_training_batch(self):
        pass

    def end_training_loop(self):
        pass

    def start_validation_loop(self):
        pass

    def start_validation_batch(self):
        pass

    def end_validation_batch(self):
        pass

    def end_validation_loop(self):
        pass

    def end_epoch(self, epoch):
        pass

    def run_stage(self, stage, *args, **kwargs):
        getattr(self, stage)(*args, **kwargs)


class ConsoleOutPlugin(Plugin):
    def __init__(self, training_data_len, training_batch_size, validation_data_len=0, validation_batch_size=0, update_interval=1):
        super().__init__()
        self.training_data_len = training_data_len
        self.validation_data_len = validation_data_len
        self.training_batch_size = training_batch_size
        self.validation_data_len = validation_data_len
        self.validation_batch_size = validation_batch_size
        self.update_interval = update_interval
        self._training_bar = None
        self._validation_bar = None
        self._total_loss = 0
        self._batch_count = 0

    def start_epoch(self, epoch):
        print(f"Epoch {epoch}")

    def start_training_loop(self):
        self._training_bar = tqdm(range(self.training_data_len), desc="Training")
        self._desc_bar = tqdm(total=0, position=1, bar_format="{desc}")

    def end_training_batch(self):
        self._training_bar.update(1)
        if (self._training_bar.n + 1) % self.update_interval == 0:
            self._training_bar.set_postfix_str(f"loss: {self._total_loss / self._batch_count:.4f}")

    def end_training_loop(self):
        if self._training_bar is not None:
            self._training_bar.close()

    def start_validation_loop(self):
        if self.validation_data_len:
            self._validation_bar = tqdm(range(self.validation_data_len), desc="Validation")

    def end_validation_batch(self):
        self._validation_bar.update(1)

    def end_validation_loop(self):
        if self._validation_bar is not None:
            self._validation_bar.close()

    def update_loss(self, loss):
        self._total_loss += loss
        self._batch_count += self.training_batch_size

    @staticmethod
    def init(spec):
        validation_data_len = 0
        validation_batch_size = 0
        if spec.validation_loader:
            validation_data_len = len(spec.validation_loader)
            validation_batch_size = spec.validation_loader.batch_size
        return ConsoleOutPlugin(len(spec.training_loader), spec.training_loader.batch_size, validation_data_len=validation_data_len, validation_batch_size=validation_batch_size)


class TensorBoardPlugin(Plugin):
    def __init__(self, log_dir="logs/fit"):
        super().__init__()
        self.writer = SummaryWriter(log_dir)
        self.epoch = 0
        self.training_step = 0
        self.validation_step = 0
        self._loss_total = 0.0
        self._batch_count = 0

    def start_epoch(self, epoch):
        self.epoch = epoch
        self._loss_total = 0.0
        self._batch_count = 0

    def update_loss(self, loss):
        self._loss_total += loss
        self._batch_count += 1
        self.writer.add_scalar("Loss/Training Step", loss, self.training_step)
        self.training_step += 1

    def end_training_loop(self):
        avg_loss = self._loss_total / max(1, self._batch_count)
        self.writer.add_scalar("Loss/Training Epoch", avg_loss, self.epoch)

    def start_validation_loop(self):
        self._val_preds = []
        self._val_labels = []

    def end_validation_batch(self):
        self.validation_step += 1

    def end_validation_loop(self):
        # You can optionally compute and log validation metrics here
        pass  # Extend if you want to add validation loss or metrics

    def end_epoch(self, epoch):
        # Optionally flush and save at the end of each epoch
        self.writer.flush()

    def close(self):
        self.writer.close()

    @staticmethod
    def init(spec, log_dir="runs/experiment"):
        return TensorBoardPlugin(log_dir)


def default_plugins(spec):
    return [ConsoleOutPlugin.init(spec)]
