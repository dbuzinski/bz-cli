import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Plugin:
    def start_training_session(self, context):
        pass

    def start_epoch(self, context):
        pass

    def start_training_loop(self, context):
        pass

    def start_training_batch(self, context):
        pass

    def end_training_batch(self, context):
        pass

    def end_training_loop(self, context):
        pass

    def start_validation_loop(self, context):
        pass

    def start_validation_batch(self, context):
        pass

    def end_validation_batch(self, context):
        pass

    def end_validation_loop(self, context):
        pass

    def checkpoint(self, context):
        pass

    def end_epoch(self, context):
        pass

    def end_training_session(self, context):
        pass

    def run_stage(self, stage, context):
        getattr(self, stage)(context)


class ConsoleOutPlugin(Plugin):
    def __init__(self, training_data_len, validation_data_len=0, update_interval=1):
        super().__init__()
        self.training_data_len = training_data_len
        self.validation_data_len = validation_data_len
        self.update_interval = update_interval
        self._training_bar = None
        self._validation_bar = None
        self.training_start_time = None

    def start_training_session(self, context):
        self.training_start_time = time.time()

    def start_epoch(self, context):
        print(f"\nEpoch {context.epoch + 1}:")

    def start_training_loop(self, context):
        self._training_bar = tqdm(range(self.training_data_len), desc="Training", bar_format="{desc}:   {percentage:3.0f}%|{bar:40}{r_bar}", unit="batch")

    def end_training_batch(self, context):
        self._training_bar.update(1)
        if (self._training_bar.n + 1) % self.update_interval == 0:
            postfix_dict = context.training_metrics.copy()
            postfix_dict["loss"] = context.training_loss_total/ context.training_batch_count
            self._training_bar.set_postfix(postfix_dict)

    def end_training_loop(self, context):
        if self._training_bar is not None:
            self._training_bar.close()

    def start_validation_loop(self, context):
        if self.validation_data_len:
            self._validation_bar = tqdm(range(self.validation_data_len), desc="Validation", bar_format="{desc}: {percentage:3.0f}%|{bar:40}{r_bar}", unit="batch")

    def end_validation_batch(self, context):
        self._validation_bar.update(1)
        if (self._validation_bar.n + 1) % self.update_interval == 0:
            postfix_dict = context.validation_metrics.copy()
            postfix_dict["loss"] = context.validation_loss_total/ context.validation_batch_count
            self._validation_bar.set_postfix(postfix_dict)


    def end_validation_loop(self, context):
        if self._validation_bar is not None:
            self._validation_bar.close()

    def end_training_session(self, context):
        total_time = time.time() - self.training_start_time
        print("\n" + "=" * 26)
        print("   Training Complete")
        print("=" * 26)
        print(f"\nTotal Epochs\t\t: {context.epoch}")
        print(f"Total Time\t\t: {int(total_time // 60)}m {int(total_time % 60)}s")

        print(f"Training Loss\t\t: {context.training_loss_total/context.training_batch_count:.4f}")
        for name, val in context.training_metrics.items():
            print(f"Training {name}\t: {val:.4f}")
        print(f"Validation Loss\t\t: {context.validation_loss_total/context.validation_batch_count:.4f}")
        for name, val in context.validation_metrics.items():
            print(f"Validation {name}\t: {val:.4f}")
        print("\n" + "=" * 26)

    @staticmethod
    def init(spec):
        validation_data_len = 0
        validation_batch_size = 0
        if spec.validation_loader:
            validation_data_len = len(spec.validation_loader)
            validation_batch_size = spec.validation_loader.batch_size
        return ConsoleOutPlugin(len(spec.training_loader), validation_data_len=validation_data_len)


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
