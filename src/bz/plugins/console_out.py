import time
from tqdm import tqdm

from .plugin import Plugin


class ConsoleOutPlugin(Plugin):
    def __init__(self, training_data_len, validation_data_len=0, update_interval=1):
        self.training_data_len = training_data_len
        self.validation_data_len = validation_data_len
        self.update_interval = update_interval
        self._training_bar = None
        self._validation_bar = None
        self.training_start_time = None

    def start_training_session(self, context):
        self.training_start_time = time.time()

    def load_checkpoint(self, context):
        print()
        start_epoch = context.extra.get("start_epoch")
        checkpoint = context.extra.get("checkpoint_path")
        print(f"✓ Epoch {start_epoch} loaded from {checkpoint}")

    def start_epoch(self, context):
        print()
        print(f"Epoch {context.epoch + 1}:")

    def start_training_loop(self, context):
        self._training_bar = tqdm(
            range(self.training_data_len),
            desc="Training",
            bar_format="{desc}:   {percentage:3.0f}%|{bar:40}{r_bar}",
            unit="batch",
        )

    def end_training_batch(self, context):
        self._training_bar.update(1)
        if (self._training_bar.n + 1) % self.update_interval == 0:
            postfix_dict = context.training_metrics.copy()
            postfix_dict["loss"] = context.training_loss_total / context.training_batch_count
            self._training_bar.set_postfix(postfix_dict)

    def end_training_loop(self, context):
        if self._training_bar is not None:
            self._training_bar.close()

    def start_validation_loop(self, context):
        if self.validation_data_len:
            self._validation_bar = tqdm(
                range(self.validation_data_len),
                desc="Validation",
                bar_format="{desc}: {percentage:3.0f}%|{bar:40}{r_bar}",
                unit="batch",
            )

    def end_validation_batch(self, context):
        self._validation_bar.update(1)
        if (self._validation_bar.n + 1) % self.update_interval == 0:
            postfix_dict = context.validation_metrics.copy()
            postfix_dict["loss"] = context.validation_loss_total / context.validation_batch_count
            self._validation_bar.set_postfix(postfix_dict)

    def end_validation_loop(self, context):
        if self._validation_bar is not None:
            self._validation_bar.close()

    def save_checkpoint(self, context):
        checkpoint = context.extra.get("checkpoint_path")
        print(f"✓ Checkpoint saved to {checkpoint}")

    def end_training_session(self, context):
        # Epochs run formatting
        start_epoch = context.extra.get("start_epoch", 0)
        epochs_run = context.epoch - start_epoch
        resumed_string = f" (resumed from epoch {start_epoch})" if start_epoch else ""

        # Time formatting
        total_time = time.time() - self.training_start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        summary_item = {
            "Epochs Run": f"{epochs_run}{resumed_string}",
            "Total Time": time_str,
        }
        if context.training_batch_count:
            summary_item["Training Loss"] = f"{context.training_loss_total/context.training_batch_count:.4f}"
            for name, val in context.training_metrics.items():
                summary_item[f"Training {name}"] = f"{val:.4f}"
        if context.validation_batch_count:
            summary_item["Validation Loss"] = f"{context.validation_loss_total/context.validation_batch_count:.4f}"
            for name, val in context.validation_metrics.items():
                summary_item[f"Validation {name}"] = f"{val:.4f}"

        ljust_len = max(len(label) for label in summary_item.keys()) + 1
        total_len = max(ljust_len + len(val) for val in summary_item.values()) + 4

        # Print header
        print("\n" + "=" * total_len)
        print((total_len - 18) // 2 * " " + "Training Complete")
        print("=" * total_len)
        print()

        # Print status
        for label, val in summary_item.items():
            print(f" {label.ljust(ljust_len)}: {val}")

        # Print footer
        print("\n" + "=" * total_len)

    @staticmethod
    def init(spec):
        validation_data_len = 0
        if spec.validation_loader:
            validation_data_len = len(spec.validation_loader)
        return ConsoleOutPlugin(len(spec.training_loader), validation_data_len=validation_data_len)
