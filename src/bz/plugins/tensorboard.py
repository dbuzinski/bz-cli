from torch.utils.tensorboard import SummaryWriter

from .plugin import Plugin


class TensorBoardPlugin(Plugin):
    def __init__(self, training_loader_len, log_dir):
        self.training_loader_len = training_loader_len
        self.writer = SummaryWriter(log_dir)

    def end_training_batch(self, context):
        avg_loss = context.training_loss_total / context.training_batch_count
        self.writer.add_scalar(
            "Loss/Train Step",
            avg_loss,
            context.epoch * self.training_loader_len + context.training_batch_count,
        )

    def end_training_loop(self, context):
        if context.training_batch_count:
            avg_loss = context.training_loss_total / context.training_batch_count
            self.writer.add_scalar("Loss/Train Epoch", avg_loss, context.epoch)
            for name, value in context.training_metrics.items():
                self.writer.add_scalar(f"Metric/Train/{name} Epoch", value, context.epoch)

    def end_validation_loop(self, context):
        if context.validation_batch_count > 0:
            avg_loss = context.validation_loss_total / context.validation_batch_count
            self.writer.add_scalar("Loss/Validation Epoch", avg_loss, context.epoch)
            for name, value in context.validation_metrics.items():
                self.writer.add_scalar(f"Metric/Validation/{name} Epoch", value, context.epoch)

    def end_epoch(self, context):
        self.writer.flush()

    def end_training_session(self, context):
        self.writer.close()

    @staticmethod
    def init(spec, log_dir="runs/experiment"):
        # You could also use a hash from spec to create a unique subdir
        # log_dir = os.path.join(log_dir, spec.model_name or "default")
        training_loader_len = len(spec.training_loader)
        return TensorBoardPlugin(training_loader_len, log_dir)
