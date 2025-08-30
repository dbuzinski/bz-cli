class Plugin:
    def start_training_session(self, context):
        pass

    def load_checkpoint(self, context):
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

    def save_checkpoint(self, context):
        pass

    def end_epoch(self, context):
        pass

    def end_training_session(self, context):
        pass

    def run_stage(self, stage, context):
        getattr(self, stage)(context)
