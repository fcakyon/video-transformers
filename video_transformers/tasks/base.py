class TaskMixin:
    def training_step(self, batch):
        raise NotImplementedError()

    def on_training_epoch_end(self):
        raise NotImplementedError()

    def validation_step(self, batch):
        raise NotImplementedError()

    def on_validation_epoch_end(self):
        raise NotImplementedError()
