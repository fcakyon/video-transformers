class TaskMixin:
    def training_step(self, batch):
        raise NotImplementedError()

    def on_training_epoch_end(self):
        raise NotImplementedError()

    def validation_step(self, batch):
        raise NotImplementedError()

    def on_validation_epoch_end(self):
        raise NotImplementedError()

    @property
    def train_metrics(self):
        raise NotImplementedError()

    @property
    def val_metrics(self):
        raise NotImplementedError()

    @property
    def last_train_result(self):
        return None

    @property
    def last_val_result(self):
        return None
