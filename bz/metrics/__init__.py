class Metric:
    def reset(self):
        raise NotImplementedError

    def update(self, preds, targets):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__


class Accuracy(Metric):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, preds, targets):
        predicted_labels = preds.argmax(dim=1)
        self.correct += (predicted_labels == targets).sum().item()
        self.total += targets.size(0)

    def compute(self):
        return self.correct / self.total if self.total > 0 else 0.0


class Precision(Metric):
    def __init__(self, average='micro'):
        self.true_positives = 0
        self.predicted_positives = 0
        self.average = average

    def reset(self):
        self.true_positives = 0
        self.predicted_positives = 0

    def update(self, preds, targets):
        pred_labels = preds.argmax(dim=1)
        self.true_positives += ((pred_labels == 1) & (targets == 1)).sum().item()
        self.predicted_positives += (pred_labels == 1).sum().item()

    def compute(self):
        if self.predicted_positives == 0:
            return 0.0
        return self.true_positives / self.predicted_positives


class Recall(Metric):
    def __init__(self):
        self.true_positives = 0
        self.actual_positives = 0

    def reset(self):
        self.true_positives = 0
        self.actual_positives = 0

    def update(self, preds, targets):
        pred_labels = preds.argmax(dim=1)
        self.true_positives += ((pred_labels == 1) & (targets == 1)).sum().item()

        self.actual_positives += (targets == 1).sum().item()

    def compute(self):
        if self.actual_positives == 0:
            return 0.0
        return self.true_positives / self.actual_positives


class F1Score(Metric):
    def __init__(self):
        self.precision_metric = Precision()
        self.recall_metric = Recall()

    def reset(self):
        self.precision_metric.reset()
        self.recall_metric.reset()

    def update(self, preds, targets):
        self.precision_metric.update(preds, targets)
        self.recall_metric.update(preds, targets)

    def compute(self):
        p = self.precision_metric.compute()
        r = self.recall_metric.compute()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)


class MeanSquaredError(Metric):
    def __init__(self):
        self.sum_squared_error = 0.0
        self.total = 0

    def reset(self):
        self.sum_squared_error = 0.0
        self.total = 0

    def update(self, preds, targets):
        self.sum_squared_error += ((preds - targets) ** 2).sum().item()
        self.total += targets.numel()

    def compute(self):
        return self.sum_squared_error / self.total if self.total > 0 else 0.0
