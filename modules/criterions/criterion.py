from torch.nn import CrossEntropyLoss


class GeneralCriterion(object):

    @classmethod
    def create(cls, ignore_index=-100, task_type="cls"):
        loss = CrossEntropyLoss(ignore_index=ignore_index)
        return cls(loss, task_type)

    def __init__(self, loss, task_type):
        self.loss = loss
        self.task_type = task_type

    def __call__(self, y_pred, y_true):
        loss = sum([self.loss(y_pred[key], y_true[key]) for key in y_true])
        res = self.metrics(y_pred, y_true)
        res["loss"] = loss.data.cpu().tolist()
        return loss, res

    def metrics(self, y_pred, y_true):
        if self.task_type == "cls":
            y_pred = y_pred["cls"].argmax(-1).cpu().numpy()
            y_true = y_true["cls"].cpu().numpy()
            return {
                "n_correct": y_pred == y_true,
                "n_samples": len(y_true)
            }
        raise NotImplemented("Only classification task is implemented.")
