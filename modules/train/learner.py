from modules import tqdm
from .tblog import TensorboardLog
from modules.models import GeneralModel
from modules.data import TransformerData
from .optim import BertAdam as Optimizer
from modules.criterions import GeneralCriterion
import torch
import os
from collections import defaultdict
from modules.data.utils import save_pkl


class Learner(object):

    @classmethod
    def create(cls):
        pass

    @classmethod
    def _create(cls, tensorboard_dir, data_args, model_args, optimizer_args, criterion_args,
                epochs=10, save_every=1, update_freq=1, device="cuda", target_metric="accuracy",
                checkpoint_dir="checkpoints"):
        data = TransformerData.create(**data_args)
        model = GeneralModel.create(**model_args)
        if device == "cuda":
            model = model.cuda()
        optimizer = Optimizer(model=model, **optimizer_args)
        criterion = GeneralCriterion.create(**criterion_args)
        tb_log = TensorboardLog(tensorboard_dir)
        return cls(
            data, model, optimizer, criterion, tb_log, epochs,
            save_every, update_freq, target_metric, checkpoint_dir)

    def __init__(self, data, model, optimizer, criterion, tb_log,
                 epochs=10, save_every=1, update_freq=1, device="cuda",
                 target_metric="accuracy", checkpoint_dir="checkpoints"):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.tb_log = tb_log
        self.epochs = epochs
        self.save_every = save_every
        self.update_freq = update_freq
        self.device = device
        self.splits = ["train", "valid", "test"]
        self.target_metric = target_metric
        self.checkpoint_dir = checkpoint_dir
        self._best_model_name = "best.cpt"
        self._last_model_name = "last.cpt"
        self._history_name = "history.pkl"
        self.history_path = os.path.join(self.checkpoint_dir, self._history_name)
        self.last_model_path = os.path.join(self.checkpoint_dir, self._last_model_name)
        self.best_model_path = os.path.join(self.checkpoint_dir, self._best_model_name)
        self.history = defaultdict(list)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def learn(self):
        best_metric = 0
        for epoch in range(self.epochs):
            epoch += 1
            for split in self.splits:
                if split in self.data.dataloaders:
                    epoch_metrics = self.step(self.data.dataloaders, epoch, split)
                    if split == "train" and epoch % self.save_every:
                        self.save_model(self.last_model_path)
                    if split == "valid" and best_metric < epoch_metrics[self.target_metric]:
                        self.save_model(self.best_model_path)
                    self.history[split].append(epoch_metrics)
                    save_pkl(self.history, self.history_path)

    def step(self, dl, epoch, tag):
        if tag == "train":
            self.model.train()
        else:
            self.model.eval()
        epoch_metrics = dict()
        pr = tqdm(dl, total=len(dl), leave=False)
        num_batches = epoch * len(dl)
        log_metrics = {}
        for idx, batch in enumerate(pr, 1):
            logits = self.model(**batch["net_input"])
            y_true = batch["target"]
            loss, metrics = self.criterion(y_true, logits)

            for key, val in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0
                epoch_metrics[key] += val
                if key == "loss":
                    log_metrics[key] = epoch_metrics[key] / idx
                elif key == "n_correct":
                    log_metrics[key] = val
                    log_metrics["accuracy"] = val / epoch_metrics["n_samples"]
                else:
                    log_metrics[key] = val

            if tag == "train":
                loss /= self.update_freq
                loss.backward()
                if idx % self.update_freq == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.model.zero_grad()
            self.tb_log(log_metrics, epoch, tag, num_batches + idx, pr)
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return epoch_metrics

    def save_model(self, path=None):
        path = path if path else os.path.join(self.checkpoint_dir, self.best_model_path)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path=None):
        path = path if path else os.path.join(self.checkpoint_dir, self.best_model_path)
        self.model.load_state_dict(torch.load(path))
