from modules import tqdm
from modules.utils import if_none
from .tblog import TensorboardLog
from modules.models import GeneralModel
from modules.data import TransformerData
from .optim import BertAdam as Optimizer
from modules.criterions import GeneralCriterion
import torch
import os
from collections import defaultdict
from modules.data.utils import save_pkl
from transformers import tokenization_auto


class Learner(object):

    @classmethod
    def create(
            cls,
            tensorboard_dir,
            # Model args
            model_name, model_type,
            # Criterion
            ignore_index=-100,
            #  Data args
            train_df_path=None, valid_df_path=None, test_df_path=None, dictionaries=None,
            dictionaries_path=None, tokenizer_cls=tokenization_auto.AutoTokenizer, tokenizer_args=None,
            max_tokens=512, clear_cache=False, online=False, shuffle=True, cache_dir="./",
            pad_idx=0, markup="IO", batch_size=32,
            # Optimizer
            lr=2e-5, warmup=0.1, t_total=None, schedule='warmup_linear',
            b1=0.8, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0,
            epochs=10, save_every=1, update_freq=1, device="cuda", target_metric="accuracy",
            checkpoint_dir="checkpoints"
    ):
        data_args = {
            "model_name": model_name,
            "train_df_path": train_df_path,
            "valid_df_path": valid_df_path,
            "test_df_path": test_df_path,
            "dictionaries": dictionaries,
            "dictionaries_path": dictionaries_path,
            "tokenizer_cls": tokenizer_cls,
            "tokenizer_args": tokenizer_args,
            "max_tokens": max_tokens,
            "clear_cache": clear_cache,
            "online": online,
            "shuffle": shuffle,
            "cache_dir": cache_dir,
            "pad_idx": pad_idx,
            "markup": markup,
            "batch_size": batch_size
        }
        model_args = {
            "model_name": model_name,
            "model_type": model_type
        }
        optimizer_args = {
            "lr": lr,
            "warmup": warmup,
            "t_total": t_total,
            "schedule": schedule,
            "b1": b1,
            "b2": b2,
            "e": e,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm
        }
        criterion_args = {
            "ignore_index": ignore_index,
            "model_type": model_type
        }
        return cls._create(
            tensorboard_dir, data_args, model_args, optimizer_args, criterion_args,
            epochs=epochs, save_every=save_every, update_freq=update_freq, device=device,
            target_metric=target_metric, checkpoint_dir=checkpoint_dir
        )

    @classmethod
    def _create(cls, tensorboard_dir, data_args, model_args, optimizer_args, criterion_args,
                epochs=10, save_every=1, update_freq=1, device="cuda", target_metric="accuracy",
                checkpoint_dir="checkpoints"):
        data = TransformerData.create(**data_args)
        model = GeneralModel.create(**model_args)
        if device == "cuda":
            model = model.cuda()
        optimizer_args["t_total"] = if_none(optimizer_args["t_total"], epochs * len(data.dataloaders["train"]))
        optimizer = Optimizer(model=model, **optimizer_args)
        criterion = GeneralCriterion.create(**criterion_args)
        tb_log = TensorboardLog(tensorboard_dir)
        return cls(
            data=data, model=model, optimizer=optimizer, criterion=criterion, tb_log=tb_log, epochs=epochs,
            save_every=save_every, update_freq=update_freq, device=device,
            target_metric=target_metric, checkpoint_dir=checkpoint_dir)

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
                if self.data.dataloaders.get(split) is not None:
                    epoch_metrics = self.step(self.data.dataloaders[split], epoch, split)
                    if split == "train" and epoch % self.save_every == 0:
                        self.save_model(self.last_model_path)
                    if split == "valid" and best_metric < epoch_metrics.get(self.target_metric, 0):
                        self.save_model(self.best_model_path)
                    self.history[split].append(epoch_metrics)
                    save_pkl(self.history, self.history_path)

    def step(self, dl, epoch, tag):
        if tag == "train":
            self.model.train()
        else:
            self.model.eval()
        self.optimizer.zero_grad()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        epoch_metrics = {"n_samples": 0}
        len_dl = len(dl)
        pr = tqdm(dl, total=len_dl, leave=False)
        num_batches = (epoch - 1) * len_dl
        log_metrics = {}
        for idx, batch in enumerate(pr, 1):
            logits = self.model(batch)
            y_true = batch["target"]
            if tag == "train":
                loss, metrics = self.criterion(logits, y_true)
            else:
                with torch.no_grad():
                    loss, metrics = self.criterion(logits, y_true)
            bs = metrics.pop("n_samples", 4)
            epoch_metrics["n_samples"] += bs
            for key, val in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0
                epoch_metrics[key] += val
                if key == "loss":
                    log_metrics[key] = val
                    log_metrics["epoch_loss"] = epoch_metrics[key] / idx
                elif key == "n_correct":
                    log_metrics[key] = epoch_metrics[key]
                    log_metrics["accuracy"] = val / bs
                    log_metrics["epoch_accuracy"] = log_metrics[key] / epoch_metrics["n_samples"]
                    epoch_metrics["accuracy"] = log_metrics["epoch_accuracy"]
                else:
                    log_metrics[key] = epoch_metrics[key]
            
            if tag == "train":
                loss /= self.update_freq
                loss.backward()
                log_metrics["lr"] = self.optimizer.get_current_lr()
                if idx % self.update_freq == 0 or idx == len_dl:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.model.zero_grad()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
            else:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            del logits
            self.tb_log.log(log_metrics, epoch, tag, num_batches + idx, pr)
        epoch_metrics["loss"] = log_metrics["epoch_loss"]
        return epoch_metrics

    def save_model(self, path=None):
        path = path if path else os.path.join(self.checkpoint_dir, self.best_model_path)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path=None):
        path = path if path else os.path.join(self.checkpoint_dir, self.best_model_path)
        self.model.load_state_dict(torch.load(path))
