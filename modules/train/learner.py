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
from collections import defaultdict


class Learner(object):

    @classmethod
    def create(
            cls,
            tensorboard_dir,
            # Model args
            model_name, model_type, model_args=None, tokenizer_name=None,
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
        tokenizer_name = if_none(tokenizer_name, model_name)
        model_args = if_none(model_args, {})
        args = locals()
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_pkl(args, os.path.join(checkpoint_dir, "args.pkl"))
        data_args = {
            "model_name": tokenizer_name,
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
        model_args.update({
            "model_name": model_name,
            "model_type": model_type,
        })
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
            "model_type": model_type,
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
        # We provide only classification
        model_args["num_labels"] = len(list(data.datasets["train"].dictionaries.values())[0])  
        model = GeneralModel.create(**model_args)
        if device == "cuda":
            model = model.cuda()
        len_dl = 0 if data.dataloaders["train"] is None else len(data.dataloaders["train"])
        optimizer_args["t_total"] = if_none(optimizer_args["t_total"],
                                            epochs * len_dl / update_freq)
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
        self._saved_best = False
        self._final_metric_prefix = "final_{}"

    def learn(self, is_final_validate=True):
        best_metric = 0
        len_dl = len(self.data.dataloaders["train"])
        num_batches = len_dl
        for epoch in range(self.epochs):
            epoch += 1
            num_batches = epoch * len_dl
            for split in self.splits:
                if self.data.dataloaders.get(split) is not None:
                    if split == "train":
                        epoch_metrics = self.train_step(self.data.dataloaders[split], epoch, split)
                        if epoch % self.save_every == 0:
                            self.save_model(self.last_model_path)
                    else:
                        epoch_metrics = self.validate_step(self.data.dataloaders[split], epoch, split, num_batches)
                    if split == "valid" and best_metric < epoch_metrics.get(self.target_metric, 0):
                        self._saved_best = True
                        self.save_model(self.best_model_path)
                    self.history[split].append(epoch_metrics)
                    save_pkl(self.history, self.history_path)
        if is_final_validate:
            if self._saved_best:
                self.load_model()
            return self.validate(if_none(num_batches, num_batches + 1))

    def validate(self, num_batches=None):
        metrics = {}
        for split in self.splits:
            if self.data.dataloaders.get(split) is not None:
                metrics[split] = self.validate_step(
                    self.data.dataloaders[split], self.epochs, split, num_batches, False)
        for split, epoch_metrics in metrics.items():
            if epoch_metrics.get(self.target_metric) is not None:
                log_metric = {
                    self._final_metric_prefix.format(self.target_metric): epoch_metrics.get(self.target_metric)
                }
                self.tb_log.log(log_metric, self.epochs, split, num_batches)
        return metrics

    def validate_step(self, dl, epoch, tag, num_batches=None, is_log=True):
        self.model.eval()
        self.optimizer.zero_grad()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        epoch_metrics = {}
        len_dl = len(dl)
        pr = tqdm(dl, total=len_dl, leave=False, desc=tag)
        for idx, batch in enumerate(pr, 1):
            logits = self.model(batch)
            y_true = batch["target"]
            with torch.no_grad():
                loss, metrics = self.criterion(logits, y_true)
            for key, val in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0
                epoch_metrics[key] += val
            del logits
            if self.device == "cuda":
                torch.cuda.empty_cache()

        for key, val in metrics.items():
            if key == "loss":
                epoch_metrics["epoch_loss"] = epoch_metrics["loss"] / len_dl
            elif key == "n_correct":
                epoch_metrics["epoch_accuracy"] = epoch_metrics["n_correct"] / epoch_metrics["n_samples"]
        if is_log:
            self.tb_log.log(epoch_metrics, epoch, tag, num_batches, pr)
        return epoch_metrics

    def train_step(self, dl, epoch, tag, num_batches=None):
        self.model.train()
        self.optimizer.zero_grad()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        epoch_metrics = {"n_samples": 0}
        len_dl = len(dl)
        pr = tqdm(dl, total=len_dl, leave=False, desc=tag)
        num_batches = if_none(num_batches, (epoch - 1) * len_dl)
        log_metrics = {}
        idx = 1
        for idx, batch in enumerate(pr, 1):
            logits = self.model(batch)
            y_true = batch["target"]
            loss, metrics = self.criterion(logits, y_true)
            del logits
            bs = metrics.pop("n_samples")
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

            loss /= self.update_freq
            loss.backward()
            log_metrics["lr"] = self.optimizer.get_current_lr()
            if idx % self.update_freq == 0 or idx == len_dl:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.model.zero_grad()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                self.tb_log.log(log_metrics, epoch, tag, num_batches + idx, pr)
        del loss
        if idx % self.update_freq != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.model.zero_grad()
            self.tb_log.log(log_metrics, epoch, tag, num_batches + idx, pr)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        epoch_metrics["loss"] = log_metrics["epoch_loss"]
        epoch_metrics["num_batches"] = num_batches
        self.tb_log.log(log_metrics, epoch, tag, num_batches + idx, pr)
        return epoch_metrics

    def save_model(self, path=None):
        path = path if path else self.last_model_path
        torch.save(self.model.state_dict(), path)

    def load_model(self, path=None):
        path = path if path else self.last_model_path
        self.model.load_state_dict(torch.load(path))

    def predict_from_dl(self, dl, return_logits=True):
        self.model.eval()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        len_dl = len(dl)
        pr = tqdm(dl, total=len_dl, leave=False, desc="predict")
        res = defaultdict(list)
        logits_res = defaultdict(list)
        for idx, batch in enumerate(pr, 1):
            logits = self.model(batch)
            for key in logits:
                pred = list(logits[key].argmax(-1).cpu().numpy())
                res[key].extend(pred)
                logits_res[key].extend(logits.cpu().numpy())
            del logits
            if self.device == "cuda":
                torch.cuda.empty_cache()

        res = self.data.decode(res)
        if return_logits:
            return res, logits_res
        return res

    def predict(self, lst=None, dl=None, df_path=None, df=None, return_logits=True):
        if dl is None:
            dl = self.data.build_dataloader(df=df, lst=lst, df_path=df_path)
        return self.predict_from_dl(dl, return_logits)
