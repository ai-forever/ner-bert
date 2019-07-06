from modules import tqdm
from sklearn_crfsuite.metrics import flat_classification_report
import logging
import torch
from .optimization import BertAdam
from modules.analyze_utils.plot_metrics import get_mean_max_metric
from modules.data.bert_data import get_data_loader_for_predict


def train_step(dl, model, optimizer, num_epoch=1):
    model.train()
    epoch_loss = 0
    idx = 0
    pr = tqdm(dl, total=len(dl), leave=False)
    for batch in pr:
        idx += 1
        model.zero_grad()
        loss = model.score(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.data.cpu().tolist()
        epoch_loss += loss
        pr.set_description("train loss: {}".format(epoch_loss / idx))
        torch.cuda.empty_cache()
    logging.info("\nepoch {}, average train epoch loss={:.5}\n".format(
        num_epoch, epoch_loss / idx))


def transformed_result(preds, mask, id2label, target_all=None, pad_idx=0):
    preds_cpu = []
    targets_cpu = []
    lc = len(id2label)
    if target_all is not None:
        for batch_p, batch_t, batch_m in zip(preds, target_all, mask):
            for pred, true_, bm in zip(batch_p, batch_t, batch_m):
                sent = []
                sent_t = []
                bm = bm.sum().cpu().data.tolist()
                for p, t in zip(pred[:bm], true_[:bm]):
                    p = p.cpu().data.tolist()
                    p = p if p < lc else pad_idx
                    sent.append(p)
                    sent_t.append(t.cpu().data.tolist())
                preds_cpu.append([id2label[w] for w in sent])
                targets_cpu.append([id2label[w] for w in sent_t])
    else:
        for batch_p, batch_m in zip(preds, mask):
            
            for pred, bm in zip(batch_p, batch_m):
                assert len(pred) == len(bm)
                bm = bm.sum().cpu().data.tolist()
                sent = pred[:bm].cpu().data.tolist()
                preds_cpu.append([id2label[w] for w in sent])
    if target_all is not None:
        return preds_cpu, targets_cpu
    else:
        return preds_cpu

    
def transformed_result_cls(preds, target_all, cls2label, return_target=True):
    preds_cpu = []
    targets_cpu = []
    for batch_p, batch_t in zip(preds, target_all):
        for pred, true_ in zip(batch_p, batch_t):
            preds_cpu.append(cls2label[pred.cpu().data.tolist()])
            if return_target:
                targets_cpu.append(cls2label[true_.cpu().data.tolist()])
    if return_target:
        return preds_cpu, targets_cpu
    return preds_cpu


def validate_step(dl, model, id2label, sup_labels, id2cls=None):
    model.eval()
    idx = 0
    preds_cpu, targets_cpu = [], []
    preds_cpu_cls, targets_cpu_cls = [], []
    for batch in tqdm(dl, total=len(dl), leave=False):
        idx += 1
        labels_mask, labels_ids = batch[1], batch[3]
        preds = model.forward(batch)
        if id2cls is not None:
            preds, preds_cls = preds
            preds_cpu_, targets_cpu_ = transformed_result_cls([preds_cls], [batch[-1]], id2cls)
            preds_cpu_cls.extend(preds_cpu_)
            targets_cpu_cls.extend(targets_cpu_)
        preds_cpu_, targets_cpu_ = transformed_result([preds], [labels_mask], id2label, [labels_ids])
        preds_cpu.extend(preds_cpu_)
        targets_cpu.extend(targets_cpu_)
    clf_report = flat_classification_report(targets_cpu, preds_cpu, labels=sup_labels, digits=3)
    if id2cls is not None:
        clf_report_cls = flat_classification_report([targets_cpu_cls], [preds_cpu_cls], digits=3)
        return clf_report, clf_report_cls
    return clf_report


def predict(dl, model, id2label, id2cls=None):
    model.eval()
    idx = 0
    preds_cpu = []
    preds_cpu_cls = []
    for batch in tqdm(dl, total=len(dl), leave=False, desc="Predicting"):
        idx += 1
        labels_mask, labels_ids = batch[1], batch[3]
        preds = model.forward(batch)
        if id2cls is not None:
            preds, preds_cls = preds
            preds_cpu_ = transformed_result_cls([preds_cls], [preds_cls], id2cls, False)
            preds_cpu_cls.extend(preds_cpu_)

        preds_cpu_ = transformed_result([preds], [labels_mask], id2label)
        preds_cpu.extend(preds_cpu_)
    if id2cls is not None:
        return preds_cpu, preds_cpu_cls
    return preds_cpu


class NerLearner(object):

    def __init__(self, model, data, best_model_path, lr=0.001, betas=[0.8, 0.9], clip=1.0,
                 verbose=True, sup_labels=None, t_total=-1, warmup=0.1, weight_decay=0.01,
                 validate_every=1, schedule="warmup_linear", e=1e-6):
        logging.basicConfig(level=logging.INFO)
        self.model = model
        self.optimizer = BertAdam(model, lr, t_total=t_total, b1=betas[0], b2=betas[1], max_grad_norm=clip)
        self.optimizer_defaults = dict(
            model=model, lr=lr, warmup=warmup, t_total=t_total, schedule=schedule,
            b1=betas[0], b2=betas[1], e=e, weight_decay=weight_decay,
            max_grad_norm=clip)

        self.lr = lr
        self.betas = betas
        self.clip = clip
        self.sup_labels = sup_labels
        self.t_total = t_total
        self.warmup = warmup
        self.weight_decay = weight_decay
        self.validate_every = validate_every
        self.schedule = schedule
        self.data = data
        self.e = e
        if sup_labels is None:
            sup_labels = data.train_ds.idx2label[4:]
        self.sup_labels = sup_labels
        self.best_model_path = best_model_path
        self.verbose = verbose
        self.history = []
        self.cls_history = []
        self.epoch = 0
        self.best_target_metric = 0.

    def fit(self, epochs=100, resume_history=True, target_metric="f1"):
        if not resume_history:
            self.optimizer_defaults["t_total"] = epochs * len(self.data.train_dl)
            self.optimizer = BertAdam(**self.optimizer_defaults)
            self.history = []
            self.cls_history = []
            self.epoch = 0
            self.best_target_metric = 0.
        elif self.verbose:
            logging.info("Resuming train... Current epoch {}.".format(self.epoch))
        try:
            for _ in range(epochs):
                self.epoch += 1
                self.fit_one_cycle(self.epoch, target_metric)
        except KeyboardInterrupt:
            pass

    def fit_one_cycle(self, epoch, target_metric="f1"):
        train_step(self.data.train_dl, self.model, self.optimizer, epoch)
        if epoch % self.validate_every == 0:
            if self.data.train_ds.is_cls:
                rep, rep_cls = validate_step(
                    self.data.valid_dl, self.model, self.data.train_ds.idx2label, self.sup_labels,
                    self.data.train_ds.idx2cls)
                self.cls_history.append(rep_cls)
            else:
                rep = validate_step(
                    self.data.valid_dl, self.model, self.data.train_ds.idx2label, self.sup_labels)
            self.history.append(rep)
        idx, metric = get_mean_max_metric(self.history, target_metric, True)
        if self.verbose:
            logging.info("on epoch {} by max_{}: {}".format(idx, target_metric, metric))
            print(self.history[-1])
            if self.data.train_ds.is_cls:
                logging.info("on epoch {} classification report:")
                print(self.cls_history[-1])
        # Store best model
        if self.best_target_metric < metric:
            self.best_target_metric = metric
            if self.verbose:
                logging.info("Saving new best model...")
            self.save_model()

    def predict(self, dl=None, df_path=None, df=None):
        if dl is None:
            dl = get_data_loader_for_predict(self.data, df_path, df)
        if self.data.train_ds.is_cls:
            return predict(dl, self.model, self.data.train_ds.idx2label, self.data.train_ds.idx2cls)
        return predict(dl, self.model, self.data.train_ds.idx2label)
    
    def save_model(self, path=None):
        path = path if path else self.best_model_path
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path=None):
        path = path if path else self.best_model_path
        self.model.load_state_dict(torch.load(path))
