import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from .utils import tokens2spans, bert_labels2tokens, voting_choicer, first_choicer
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import f1_score


def plot_by_class_curve(history, metric_, sup_labels):
    by_class = get_by_class_metric(history, metric_, sup_labels)
    vals = list(by_class.values())
    x = np.arange(len(vals[0]))
    args = []
    for val in vals:
        args.append(x)
        args.append(val)
    plt.figure(figsize=(15, 10))
    plt.grid(True)
    plt.plot(*args)
    plt.legend(list(by_class.keys()))
    _, _ = plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()


def get_metrics_by_class(text_res, sup_labels):
    # text_res = flat_classification_report(y_true, y_pred, labels=labels, digits=3)
    res = {}
    for line in text_res.split("\n"):
        line = line.split()
        if len(line) and line[0] in sup_labels:
            res[line[0]] = {key: val for key, val in zip(["prec", "rec", "f1"], line[1:-1])}
    return res


def get_by_class_metric(history, metric_, sup_labels):
    res = defaultdict(list)
    for h in history:
        h = get_metrics_by_class(h, sup_labels)
        for class_, metrics_ in h.items():
            res[class_].append(float(metrics_[metric_]))
    return res


def get_max_metric(history, metric_, sup_labels, return_idx=False):
    by_class = get_by_class_metric(history, metric_, sup_labels)
    by_class_arr = np.array(list(by_class.values()))
    idx = np.array(by_class_arr.sum(0)).argmax()
    if return_idx:
        return list(zip(by_class.keys(), by_class_arr[:, idx])), idx
    return list(zip(by_class.keys(), by_class_arr[:, idx]))


def get_mean_max_metric(history, metric_="f1", return_idx=False):
    m_idx = 0
    if metric_ == "f1":
        m_idx = 2
    elif m_idx == "rec":
        m_idx = 1
    metrics = [float(h.split("\n")[-3].split()[2 + m_idx]) for h in history]
    idx = np.argmax(metrics)
    res = metrics[idx]
    if return_idx:
        return idx, res
    return res


def get_bert_span_report(dl, preds, labels=None, fn=voting_choicer):
    pred_tokens, pred_labels = bert_labels2tokens(dl, preds)
    true_tokens, true_labels = bert_labels2tokens(dl, [x.bert_labels for x in dl.dataset])
    spans_pred = tokens2spans(pred_tokens, pred_labels)
    spans_true = tokens2spans(true_tokens, true_labels)
    res_t = []
    res_p = []
    for pred_span, true_span in zip(spans_pred, spans_true):
        text2span = {t: l for t, l in pred_span}
        for (pt, pl), (tt, tl) in zip(pred_span, true_span):
            res_t.append(tl)
            if tt in text2span:
                res_p.append(pl)
            else:
                res_p.append("O")
    return flat_classification_report([res_t], [res_p], labels=labels, digits=4)


def analyze_bert_errors(dl, labels, fn=voting_choicer):
    errors = []
    res_tokens = []
    res_labels = []
    r_labels = [x.labels for x in dl.dataset]
    for f, l_, rl in zip(dl.dataset, labels, r_labels):
        label = fn(f.tok_map, l_)
        label_r = fn(f.tok_map, rl)
        prev_idx = 0
        errors_ = []
        # if len(label_r) > 1:
        # assert len(label_r) == len(f.tokens) - 1
        for idx, (lbl, rl, t) in enumerate(zip(label, label_r, f.tokens)):
            if lbl != rl:
                errors_.append(
                    {"token: ": t,
                     "real_label": rl,
                     "pred_label": lbl,
                     "bert_token": f.bert_tokens[prev_idx:f.tok_map[idx]],
                     "real_bert_label": f.labels[prev_idx:f.tok_map[idx]],
                     "pred_bert_label": l_[prev_idx:f.tok_map[idx]],
                     "text_example": " ".join(f.tokens[1:-1]),
                     "labels": " ".join(label_r[1:])})
            prev_idx = f.tok_map[idx]
        errors.append(errors_)
        res_tokens.append(f.tokens[1:-1])
        res_labels.append(label[1:])
    return res_tokens, res_labels, errors


def get_f1_score(y_true, y_pred, labels):
    res_t = []
    res_p = []
    for yts, yps in zip(y_true, y_pred):
        for yt, yp in zip(yts, yps):
                res_t.append(yt)
                res_p.append(yp)
    return f1_score(res_t, res_p, average="macro", labels=labels)
