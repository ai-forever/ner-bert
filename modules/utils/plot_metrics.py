import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from .utils import tokens2spans, bert_labels2tokens, voting_choicer, first_choicer
from sklearn_crfsuite.metrics import flat_classification_report


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


def get_bert_span_report(dl, preds, ignore_labels=["O"], fn=first_choicer):
    tokens, labels = bert_labels2tokens(dl, preds, fn)
    spans_pred = tokens2spans(tokens, labels)
    tokens, labels = bert_labels2tokens(dl, [x.labels for x in dl.dataset], fn)
    spans_true = tokens2spans(tokens, labels)
    set_labels = set()
    for idx in range(len(spans_pred)):
        while len(spans_pred[idx]) < len(spans_true[idx]):
            spans_pred[idx].append(("", "O"))
        while len(spans_pred[idx]) > len(spans_true[idx]):
            spans_true[idx].append(("O", "O"))
        set_labels.update([y for x, y in spans_true[idx]])
    set_labels -= set(ignore_labels)
    return flat_classification_report([[y[1] for y in x] for x in spans_true], [[y[1] for y in x] for x in spans_pred], labels=list(set_labels), digits=3)


def get_elmo_span_report(dl, preds, ignore_labels=["O"]):
    tokens, labels = [x.tokens[1:-1] for x in dl.dataset], [p[1:-1] for p in preds]
    spans_pred = tokens2spans(tokens, labels)
    labels = [x.labels[1:-1] for x in dl.dataset]
    spans_true = tokens2spans(tokens, labels)
    set_labels = set()
    for idx in range(len(spans_pred)):
        while len(spans_pred[idx]) < len(spans_true[idx]):
            spans_pred[idx].append(("", "O"))
        while len(spans_pred[idx]) > len(spans_true[idx]):
            spans_true[idx].append(("O", "O"))
        set_labels.update([y for x, y in spans_true[idx]])
    set_labels -= set(ignore_labels)
    return flat_classification_report([[y[1] for y in x] for x in spans_true], [[y[1] for y in x] for x in spans_pred], labels=list(set_labels), digits=3)


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
        assert len(label_r) == len(f.tokens) - 1
        for idx, (l, rl, t) in enumerate(zip(label, label_r, f.tokens)):
            if l != rl:
                errors_.append({"token: ": t,
                               "real_label": rl,
                               "pred_label": l,
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
