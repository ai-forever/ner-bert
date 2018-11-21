import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from modules.utils import to_words, tokens2spans
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
    idx = by_class_arr.sum(0).argmax()
    if return_idx:
        return list(zip(by_class.keys(), by_class_arr[:, idx])), idx
    return list(zip(by_class.keys(), by_class_arr[:, idx]))


def get_mean_max_metric(history, sup_labels, metric_="f1", return_idx=False):
    res, idx = get_max_metric(history, metric_, sup_labels, True)
    res = np.mean([m[1] for m in res])
    if return_idx:
        return idx, res
    return res


def get_span_report(dl, orig_to_tok_map, preds, ignore_labels=["O"]):
    tokens, labels = to_words(dl, orig_to_tok_map, preds)
    spans_pred = tokens2spans(tokens, labels)
    tokens, labels = to_words(dl, orig_to_tok_map, [x.labels for x in dl.dataset])
    spans_true = tokens2spans(tokens, labels)
    set_labels = set()
    for idx in range(len(spans_pred)):
        while len(spans_pred[idx]) < len(spans_true[idx]):
            spans_pred[idx].append(("", "ERROR"))
        while len(spans_pred[idx]) > len(spans_true[idx]):
            spans_true[idx].append(("ERROR", "ERROR"))
        set_labels.update([y for x, y in spans_true[idx]])
    set_labels -= set(ignore_labels)
    return flat_classification_report([[y[1] for y in x] for x in spans_true], [[y[1] for y in x] for x in spans_pred], labels=list(set_labels), digits=3)
