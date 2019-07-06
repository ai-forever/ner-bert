from collections import Counter
import numpy as np
import json
import numpy


def voting_choicer(tok_map, labels):
    label = []
    prev_idx = 0
    for origin_idx in tok_map:
        votes = []
        for l in labels[prev_idx:origin_idx]:
            if l != "X":
                votes.append(l)
        vote_labels = Counter(votes)
        if not len(vote_labels):
            vote_labels = {"B_O": 1}
        # vote_labels = Counter(c)
        lb = sorted(list(vote_labels), key=lambda x: vote_labels[x])
        if len(lb):
            label.append(lb[-1])
        prev_idx = origin_idx
        if origin_idx < 0:
            break

    return label


def first_choicer(tok_map, labels):
    label = []
    prev_idx = 0
    for origin_idx in tok_map:
        l = labels[prev_idx]
        if l in ["X"]:
            l = "B_O"
        if l == "B_O":
            for ll in labels[prev_idx + 1:origin_idx]:
                if ll not in ["B_O", "I_O", "X"]:
                    l = ll
                    break
        label.append(l)
        prev_idx = origin_idx
        if origin_idx < 0:
            break
    # assert "[SEP]" not in label
    return label


def bert_labels2tokens(dl, labels, fn=voting_choicer):
    res_tokens = []
    res_labels = []
    for f, l in zip(dl.dataset, labels):
        label = fn(f.tok_map, l[1:])

        res_tokens.append(f.tokens[1:-1])
        res_labels.append(label[1:])
    return res_tokens, res_labels


def tokens2spans_(tokens_, labels_):
    res = []
    idx_ = 0
    while idx_ < len(labels_):
        label = labels_[idx_]
        if label in ["I_O", "B_O", "O"]:
            res.append((tokens_[idx_], "O"))
            idx_ += 1
        elif label == "<eos>":
            break
        elif label == "[CLS]" or label == "<bos>":
            res.append((tokens_[idx_], label))
            idx_ += 1
        else:
            span = [tokens_[idx_]]
            try:
                span_label = labels_[idx_].split("_")[1]
            except IndexError:
                print(label, labels_[idx_].split("_"))
                span_label = None
            idx_ += 1
            while idx_ < len(labels_) and labels_[idx_] not in ["I_O", "B_O", "O"] \
                    and labels_[idx_].split("_")[0] == "I":
                if span_label == labels_[idx_].split("_")[1]:
                    span.append(tokens_[idx_])
                    idx_ += 1
                else:
                    break
            res.append((" ".join(span), span_label))
    return res


def tokens2spans(tokens, labels):
    assert len(tokens) == len(labels)

    return list(map(lambda x: tokens2spans_(*x), zip(tokens, labels)))


def encode_position(pos, emb_dim=10):
    """The sinusoid position encoding"""

    # keep dim 0 for padding token position encoding zero vector
    if pos == 0:
        return np.zeros(emb_dim)
    position_enc = np.array(
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)])

    # apply sin on 0th,2nd,4th...emb_dim
    position_enc[0::2] = np.sin(position_enc[0::2])
    # apply cos on 1st,3rd,5th...emb_dim
    position_enc[1::2] = np.cos(position_enc[1::2])
    return list(position_enc.reshape(-1))


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def jsonify(data):
    return json.dumps(data, cls=JsonEncoder)


def read_json(config):
    if isinstance(config, str):
        with open(config, "r") as f:
            config = json.load(f)
    return config


def save_json(config, path):
    with open(path, "w") as file:
        json.dump(config, file, cls=JsonEncoder)
