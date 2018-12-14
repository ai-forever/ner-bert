import sys
import __main__ as main


def ipython_info():
    return hasattr(main, '__file__')


def bert_labels2tokens(dl, labels):
    res_tokens = []
    res_labels = []
    for f, l in zip(dl.dataset, labels):
        label = []
        prev_idx = 0
        for origin_idx in f.tok_map:
            if l[prev_idx] in ["I_O", "[SEP]", "[CLS]"]:
                l[prev_idx] = "B_O"
            label.append(l[prev_idx])
            prev_idx = origin_idx
            if origin_idx < 0:
                break
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
        elif label == "[SEP]" or label == "<eos>":
            break
        elif label == "[CLS]" or label == "<bos>":
            res.append((tokens_[idx_], label))
            idx_ += 1
        else:
            span = [tokens_[idx_]]
            try:
                span_label = labels_[idx_].split("_")[1]
            except:
                print(label, labels_[idx_].split("_"))
            idx_ += 1
            while idx_ < len(labels_) and labels_[idx_] not in ["I_O", "B_O", "O"] and labels_[idx_].split("_")[0]=="I":
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
