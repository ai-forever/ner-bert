
def to_words(dl, tok_maps, preds=None):
    text_tokens = dl.dataset
    res = []
    res_preds = []
    assert len(tok_maps) == len(dl.dataset)
    for idx, (feature, tok_map) in enumerate(zip(dl.dataset, tok_maps)):
        map_idx = 0
        sent = []
        word = []
        labels = []
        label = []
        for idx_tok, tok in enumerate(feature.tokens[1:-1], 1):
            if map_idx >= len(tok_map):
                word.append(tok.replace("#", ""))
                if preds is not None:
                    label.append(preds[idx][idx_tok])
            elif idx_tok != tok_map[map_idx]:
                word.append(tok.replace("#", ""))
                if preds is not None:
                    label.append(preds[idx][idx_tok])
            elif idx_tok == tok_map[map_idx]:
                if len(word):
                    sent.append("".join(word))
                word = []
                word.append(tok.replace("#", ""))
                if preds is not None:
                    if len(label):
                        label = label[0]
                        labels.append(label)
                    label = []
                    label.append(preds[idx][idx_tok])
                map_idx += 1
            else:
                raise 
        if len(word):
            sent.append("".join(word))
            if preds is not None:
                labels.append(label[0])
        res.append(sent)
        res_preds.append(labels)
    return res, res_preds


def tokens2spans_(tokens_, labels_):
    res = []
    idx_ = 0
    while idx_ < len(tokens_):
        label = labels_[idx_]
        if label in ["I_O", "B_O"]:
            res.append((tokens_[idx_], "O"))
            idx_ += 1
        else:
            span = [tokens_[idx_]]
            span_label = labels_[idx_].split("_")[1]
            idx_ += 1
            while idx_ < len(tokens_) and labels_[idx_] not in ["I_O", "B_O"] and labels_[idx_].split("_")[0]=="I":
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
