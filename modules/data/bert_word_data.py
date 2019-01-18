from modules.data import tokenization
from modules.utils.utils import ipython_info
import pandas as pd
from tqdm import tqdm
import json
from .bert_data import InputFeatures, DataLoaderForTrain, DataLoaderForPredict


def get_data(
        df, tokenizer, label2idx=None, max_seq_len=424, pad="<pad>", cls2idx=None,
        is_cls=False, is_meta=False):
    if label2idx is None:
        label2idx = {pad: 0, '[CLS]': 1, '[SEP]': 2}
    features = []
    all_args = []
    if is_cls:
        # Use joint model
        if cls2idx is None:
            cls2idx = dict()
        all_args.extend([df["1"].tolist(), df["0"].tolist(), df["2"].tolist()])
    else:
        all_args.extend([df["1"].tolist(), df["0"].tolist()])
    if is_meta:
        all_args.append(df["3"].tolist())
    total = len(df["0"].tolist())
    cls = None
    meta = None
    for args in tqdm_notebook(enumerate(zip(*all_args)), total=total, leave=False):
        if is_cls:
            if is_meta:
                idx, (text, labels, cls, meta) = args
            else:
                idx, (text, labels, cls) = args
        else:
            if is_meta:
                idx, (text, labels, meta) = args
            else:
                idx, (text, labels) = args

        tok_map = []
        meta_tokens = []
        if is_meta:
            meta = json.loads(meta)
            meta_tokens.append([0] * len(meta[0]))
        bert_tokens = []
        bert_labels = []
        bert_tokens.append("[CLS]")
        bert_labels.append("[CLS]")
        orig_tokens = []
        orig_tokens.extend(text.split())
        labels = labels.split()
        pad_idx = label2idx[pad]
        assert len(orig_tokens) == len(labels)
        prev_label = ""
        for idx_, (orig_token, label) in enumerate(zip(orig_tokens, labels)):
            prefix = "B_"
            if label != "O":
                label = label.split("_")[1]
                if label == prev_label:
                    prefix = "I_"
                prev_label = label
            else:
                prev_label = label
            tok_map.append(len(bert_tokens))
            cur_tokens = tokenizer.tokenize(orig_token)
            if max_seq_len - 1 < len(bert_tokens) + len(cur_tokens):
                break

            if is_meta:
                meta_tokens.extend([meta[idx_]] * len(cur_tokens))
            bert_tokens.extend(cur_tokens)
            bert_label = [prefix + label] + ["I_" + label] * (len(cur_tokens) - 1)
            bert_labels.extend(bert_label)
        bert_tokens.append("[SEP]")
        bert_labels.append("[SEP]")
        if is_meta:
            meta_tokens.append([0] * len(meta[0]))
        orig_tokens = ["[CLS]"] + orig_tokens + ["[SEP]"]

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        # labels = bert_labels
        for l in labels:
            if l not in label2idx:
                label2idx[l] = len(label2idx)
        labels_ids = [label2idx[l] for l in labels]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        labels_mask = [1] * len(labels_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            labels_ids.append(pad_idx)
            labels_mask.append(0)
            tok_map.append(-1)
            if is_meta:
                meta_tokens.append([0] * len(meta[0]))
        # assert len(input_ids) == len(bert_labels_ids)
        input_type_ids = [0] * len(input_ids)
        # For joint model
        cls_idx = None
        if is_cls:
            if cls not in cls2idx:
                cls2idx[cls] = len(cls2idx)
            cls_idx = cls2idx[cls]
        if is_meta:
            meta = meta_tokens
        features.append(InputFeatures(
            # Bert data
            bert_tokens=bert_tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            # Origin data
            tokens=orig_tokens,
            labels=labels,
            labels_ids=labels_ids,
            labels_mask=labels_mask,
            tok_map=tok_map,
            # Joint data
            cls=cls,
            cls_idx=cls_idx,
            # Meta data
            meta=meta
        ))
        assert len(input_ids) == len(input_mask)
        assert len(input_ids) == len(input_type_ids)
        assert len(input_ids) == len(labels_ids)
        assert len(input_ids) == len(labels_mask)
    if is_cls:
        
        return features, (label2idx, cls2idx)
    return features, label2idx


def get_bert_data_loaders(train, valid, vocab_file, batch_size=16, cuda=True, is_cls=False,
                          do_lower_case=False, max_seq_len=424, is_meta=False):
    train = pd.read_csv(train)
    valid = pd.read_csv(valid)

    cls2idx = None

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    train_f, label2idx = get_data(train, tokenizer, is_cls=is_cls, max_seq_len=max_seq_len, is_meta=is_meta)
    if is_cls:
        label2idx, cls2idx = label2idx
    train_dl = DataLoaderForTrain(
        train_f, batch_size=batch_size, shuffle=True, cuda=cuda)
    valid_f, label2idx = get_data(
        valid, tokenizer, label2idx, cls2idx=cls2idx, is_cls=is_cls, max_seq_len=max_seq_len, is_meta=is_meta)
    if is_cls:
        label2idx, cls2idx = label2idx
    valid_dl = DataLoaderForTrain(
        valid_f, batch_size=batch_size, cuda=cuda, shuffle=False)
    if is_cls:
        return train_dl, valid_dl, tokenizer, label2idx, max_seq_len, cls2idx
    return train_dl, valid_dl, tokenizer, label2idx, max_seq_len


def get_bert_data_loader_for_predict(path, learner):
    df = pd.read_csv(path)
    f, _ = get_data(df, tokenizer=learner.data.tokenizer,
                    label2idx=learner.data.label2idx, cls2idx=learner.data.cls2idx,
                    is_cls=learner.data.is_cls,
                    max_seq_len=learner.data.max_seq_len, is_meta=learner.data.is_meta)
    dl = DataLoaderForPredict(
        f, batch_size=learner.data.batch_size, shuffle=False,
        cuda=True)

    return dl


class BertNerData(object):

    def __init__(self, train_dl, valid_dl, tokenizer, label2idx, max_seq_len=424,
                 cls2idx=None, batch_size=16, cuda=True, is_meta=False):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.cls2idx = cls2idx
        self.batch_size = batch_size
        self.is_meta = is_meta
        self.cuda = cuda
        self.id2label = sorted(label2idx.keys(), key=lambda x: label2idx[x])
        self.is_cls = False
        self.max_seq_len = max_seq_len
        if cls2idx is not None:
            self.is_cls = True
            self.id2cls = sorted(cls2idx.keys(), key=lambda x: cls2idx[x])

    @classmethod
    def create(cls,
               train_path, valid_path, vocab_file, batch_size=16, cuda=True, is_cls=False,
               data_type="bert_cased", max_seq_len=424, is_meta=False):
        if ipython_info():
            global tqdm_notebook
            tqdm_notebook = tqdm
        if data_type == "bert_cased":
            do_lower_case = False
            fn = get_bert_data_loaders
        elif data_type == "bert_uncased":
            do_lower_case = True
            fn = get_bert_data_loaders
        else:
            raise NotImplementedError("No requested mode :(.")
        return cls(*fn(
            train_path, valid_path, vocab_file, batch_size, cuda, is_cls, do_lower_case, max_seq_len, is_meta),
                   batch_size=batch_size, cuda=cuda, is_meta=is_meta)
