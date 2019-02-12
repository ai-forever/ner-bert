from torch.utils.data import DataLoader
from modules.data import tokenization
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import json


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            # Bert data
            bert_tokens, input_ids, input_mask, input_type_ids,
            # Origin data
            tokens, labels, labels_ids, labels_mask, tok_map, cls=None, cls_idx=None, meta=None):
        """
        Data has the following structure.
        data[0]: list, tokens ids
        data[1]: list, tokens mask
        data[2]: list, tokens type ids (for bert)
        data[3]: list, tokens meta info (if meta is not None)
        ...
        data[-2]: list, labels mask
        data[-1]: list, labels ids
        """
        self.data = []
        # Bert data
        self.bert_tokens = bert_tokens
        self.input_ids = input_ids
        self.data.append(input_ids)
        self.input_mask = input_mask
        self.data.append(input_mask)
        self.input_type_ids = input_type_ids
        self.data.append(input_type_ids)
        # Meta data
        self.meta = meta
        if meta is not None:
            self.data.append(meta)
        # Origin data
        self.tokens = tokens
        self.labels = labels
        # Used for joint model
        self.cls = cls
        self.cls_idx = cls_idx
        if cls is not None:
            self.data.append(cls_idx)
        # Labels data
        self.labels_mask = labels_mask
        self.data.append(labels_mask)
        self.labels_ids = labels_ids
        self.data.append(labels_ids)
        self.tok_map = tok_map


class DataLoaderForTrain(DataLoader):

    def __init__(self, data_set, shuffle, cuda, **kwargs):
        super(DataLoaderForTrain, self).__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            **kwargs
        )
        self.cuda = cuda

    def collate_fn(self, data):
        res = []
        token_ml = max(map(lambda x_: sum(x_.data[1]), data))
        label_ml = max(map(lambda x_: sum(x_.data[-2]), data))
        sorted_idx = np.argsort(list(map(lambda x_: sum(x_.data[1]), data)))[::-1]
        for idx in sorted_idx:
            f = data[idx]
            example = []
            for idx_, x in enumerate(f.data[:-2]):
                if isinstance(x, list):
                    x = x[:token_ml]
                example.append(x)
            example.append(f.data[-2][:label_ml])
            example.append(f.data[-1][:label_ml])
            res.append(example)
        res_ = []
        for idx, x in enumerate(zip(*res)):
            if data[0].meta is not None and idx == 3:
                res_.append(torch.FloatTensor(x))
            else:
                res_.append(torch.LongTensor(x))
        if self.cuda:
            res_ = [t.cuda() for t in res_]
        return res_


class DataLoaderForPredict(DataLoader):

    def __init__(self, data_set, cuda, **kwargs):
        super(DataLoaderForPredict, self).__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            **kwargs
        )
        self.cuda = cuda

    def collate_fn(self, data):
        res = []
        token_ml = max(map(lambda x_: sum(x_.data[1]), data))
        label_ml = max(map(lambda x_: sum(x_.data[-2]), data))
        sorted_idx = np.argsort(list(map(lambda x_: sum(x_.data[1]), data)))[::-1]
        for idx in sorted_idx:
            f = data[idx]
            example = []
            for x in f.data[:-2]:
                if isinstance(x, list):
                    x = x[:token_ml]
                example.append(x)
            example.append(f.data[-2][:label_ml])
            example.append(f.data[-1][:label_ml])
            res.append(example)
        res_ = []
        for idx, x in enumerate(zip(*res)):
            if data[0].meta is not None and idx == 3:
                res_.append(torch.FloatTensor(x))
            else:
                res_.append(torch.LongTensor(x))
        sorted_idx = torch.LongTensor(list(sorted_idx))
        if self.cuda:
            res_ = [t.cuda() for t in res_]
            sorted_idx = sorted_idx.cuda()
        return res_, sorted_idx


def get_data(
        df, tokenizer, label2idx=None, max_seq_len=424, pad="<pad>", cls2idx=None,
        is_cls=False, is_meta=False):
    tqdm_notebook = tqdm
    if label2idx is None:
        label2idx = {pad: 0, '[CLS]': 1}
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
        orig_tokens.extend(str(text).split())
        labels = str(labels).split()
        pad_idx = label2idx[pad]
        assert len(orig_tokens) == len(labels)
        # prev_label = ""
        for idx_, (orig_token, label) in enumerate(zip(orig_tokens, labels)):
            # Fix BIO to IO as BERT proposed https://arxiv.org/pdf/1810.04805.pdf
            prefix = "I_"
            if label != "O":
                label = label.split("_")[1]
                # prev_label = label
            # else:
            # prev_label = label
            
            cur_tokens = tokenizer.tokenize(orig_token)
            if max_seq_len - 1 < len(bert_tokens) + len(cur_tokens):
                break
            tok_map.append(len(bert_tokens))
            if is_meta:
                meta_tokens.extend([meta[idx_]] * len(cur_tokens))
            bert_tokens.extend(cur_tokens)
            # ["I_" + label] * (len(cur_tokens) - 1)
            bert_label = [prefix + label] + ["X"] * (len(cur_tokens) - 1)
            bert_labels.extend(bert_label)
        # bert_tokens.append("[SEP]")
        # bert_labels.append("[SEP]")
        if is_meta:
            meta_tokens.append([0] * len(meta[0]))
        # + ["[SEP]"]
        orig_tokens = ["[CLS]"] + orig_tokens

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        labels = bert_labels
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
                          do_lower_case=False, max_seq_len=424, is_meta=False, label2idx=None, cls2idx=None):
    train = pd.read_csv(train)
    valid = pd.read_csv(valid)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    train_f, label2idx = get_data(
        train, tokenizer, label2idx, cls2idx=cls2idx, is_cls=is_cls, max_seq_len=max_seq_len, is_meta=is_meta)
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

    @property
    def config(self):
        config = {
            "train_path": self.train_path,
            "valid_path": self.valid_path,
            "vocab_file": self.vocab_file,
            "data_type": self.data_type,
            "max_seq_len": self.max_seq_len,
            "batch_size": self.batch_size,
            "is_cls": self.is_cls,
            "cuda": self.cuda,
            "is_meta": self.is_meta,
            "label2idx": self.label2idx,
            "cls2idx": self.cls2idx
        }
        return config

    def __init__(self, train_path, valid_path, vocab_file, data_type,
                 train_dl=None, valid_dl=None, tokenizer=None,
                 label2idx=None, max_seq_len=424,
                 cls2idx=None, batch_size=16, cuda=True, is_meta=False):
        self.train_path = train_path
        self.valid_path = valid_path
        self.data_type = data_type
        self.vocab_file = vocab_file
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

    # TODO: write docs
    @classmethod
    def from_config(cls, config, for_train=True):
        if config["data_type"] == "bert_cased":
            do_lower_case = False
            fn = get_bert_data_loaders
        elif config["data_type"] == "bert_uncased":
            do_lower_case = True
            fn = get_bert_data_loaders
        else:
            raise NotImplementedError("No requested mode :(.")
        if config["train_path"] and config["valid_path"] and for_train:
            fn_res = fn(config["train_path"], config["valid_path"], config["vocab_file"], config["batch_size"],
                        config["cuda"], config["is_cls"], do_lower_case, config["max_seq_len"], config["is_meta"],
                        label2idx=config["label2idx"], cls2idx=config["cls2idx"])
        else:
            fn_res = (None, None, tokenization.FullTokenizer(
                vocab_file=config["vocab_file"], do_lower_case=do_lower_case), config["label2idx"],
                      config["max_seq_len"], config["cls2idx"])
        return cls(
            config["train_path"], config["valid_path"], config["vocab_file"], config["data_type"],
            *fn_res, batch_size=config["batch_size"], cuda=config["cuda"], is_meta=config["is_meta"])

        # with open(config_path, "w") as f:
        #    json.dump(config, f)

    @classmethod
    def create(cls,
               train_path, valid_path, vocab_file, batch_size=16, cuda=True, is_cls=False,
               data_type="bert_cased", max_seq_len=424, is_meta=False):
        if data_type == "bert_cased":
            do_lower_case = False
            fn = get_bert_data_loaders
        elif data_type == "bert_uncased":
            do_lower_case = True
            fn = get_bert_data_loaders
        else:
            raise NotImplementedError("No requested mode :(.")
        return cls(train_path, valid_path, vocab_file, data_type, *fn(
            train_path, valid_path, vocab_file, batch_size, cuda, is_cls, do_lower_case, max_seq_len, is_meta),
                   batch_size=batch_size, cuda=cuda, is_meta=is_meta)
