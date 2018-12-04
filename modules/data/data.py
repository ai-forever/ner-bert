from torch.utils.data import DataLoader
from modules.data import tokenization
import torch
import pandas as pd
import numpy as np


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            # Bert data
            bert_tokens, input_ids, input_mask, input_type_ids,
            # Origin data
            tokens, labels, labels_ids, labels_mask, tok_map, cls=None, cls_idx=None):
        """
        Data has the following structure.
        data[0]: list, tokens ids
        data[1]: list, tokens mask
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

    def __init__(self, data_set, cuda, **kwargs):
        super(DataLoaderForTrain, self).__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            **kwargs
        )
        self.cuda = cuda

    def collate_fn(self, data):
        res = []
        token_ml = max(map(lambda x: sum(x.data[1]), data))
        label_ml = max(map(lambda x: sum(x.data[-2]), data))
        sorted_idx = np.argsort(list(map(lambda x: sum(x.data[1]), data)))[::-1]
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
        res = list(zip(*res))
        res = [torch.LongTensor(x) for x in res]
        if self.cuda:
            res = [t.cuda() for t in res]
        return res


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
        token_ml = max(map(lambda x: sum(x.data[1]), data))
        label_ml = max(map(lambda x: sum(x.data[-2]), data))
        sorted_idx = np.argsort(list(map(lambda x: sum(x.data[1]), data)))[::-1]
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
        res = list(zip(*res))
        res = [torch.LongTensor(x) for x in res]
        sorted_idx = torch.LongTensor(list(sorted_idx))
        if self.cuda:
            res = [t.cuda() for t in res]
            sorted_idx = sorted_idx.cuda()
        return res, sorted_idx


def get_data(df, tokenizer, label2idx=None, max_seq_len=424, pad="<pad>", cls2idx=None, is_cls=False):
    if label2idx is None:
        label2idx = {pad: 0, '[CLS]': 1, '[SEP]': 2}
    features = []
    if is_cls:
        # Use joint model
        if cls2idx is None:
            cls2idx = dict()
        zip_args = zip(df["1"].tolist(), df["0"].tolist(), df["2"].tolist())
    else:
        zip_args = zip(df["1"].tolist(), df["0"].tolist())
    cls = None
    for args in enumerate(zip_args):
        if is_cls:
            idx, (text, labels, cls) = args
        else:
            idx, (text, labels) = args
        tok_map = []
        bert_tokens = []
        bert_labels = []
        bert_tokens.append("[CLS]")
        bert_labels.append("[CLS]")
        orig_tokens = []
        orig_tokens.extend(text.split())
        labels = labels.split()
        pad_idx = label2idx[pad]
        # assert len(orig_tokens) == len(labels)
        prev_label = ""
        for orig_token, label in zip(orig_tokens, labels):
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

            bert_tokens.extend(cur_tokens)
            bert_label = [prefix + label] + ["I_" + label] * (len(cur_tokens) - 1)
            bert_labels.extend(bert_label)
        bert_tokens.append("[SEP]")
        bert_labels.append("[SEP]")

        orig_tokens = ["[CLS]"] + orig_tokens + ["[SEP]"]

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
        # assert len(input_ids) == len(bert_labels_ids)
        input_type_ids = [0] * len(input_ids)
        # For joint model
        cls_idx = None
        if is_cls:
            if cls not in cls2idx:
                cls2idx[cls] = len(cls2idx)
            cls_idx = cls2idx[cls]

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
            cls_idx=cls_idx
        ))
        assert len(input_ids) == len(input_mask)
        assert len(input_ids) == len(input_type_ids)
        assert len(input_ids) == len(labels_ids)
        assert len(input_ids) == len(labels_mask)
    if is_cls:
        
        return features, (label2idx, cls2idx)
    return features, label2idx


def get_bert_data_loaders(train, valid, vocab_file, batch_size=16, cuda=True, is_cls=False, do_lower_case=False):
    train = pd.read_csv(train)
    valid = pd.read_csv(valid)

    cls2idx = None

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    train_f, label2idx = get_data(train, tokenizer, is_cls=is_cls)
    if is_cls:
        label2idx, cls2idx = label2idx
    train_dl = DataLoaderForTrain(
        train_f, batch_size=batch_size, shuffle=True, cuda=cuda)
    valid_f, label2idx = get_data(
        valid, tokenizer, label2idx, cls2idx=cls2idx, is_cls=is_cls)
    if is_cls:
        label2idx, cls2idx = label2idx
    valid_dl = DataLoaderForTrain(
        valid_f, batch_size=batch_size, cuda=cuda)
    if is_cls:
        return train_dl, valid_dl, tokenizer, label2idx, cls2idx
    return train_dl, valid_dl, tokenizer, label2idx


def get_bert_data_loader_for_predict(path, learner):
    df = pd.read_csv(path)
    f, _ = get_data(df, tokenizer=learner.data.tokenizer,
                    label2idx=learner.data.label2idx, cls2idx=learner.data.cls2idx, is_cls=learner.data.is_cls)
    dl = DataLoaderForPredict(
        f, batch_size=learner.data.batch_size, shuffle=False,
        cuda=True)

    return dl


class NerData(object):

    def __init__(self, train_dl, valid_dl, tokenizer, label2idx,
                 cls2idx=None, batch_size=16, cuda=True):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.cls2idx = cls2idx
        self.batch_size = batch_size
        self.cuda = cuda
        self.id2label = sorted(label2idx.keys(), key=lambda x: label2idx[x])
        self.is_cls = False
        if cls2idx is not None:
            self.is_cls = True
            self.id2cls = sorted(cls2idx.keys(), key=lambda x: cls2idx[x])

    @classmethod
    def create(cls,
               train_path, valid_path, vocab_file, batch_size=16, cuda=True, is_cls=False, data_type="bert_cased"):
        if data_type == "bert_cased":
            do_lower_case = False
            fn = get_bert_data_loaders
        elif data_type == "bert_uncased":
            do_lower_case = True
            fn = get_bert_data_loaders
        else:
            raise NotImplementedError("No requested mode :(.")
        return cls(*fn(
            train_path, valid_path, vocab_file, batch_size, cuda, is_cls, do_lower_case),
                   batch_size=batch_size, cuda=cuda)
