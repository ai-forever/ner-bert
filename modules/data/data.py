from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from modules.data import tokenization
import torch
import pandas as pd


class BertDataSet(Dataset):

    def __init__(self, data):
        super(Dataset, self).__init__()
        if isinstance(data, list):
            self.data = data
        else:
            self.data = [data]

    def __getitem__(self, item):
        return [d[item] for d in self.data]

    def __len__(self):
        return len(self.data)


class DataLoaderHelper(DataLoader):

    def __init__(self, data_set, cuda, is_cls=False, **kwargs):
        super(DataLoaderHelper, self).__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            **kwargs
        )
        self.cuda = cuda
        self.is_cls = is_cls

    def collate_fn(self, data):
        input_ids = []
        input_mask = []
        input_type_ids = []
        labels_ids = []
        cls_ids = []
        lens = list(map(lambda x: len(x.tokens), data))
        max_len = max(lens)
        for f in data:
            input_ids.append(f.input_ids[:max_len])
            input_mask.append(f.input_mask[:max_len])
            input_type_ids.append(f.input_type_ids[:max_len])
            labels_ids.append(f.labels_ids[:max_len])
            if self.is_cls:
                cls_ids.append(f.cls_idx)
        res = [torch.LongTensor(input_ids),
               torch.LongTensor(input_mask),
               torch.LongTensor(input_type_ids),
               torch.LongTensor(labels_ids)]
        if self.is_cls:
            res.append(torch.LongTensor(cls_ids))
        if self.cuda:
            res = [t.cuda() for t in res]
        return res


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids,
                 labels, labels_ids, cls=None, cls_idx=None):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.labels = labels
        self.labels_ids = labels_ids
        # Used for joint model
        self.cls = cls
        self.cls_idx = cls_idx


def get_bert_data(df, tokenizer, label2idx=None, max_seq_len=424, pad="<pad>", cls2idx=None, is_cls=False):
    if label2idx is None:
        label2idx = {pad: 0, '[CLS]': 1, '[SEP]': 2, 'B_O': 3, 'I_O': 4}
    orig_to_tok_map = []
    features = []
    # is_cls = "2" in df.columns
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
        orig_tokens = text.split()
        labels = labels.split()
        pad_idx = label2idx[pad]
        assert len(orig_tokens) == len(labels)
        for orig_token, label in zip(orig_tokens, labels):
            label = label if label == "O" else label.split("_")[1]
            tok_map.append(len(bert_tokens))
            cur_tokens = tokenizer.tokenize(orig_token)
            bert_tokens.extend(cur_tokens)
            bert_label = ["B_"+label] + ["I_"+label] * (len(cur_tokens) - 1)
            bert_labels.extend(bert_label)
        bert_tokens.append("[SEP]")
        bert_labels.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        for l in bert_labels:
            if l not in label2idx:
                label2idx[l] = len(label2idx)
        bert_labels_ids = [label2idx[l] for l in bert_labels]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            bert_labels_ids.append(pad_idx)
        assert len(input_ids) == len(bert_labels_ids)
        input_type_ids = [0] * len(input_ids)
        # For joint model
        cls_idx = None
        if is_cls:
            if cls not in cls2idx:
                cls2idx[cls] = len(cls2idx)
            cls_idx = cls2idx[cls]
        features.append(InputFeatures(
            unique_id=idx, 
            tokens=bert_tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            labels=bert_labels,
            labels_ids=bert_labels_ids,
            cls=cls,
            cls_idx=cls_idx
        ))
        orig_to_tok_map.append(tok_map)
    if is_cls:
        return features, orig_to_tok_map, (label2idx, cls2idx)
    return features, orig_to_tok_map, label2idx


def get_data_loaders(train, valid, vocab_file, batch_size=16, cuda=True, is_cls=False):
    train = pd.read_csv(train)
    valid = pd.read_csv(valid)

    cls2idx = None

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    train_f, train_orig_to_tok_map, label2idx = get_bert_data(train, tokenizer, is_cls=is_cls)
    if is_cls:
        label2idx, cls2idx = label2idx
    train_dl = DataLoaderHelper(
        train_f, batch_size=batch_size, shuffle=True, cuda=cuda, is_cls=is_cls)
    valid_f, valid_orig_to_tok_map, label2idx = get_bert_data(
        valid, tokenizer, label2idx, cls2idx=cls2idx, is_cls=is_cls)
    if is_cls:
        label2idx, cls2idx = label2idx
    valid_dl = DataLoaderHelper(
        valid_f, batch_size=batch_size, cuda=cuda, is_cls=is_cls)
    if is_cls:
        return train_dl, train_orig_to_tok_map, valid_dl,\
               valid_orig_to_tok_map, tokenizer, label2idx, cls2idx
    return train_dl, train_orig_to_tok_map, valid_dl, valid_orig_to_tok_map, tokenizer, label2idx


def get_data_loader_for_predict(path, learner):
    df = pd.read_csv(path)
    f, orig_to_tok_map, _ = get_bert_data(
        df, learner.data.tokenizer, learner.data.label2idx, cls2idx=learner.data.cls2idx)
    dl = DataLoaderHelper(
        f, batch_size=learner.data.batch_size, shuffle=False,
        cuda=learner.model.use_cuda, is_cls=learner.data.is_cls)

    return dl, orig_to_tok_map


class NerData(object):

    def __init__(self, train_dl, train_orig_to_tok_map, valid_dl, valid_orig_to_tok_map,
                 tokenizer, label2idx, cls2idx=None, batch_size=16, cuda=True):
        self.train_dl = train_dl
        self.train_orig_to_tok_map = train_orig_to_tok_map
        self.valid_orig_to_tok_map = valid_orig_to_tok_map
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

    @staticmethod
    def create(train_path, valid_path, vocab_file, batch_size=16, cuda=True, is_cls=False):

        return NerData(*get_data_loaders(
            train_path, valid_path, vocab_file, batch_size, cuda, is_cls), batch_size=batch_size, cuda=cuda)
