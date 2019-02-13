from torch.utils.data import DataLoader
from modules.data import tokenization
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from modules.utils import read_json, save_json
import logging
import os


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            # Bert data
            bert_tokens, input_ids, input_mask, input_type_ids,
            # Origin data
            tokens, labels, labels_ids, labels_mask, tok_map, cls=None, cls_idx=None,
            meta_tokens=None,
            meta=None):
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
        self.meta_tokens = meta_tokens
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
        df, tokenizer, label2idx=None, cls2idx=None, meta2idx=None, is_cls=False, is_meta=False,
        max_seq_len=424, pad="<pad>"):
    if label2idx is None:
        label2idx = {pad: 0, '[CLS]': 1}
    features = []
    all_args = [df["1"].tolist(), df["0"].tolist()]
    if is_cls:
        # Use joint model
        if cls2idx is None:
            cls2idx = dict()
        all_args.append(df["2"].tolist())

    if is_meta:
        # TODO: add multiply meta info
        if meta2idx is None:
            meta2idx = {pad: 0, '[CLS]': 1}
        all_args.append(df["3"].tolist())
    # TODO: add chunks
    total = len(df["0"].tolist())
    cls = None
    meta = None
    for args in tqdm(enumerate(zip(*all_args)), total=total, leave=False):
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
        bert_tokens = []
        bert_labels = []
        bert_tokens.append("[CLS]")
        bert_labels.append("[CLS]")
        orig_tokens = str(text).split()
        labels = str(labels).split()
        pad_idx = label2idx[pad]
        assert len(orig_tokens) == len(labels)
        args = [orig_tokens, labels]
        tok_map = []
        meta_tokens = None
        if is_meta:
            meta_tokens = ["[CLS]"]
            meta = str(meta).split()
            args.append(meta)
        for idx_, ars in enumerate(zip(*args)):
            orig_token, label = ars[:2]
            m = pad
            if is_meta:
                m = ars[2]
            # BIO to IO as BERT proposed https://arxiv.org/pdf/1810.04805.pdf
            prefix = "I_"
            if label != "O":
                label = label.split("_")[1]

            cur_tokens = tokenizer.tokenize(orig_token)
            if max_seq_len - 1 < len(bert_tokens) + len(cur_tokens):
                break
            tok_map.append(len(bert_tokens))
            if is_meta:
                meta_tokens.extend([m] + ["X"] * (len(cur_tokens) - 1))
            bert_tokens.extend(cur_tokens)
            bert_label = [prefix + label] + ["X"] * (len(cur_tokens) - 1)
            bert_labels.extend(bert_label)

        orig_tokens = ["[CLS]"] + orig_tokens

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        labels = bert_labels
        for l in labels:
            if l not in label2idx:
                label2idx[l] = len(label2idx)
        labels_ids = [label2idx[l] for l in labels]
        meta_ids = None
        if is_meta:
            for l in meta_tokens:
                if l not in meta2idx:
                    meta2idx[l] = len(meta2idx)
            meta_ids = [meta2idx[l] for l in meta_tokens]

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
            if is_meta:
                meta_ids.append(meta2idx[pad])
            tok_map.append(-1)
            if is_meta:
                meta_tokens.append([0] * len(meta[0]))
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
            cls_idx=cls_idx,
            # Meta data
            meta_tokens=meta_tokens,
            meta=meta_ids
        ))
        assert len(input_ids) == len(input_mask)
        assert len(input_ids) == len(input_type_ids)
        assert len(input_ids) == len(labels_ids)
        assert len(input_ids) == len(labels_mask)
    return features, label2idx, cls2idx, meta2idx


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

    def get_config(self):
        config = {
            "train_path": self.train_path,
            "valid_path": self.valid_path,
            "bert_vocab_file": self.bert_vocab_file,
            "bert_model_type": self.bert_model_type,
            "idx2label_path": self.idx2label_path,
            "idx2cls_path": self.idx2cls_path,
            "idx2meta_path": self.idx2meta_path,
            "max_seq_len": self.max_seq_len,
            "batch_size": self.batch_size,
            "is_cls": self.is_cls,
            "is_meta": self.is_meta,
            "pad": "<pad>",
            "use_cuda": self.use_cuda,
            "config_path": self.config_path
        }
        return config

    def __init__(self, bert_vocab_file, idx2label, config_path=None, train_path=None, valid_path=None,
                 train_dl=None, valid_dl=None, tokenizer=None,
                 bert_model_type="bert_cased", idx2cls=None, idx2meta=None, max_seq_len=424,
                 batch_size=16, is_meta=False, is_cls=False,
                 idx2label_path=None, idx2cls_path=None, idx2meta_path=None, pad="<pad>", use_cuda=True):
        """Store attributes in one cls. For more doc see BertNerData.create"""
        self.train_path = train_path
        self.valid_path = valid_path
        self.config_path = config_path
        self.bert_model_type = bert_model_type
        self.bert_vocab_file = bert_vocab_file
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.idx2label = idx2label
        self.label2idx = {label: idx for idx, label in enumerate(idx2label)}

        self.idx2meta = idx2meta
        self.is_meta = is_meta
        if is_meta:
            self.meta2idx = {label: idx for idx, label in enumerate(idx2meta)}
        self.idx2cls = idx2cls
        self.is_cls = is_cls
        if is_cls:
            self.cls2idx = {label: idx for idx, label in enumerate(idx2cls)}

        self.use_cuda = use_cuda

        self.pad = pad

        self.idx2label_path = idx2label_path
        self.idx2cls_path = idx2cls_path
        self.idx2meta_path = idx2meta_path

        if is_cls and not idx2cls:
            raise ValueError("Must set idx2cls if run on classification mode.")
        if is_meta and not idx2meta:
            raise ValueError("Must set idx2meta if run on meta info using mode.")

    # TODO: write docs
    @classmethod
    def from_config(cls, config, clear_cache=True):
        """Read config and call create. For more docs, see BertNerData.create"""
        config = read_json(config)
        config["clear_cache"] = clear_cache
        return cls.create(**config)

    @classmethod
    def create(cls,
               bert_vocab_file, config_path=None, train_path=None, valid_path=None,
               idx2label=None, bert_model_type="bert_cased", idx2cls=None, idx2meta=None,
               max_seq_len=424,
               batch_size=16, is_meta=False, is_cls=False,
               idx2label_path=None, idx2cls_path=None, idx2meta_path=None, pad="<pad>", use_cuda=True,
               clear_cache=True):
        """
        Create or skip data loaders, load or create vocabs.

        Parameters
        ----------
        bert_vocab_file : str
            Path of vocabulary for BERT tokenizer.
        config_path : str, or None, optional (default=None)
            Path of config of BertNerData.
        train_path : str or None, optional (default=None)
            Path of train data frame. If not None update idx2label, idx2cls, idx2meta.
        valid_path : str or None, optional (default=None)
            Path of valid data frame. If not None update idx2label, idx2cls, idx2meta.
        idx2label : list or None, optional (default=None)
            Map form index to label.
        bert_model_type : str, optional (default="bert_cased")
            Mode of BERT model (CASED or UNCASED).
        idx2cls : list or None, optional (default=None)
            Map form index to cls.
        idx2meta : list or None, optional (default=None)
            Map form index to meta.
        max_seq_len : int, optional (default=424)
            Max sequence length.
        batch_size : int, optional (default=16)
            Batch size.
        is_meta : bool, optional (default=False)
            Use meta info or not.
        is_cls : bool, optional (default=False)
            Use joint model or single.
        idx2label_path : str or None, optional (default=None)
            Path to idx2label map. If not None and idx2label is None load idx2label.
        idx2cls_path : str or None, optional (default=None)
            Path to idx2cls map. If not None and idx2cls is None load idx2cls.
        idx2meta_path : str or None, optional (default=None)
            Path to idx2meta map. If not None and idx2meta is None load idx2meta.
        pad : str, optional (default="<pad>")
            Padding token.
        use_cuda : bool, optional (default=True)
            Run model on gpu or cpu. If gpu pin tensors in data loaders to gpu.
        clear_cache :
            If True, rewrite all vocabs and BertNerData config.

        Returns
        ----------
        data : BertNerData
            Created object of BertNerData.
        """
        if idx2label is None and idx2label_path is None:
            raise ValueError("Must set idx2label_path.")

        if bert_model_type == "bert_cased":
            do_lower_case = False
        elif bert_model_type == "bert_uncased":
            do_lower_case = True
        else:
            raise NotImplementedError("No requested mode :(.")

        if is_meta and idx2meta is None and idx2meta_path is None:
            raise ValueError("Must idx2meta or idx2meta_path.")

        tokenizer = tokenization.FullTokenizer(vocab_file=bert_vocab_file, do_lower_case=do_lower_case)

        meta2idx = None
        cls2idx = None
        label2idx = None

        if idx2label is None and os.path.exists(str(idx2label_path)):
            idx2label = read_json(idx2label_path)
            label2idx = {label: idx for idx, label in enumerate(idx2label)}
        if is_meta and idx2meta is None and os.path.exists(str(idx2meta_path)):
            idx2meta = read_json(idx2meta_path)
            meta2idx = {label: idx for idx, label in enumerate(idx2meta)}
        if is_cls and idx2cls is None and os.path.exists(str(idx2cls_path)):
            idx2cls = read_json(idx2cls_path)
            cls2idx = {label: idx for idx, label in enumerate(idx2cls)}

        train_dl = None
        if train_path:
            train_df = pd.read_csv(train_path)

            train_f, label2idx, cls2idx, meta2idx = get_data(
                train_df, tokenizer, label2idx, cls2idx, meta2idx, is_cls, is_meta, max_seq_len, pad)
            train_dl = DataLoaderForTrain(
                train_f, batch_size=batch_size, shuffle=True, cuda=use_cuda)
        valid_dl = None
        if valid_path:
            valid_df = pd.read_csv(valid_path)
            valid_f, label2idx, cls2idx, meta2idx = get_data(
                valid_df, tokenizer, label2idx, cls2idx, meta2idx, is_cls, is_meta, max_seq_len)
            valid_dl = DataLoaderForTrain(
                valid_f, batch_size=batch_size, cuda=use_cuda, shuffle=False)

        data = cls(bert_vocab_file, idx2label, config_path, train_path, valid_path,
                   train_dl, valid_dl, tokenizer,
                   bert_model_type, idx2cls, idx2meta, max_seq_len,
                   batch_size, is_meta, is_cls,
                   idx2label_path, idx2cls_path, idx2meta_path, pad, use_cuda)
        if clear_cache:
            logging.info("Saving vocabs...")
            save_json(idx2label, idx2label_path)
            save_json(idx2cls, idx2cls_path)
            save_json(idx2meta, idx2meta_path)
            save_json(data.get_config(), config_path)

        return data
