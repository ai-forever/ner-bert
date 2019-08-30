from torch.utils.data import DataLoader
import torch
from pytorch_pretrained_bert import BertTokenizer
from modules.utils import read_config, if_none
from modules import tqdm
import pandas as pd
from copy import deepcopy


class InputFeature(object):
    """A single set of features of data."""

    def __init__(
            self,
            # Bert data
            bert_tokens, input_ids, input_mask, input_type_ids,
            # Ner data
            bert_labels, labels_ids, labels,
            # Origin data
            tokens, tok_map,
            # Cls data
            cls=None, id_cls=None):
        """
        Data has the following structure.
        data[0]: list, tokens ids
        data[1]: list, tokens mask
        data[2]: list, tokens type ids (for bert)
        data[3]: list, bert labels ids
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
        # Ner data
        self.bert_labels = bert_labels
        self.labels_ids = labels_ids
        self.data.append(labels_ids)
        # Classification data
        self.cls = cls
        self.id_cls = id_cls
        if id_cls is not None:
            self.data.append(id_cls)
        # Origin data
        self.tokens = tokens
        self.tok_map = tok_map
        self.labels = labels

    def __iter__(self):
        return iter(self.data)


class TextDataLoader(DataLoader):
    def __init__(self, data_set, shuffle=False, device="cuda", batch_size=16):
        super(TextDataLoader, self).__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            batch_size=batch_size
        )
        self.device = device

    def collate_fn(self, data):
        res = []
        token_ml = max(map(lambda x_: sum(x_.data[1]), data))
        for sample in data:
            example = []
            for x in sample:
                if isinstance(x, list):
                    x = x[:token_ml]
                example.append(x)
            res.append(example)
        res_ = []
        for x in zip(*res):
            res_.append(torch.LongTensor(x))
        return [t.to(self.device) for t in res_]


class TextDataSet(object):

    @classmethod
    def from_config(cls, config, clear_cache=False, df=None):
        return cls.create(**read_config(config), clear_cache=clear_cache, df=df)

    @classmethod
    def create(cls,
               idx2labels_path,
               df_path=None,
               idx2labels=None,
               idx2cls=None,
               idx2cls_path=None,
               min_char_len=1,
               model_name="bert-base-multilingual-cased",
               max_sequence_length=424,
               pad_idx=0,
               clear_cache=False,
               is_cls=False,
               markup="IO",
               df=None, tokenizer=None):
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(model_name)
        config = {
            "min_char_len": min_char_len,
            "model_name": model_name,
            "max_sequence_length": max_sequence_length,
            "clear_cache": clear_cache,
            "df_path": df_path,
            "pad_idx": pad_idx,
            "is_cls": is_cls,
            "idx2labels_path": idx2labels_path,
            "idx2cls_path": idx2cls_path,
            "markup": markup
        }
        if df is None and df_path is not None:
            df = pd.read_csv(df_path, sep='\t')
        elif df is None:
            if is_cls:
                df = pd.DataFrame(columns=["labels", "text", "clf"])
            else:
                df = pd.DataFrame(columns=["labels", "text"])
        if clear_cache:
            _ = cls.create_vocabs(
                df, tokenizer, idx2labels_path, markup, idx2cls_path, pad_idx, is_cls, idx2labels, idx2cls)
        self = cls(tokenizer, df=df, config=config, is_cls=is_cls)
        self.load(df=df)
        return self

    @staticmethod
    def create_vocabs(
            df, tokenizer, idx2labels_path, markup="IO",
            idx2cls_path=None, pad_idx=0, is_cls=False, idx2labels=None, idx2cls=None):
        if idx2labels is None:
            label2idx = {"[PAD]": pad_idx, '[CLS]': 1, '[SEP]': 2, "X": 3}
            idx2label = ["[PAD]", '[CLS]', '[SEP]', "X"]
        else:
            label2idx = {label: idx for idx, label in enumerate(idx2labels)}
            idx2label = idx2labels
        idx2cls = idx2cls
        cls2idx = None
        if is_cls:
            idx2cls = []
            cls2idx = {label: idx for idx, label in enumerate(idx2cls)}
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False, desc="Creating labels vocabs"):
            labels = row.labels.split()
            origin_tokens = row.text.split()
            if is_cls and row.cls not in cls2idx:
                cls2idx[row.cls] = len(cls2idx)
                idx2cls.append(row.cls)
            prev_label = ""
            for origin_token, label in zip(origin_tokens, labels):
                if markup == "BIO":
                    prefix = "B_"
                else:
                    prefix = "I_"
                if label != "O":
                    label = label.split("_")[1]
                    if label == prev_label:
                        prefix = "I_"
                    prev_label = label
                else:
                    prev_label = label
                cur_tokens = tokenizer.tokenize(origin_token)
                bert_label = [prefix + label] + ["X"] * (len(cur_tokens) - 1)
                for label_ in bert_label:
                    if label_ not in label2idx:
                        label2idx[label_] = len(label2idx)
                        idx2label.append(label_)
        with open(idx2labels_path, "w", encoding="utf-8") as f:
            for label in idx2label:
                f.write("{}\n".format(label))

        if is_cls:
            with open(idx2cls_path, "w", encoding="utf-8") as f:
                for label in idx2cls:
                    f.write("{}\n".format(label))

        return label2idx, idx2label, cls2idx, idx2cls

    def load(self, df_path=None, df=None):
        df_path = if_none(df_path, self.config["df_path"])
        if df is None:
            self.df = pd.read_csv(df_path, sep='\t')
        self.label2idx = {}
        self.idx2label = []
        with open(self.config["idx2labels_path"], "r", encoding="utf-8") as f:
            for idx, label in enumerate(f.readlines()):
                label = label.strip()
                self.label2idx[label] = idx
                self.idx2label.append(label)

        if self.config["is_cls"]:
            self.idx2cls = []
            self.cls2idx = {}
            with open(self.config["idx2cls_path"], "r", encoding="utf-8") as f:
                for idx, label in enumerate(f.readlines()):
                    label = label.strip()
                    self.cls2idx[label] = idx
                    self.idx2cls.append(label)

    def create_feature(self, row):
        bert_tokens = []
        bert_labels = []
        orig_tokens = row.text.split()
        origin_labels = row.labels.split()
        tok_map = []
        prev_label = ""
        for orig_token, label in zip(orig_tokens, origin_labels):
            cur_tokens = self.tokenizer.tokenize(orig_token)
            if self.config["max_sequence_length"] - 2 < len(bert_tokens) + len(cur_tokens):
                break
            if self.config["markup"] == "BIO":
                prefix = "B_"
            else:
                prefix = "I_"
            if label != "O":
                label = label.split("_")[1]
                if label == prev_label:
                    prefix = "I_"
                prev_label = label
            else:
                prev_label = label
            cur_tokens = self.tokenizer.tokenize(orig_token)
            bert_label = [prefix + label] + ["X"] * (len(cur_tokens) - 1)
            tok_map.append(len(bert_tokens))
            bert_tokens.extend(cur_tokens)
            bert_labels.extend(bert_label)

        orig_tokens = ["[CLS]"] + orig_tokens + ["[SEP]"]
        bert_labels = ["[CLS]"] + bert_labels + ["[SEP]"]
        if self.config["markup"] == "BIO":
            O_label = self.label2idx.get("B_O")
        else:
            O_label = self.label2idx.get("I_O")
        input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_tokens + ['[SEP]'])
        labels_ids = [self.label2idx.get(l, O_label) for l in bert_labels]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < self.config["max_sequence_length"]:
            input_ids.append(self.config["pad_idx"])
            labels_ids.append(self.config["pad_idx"])
            input_mask.append(0)
            tok_map.append(-1)
        input_type_ids = [0] * len(input_ids)
        cls = None
        id_cls = None
        if self.is_cls:
            cls = row.cls
            try:
                id_cls = self.cls2idx[cls]
            except KeyError:
                id_cls = self.cls2idx[str(cls)]
        return InputFeature(
            # Bert data
            bert_tokens=bert_tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            bert_labels=bert_labels, labels_ids=labels_ids, labels=origin_labels,
            # Origin data
            tokens=orig_tokens,
            tok_map=tok_map,
            # Cls
            cls=cls, id_cls=id_cls
        )

    def __getitem__(self, item):
        if self.config["df_path"] is None and self.df is None:
            raise ValueError("Should setup df_path or df.")
        if self.df is None:
            self.load()

        return self.create_feature(self.df.iloc[item])

    def __len__(self):
        return len(self.df) if self.df is not None else 0

    def save(self, df_path=None):
        df_path = if_none(df_path, self.config["df_path"])
        self.df.to_csv(df_path, sep='\t', index=False)

    def __init__(
            self, tokenizer,
            df=None,
            config=None,
            idx2label=None,
            idx2cls=None,
            is_cls=False):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.idx2label = idx2label
        self.label2idx = None
        if idx2label is not None:
            self.label2idx = {label: idx for idx, label in enumerate(idx2label)}

        self.idx2cls = idx2cls

        if idx2cls is not None:
            self.cls2idx = {label: idx for idx, label in enumerate(idx2cls)}
        self.is_cls = is_cls


class LearnData(object):
    def __init__(self, train_ds=None, train_dl=None, valid_ds=None, valid_dl=None):
        self.train_ds = train_ds
        self.train_dl = train_dl
        self.valid_ds = valid_ds
        self.valid_dl = valid_dl

    @classmethod
    def create(cls,
               # DataSet params
               train_df_path,
               valid_df_path,
               idx2labels_path,
               idx2labels=None,
               idx2cls=None,
               idx2cls_path=None,
               min_char_len=1,
               model_name="bert-base-multilingual-cased",
               max_sequence_length=424,
               pad_idx=0,
               clear_cache=False,
               is_cls=False,
               markup="IO",
               train_df=None,
               valid_df=None,
               # DataLoader params
               device="cuda", batch_size=16):
        train_ds = None
        train_dl = None
        valid_ds = None
        valid_dl = None
        if idx2labels_path is not None:
            train_ds = TextDataSet.create(
                idx2labels_path,
                train_df_path,
                idx2labels=idx2labels,
                idx2cls=idx2cls,
                idx2cls_path=idx2cls_path,
                min_char_len=min_char_len,
                model_name=model_name,
                max_sequence_length=max_sequence_length,
                pad_idx=pad_idx,
                clear_cache=clear_cache,
                is_cls=is_cls,
                markup=markup,
                df=train_df)
            if len(train_ds):
                train_dl = TextDataLoader(train_ds, device=device, shuffle=True, batch_size=batch_size)
        if valid_df_path is not None:
            valid_ds = TextDataSet.create(
                idx2labels_path,
                valid_df_path,
                idx2labels=train_ds.idx2label,
                idx2cls=train_ds.idx2cls,
                idx2cls_path=idx2cls_path,
                min_char_len=min_char_len,
                model_name=model_name,
                max_sequence_length=max_sequence_length,
                pad_idx=pad_idx,
                clear_cache=False,
                is_cls=is_cls,
                markup=markup,
                df=valid_df, tokenizer=train_ds.tokenizer)
            valid_dl = TextDataLoader(valid_ds, device=device, batch_size=batch_size)

        self = cls(train_ds, train_dl, valid_ds, valid_dl)
        self.device = device
        self.batch_size = batch_size
        return self

    def load(self):
        if self.train_ds is not None:
            self.train_ds.load()
        if self.valid_ds is not None:
            self.valid_ds.load()

    def save(self):
        if self.train_ds is not None:
            self.train_ds.save()
        if self.valid_ds is not None:
            self.valid_ds.save()


def get_data_loader_for_predict(data, df_path=None, df=None):
    config = deepcopy(data.train_ds.config)
    config["df_path"] = df_path
    config["clear_cache"] = False
    ds = TextDataSet.create(
        idx2labels=data.train_ds.idx2label,
        idx2cls=data.train_ds.idx2cls,
        df=df, tokenizer=data.train_ds.tokenizer, **config)
    return TextDataLoader(
        ds, device=data.device, batch_size=data.batch_size, shuffle=False)
