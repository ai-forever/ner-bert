from .bert_data import TextDataLoader
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
            # Origin data
            tokens, tok_map,
            # Cls data
            cls=None, id_cls=None):
        """
        Data has the following structure.
        data[0]: list, tokens ids
        data[1]: list, tokens mask
        data[2]: list, tokens type ids (for bert)
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
        # Classification data
        self.cls = cls
        self.id_cls = id_cls
        if cls is not None:
            self.data.append(id_cls)
        # Origin data
        self.tokens = tokens
        self.tok_map = tok_map

    def __iter__(self):
        return iter(self.data)


class TextDataSet(object):

    @classmethod
    def from_config(cls, config, clear_cache=False, df=None):
        return cls.create(**read_config(config), clear_cache=clear_cache, df=df)

    @classmethod
    def create(cls,
               df_path=None,
               idx2cls=None,
               idx2cls_path=None,
               min_char_len=1,
               model_name="bert-base-multilingual-cased",
               max_sequence_length=424,
               pad_idx=0,
               clear_cache=False,
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
            "idx2cls_path": idx2cls_path
        }
        if df is None and df_path is not None:
            df = pd.read_csv(df_path, sep='\t', engine='python')
        elif df is None:
            df = pd.DataFrame(columns=["text", "clf"])
        if clear_cache:
            _, idx2cls = cls.create_vocabs(df, idx2cls_path, idx2cls)
        self = cls(tokenizer, df=df, config=config, idx2cls=idx2cls)
        self.load(df=df)
        return self

    @staticmethod
    def create_vocabs(
            df, idx2cls_path, idx2cls=None):
        idx2cls = idx2cls
        cls2idx = {}
        if idx2cls is not None:
            cls2idx = {label: idx for idx, label in enumerate(idx2cls)}
        else:
            idx2cls = []
        for _, row in tqdm(df.iterrows(), total=len(df), leave=False, desc="Creating labels vocabs"):
            if row.cls not in cls2idx:
                cls2idx[row.cls] = len(cls2idx)
                idx2cls.append(row.cls)

        with open(idx2cls_path, "w", encoding="utf-8") as f:
            for label in idx2cls:
                f.write("{}\n".format(label))

        return cls2idx, idx2cls

    def load(self, df_path=None, df=None):
        df_path = if_none(df_path, self.config["df_path"])
        if df is None:
            self.df = pd.read_csv(df_path, sep='\t')

        self.idx2cls = []
        self.cls2idx = {}
        with open(self.config["idx2cls_path"], "r", encoding="utf-8") as f:
            for idx, label in enumerate(f.readlines()):
                label = label.strip()
                self.cls2idx[label] = idx
                self.idx2cls.append(label)

    def create_feature(self, row):
        bert_tokens = []
        orig_tokens = row.text.split()
        tok_map = []
        for orig_token in orig_tokens:
            cur_tokens = self.tokenizer.tokenize(orig_token)
            if self.config["max_sequence_length"] - 2 < len(bert_tokens) + len(cur_tokens):
                break
            cur_tokens = self.tokenizer.tokenize(orig_token)
            tok_map.append(len(bert_tokens))
            bert_tokens.extend(cur_tokens)

        orig_tokens = ["[CLS]"] + orig_tokens + ["[SEP]"]

        input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_tokens + ['[SEP]'])
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < self.config["max_sequence_length"]:
            input_ids.append(self.config["pad_idx"])
            input_mask.append(0)
            tok_map.append(-1)
        input_type_ids = [0] * len(input_ids)
        cls = str(row.cls)
        id_cls = self.cls2idx[cls]
        return InputFeature(
            # Bert data
            bert_tokens=bert_tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
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
            idx2cls=None):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.label2idx = None

        self.idx2cls = idx2cls
        if idx2cls is not None:
            self.cls2idx = {label: idx for idx, label in enumerate(idx2cls)}


class LearnDataClass(object):
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
               idx2cls=None,
               idx2cls_path=None,
               min_char_len=1,
               model_name="bert-base-multilingual-cased",
               max_sequence_length=424,
               pad_idx=0,
               clear_cache=False,
               train_df=None,
               valid_df=None,
               # DataLoader params
               device="cuda", batch_size=16):
        train_ds = None
        train_dl = None
        valid_ds = None
        valid_dl = None
        if idx2cls_path is not None:
            train_ds = TextDataSet.create(
                train_df_path,
                idx2cls=idx2cls,
                idx2cls_path=idx2cls_path,
                min_char_len=min_char_len,
                model_name=model_name,
                max_sequence_length=max_sequence_length,
                pad_idx=pad_idx,
                clear_cache=clear_cache,
                df=train_df)
            if len(train_ds):
                train_dl = TextDataLoader(train_ds, device=device, shuffle=True, batch_size=batch_size)
        if valid_df_path is not None:
            valid_ds = TextDataSet.create(
                valid_df_path,
                idx2cls=train_ds.idx2cls,
                idx2cls_path=idx2cls_path,
                min_char_len=min_char_len,
                model_name=model_name,
                max_sequence_length=max_sequence_length,
                pad_idx=pad_idx,
                clear_cache=False,
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
        idx2cls=data.train_ds.idx2cls,
        df=df, tokenizer=data.train_ds.tokenizer, **config)
    return TextDataLoader(
        ds, device=data.device, batch_size=data.batch_size, shuffle=False), ds
