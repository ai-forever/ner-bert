from transformers import tokenization_auto
from modules import logger, tqdm
import pandas as pd
from .utils import save_pkl, load_pkl, collate_tokens
import os
from copy import copy
import torch
from modules.utils import get_hash, if_none


def default_collate(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, list):
        return list(map(torch.LongTensor, batch))
    return torch.LongTensor(batch)


class TransformersDataset(torch.utils.data.Dataset):
    """
    A dataset that provides helpers for batching of documents
    with models from transformers for classification task.
    """

    @staticmethod
    def build_feature(tokenizer, row, dictionaries, markup="IO", max_tokens=512):
        tokenized = tokenizer.encode_plus(row.text)
        for key in tokenized:
            tokenized[key] = [tokenized[key][0]] + tokenized[key][1:-1][:max_tokens - 2] + [tokenized[key][-1]]
        ner2idx = dictionaries.get("ner", {"PAD": 0})
        bos = tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])
        eos = tokenizer.convert_ids_to_tokens(tokenized['input_ids'][-1])
        transformer_tokens = []
        transformer_targets = []
        origin_tokens = row.text.split()
        origin_targets = None
        changed = False
        if hasattr(row, "ner"):
            origin_targets = row.ner.split()
            if bos not in ner2idx:
                changed = True
                ner2idx[bos] = len(ner2idx)
            if eos not in ner2idx:
                changed = True
                ner2idx[bos] = len(ner2idx)
            if eos not in ner2idx:
                changed = True
                ner2idx[bos] = len(ner2idx)

        tok_map = []
        prev_target = ""

        target = {}
        target_meta = {}
        for idx, orig_token in enumerate(origin_tokens):
            cur_tokens = tokenizer.tokenize(orig_token)
            tok_map.append(len(transformer_tokens))
            transformer_tokens.extend(cur_tokens)
            if hasattr(row, "ner"):
                orig_target = origin_targets[idx]
                if markup == "BIO":
                    prefix = "B_"
                else:
                    prefix = "I_"
                if orig_target != "O":
                    orig_target = orig_target.split("_")[1]
                    if orig_target == prev_target:
                        prefix = "I_"
                prev_target = orig_target
                transformer_target = [prefix + orig_target] + ["X"] * (len(cur_tokens) - 1)
                for label_ in transformer_target:
                    if label_ not in ner2idx:
                        changed = True
                        ner2idx[label_] = len(ner2idx)
                transformer_targets.extend(transformer_target)
        if hasattr(row, "ner"):
            if bos not in ner2idx:
                changed = True
                ner2idx[bos] = len(ner2idx)
            if eos not in ner2idx:
                changed = True
                ner2idx[eos] = len(ner2idx)
            origin_targets = [bos] + origin_targets[:max_tokens - 2] + [eos]
            transformer_targets = [bos] + transformer_targets[:max_tokens - 2] + [eos]
            if markup == "BIO":
                o_label = ner2idx.get("B_O")
            else:
                o_label = ner2idx.get("I_O")
            transformer_targets_ids = [ner2idx.get(x, o_label) for x in transformer_targets]
            assert len(tokenized['input_ids']) == len(transformer_targets_ids)
            target["ner"] = transformer_targets_ids
            target_meta["origin_targets"] = origin_targets
            target_meta["transformer_targets"] = transformer_targets
        orig_tokens = [bos] + origin_tokens[:max_tokens - 2] + [eos]
        transformer_tokens = [bos] + transformer_tokens[:max_tokens - 2] + [eos]
        meta = {
            "orig_tokens": orig_tokens,
            "transformer_tokens": transformer_tokens,
            "source": row.text,
            "size": len(list(tokenized.values())[0])
        }
        if hasattr(row, "ner"):
            dictionaries["ner"] = ner2idx
        if hasattr(row, "cls"):
            if dictionaries.get("cls") is None:
                dictionaries["cls"] = {}
            if row.cls not in dictionaries["cls"]:
                changed = True
                dictionaries["cls"][row.cls] = len(dictionaries["cls"])
            target["cls"] = dictionaries["cls"][row.cls]
            target_meta["cls"] = row.cls
        res = {
            "target": target,
            "net_input": tokenized,
            "target_meta": target_meta,
            "meta": meta
        }
        return res, changed, dictionaries

    @classmethod
    def build_features(cls, tokenizer, data_path, dictionaries=None, markup="IO", max_tokens=512):
        df = pd.read_csv(data_path, sep="\t")
        features = []
        is_changing = dictionaries is None
        dictionaries = if_none(dictionaries, {})
        unchanged = False
        for idx, row in tqdm(df.iterrows(), leave=False, total=len(df)):
            res, changed, dictionaries = cls.build_feature(tokenizer, row, dictionaries, markup, max_tokens)
            features.append(res)
        if not is_changing and unchanged:
            logger.warning("You specify dictionaries, that is not contain all labels from dataset. We update that.")
        return features, dictionaries

    @classmethod
    def create(
            cls, model_name_or_path,
            df_path=None,
            dictionaries=None,
            dictionaries_path=None,
            tokenizer_cls=tokenization_auto.AutoTokenizer,
            tokenizer_args=None,
            max_tokens=512,
            clear_cache=False,
            online=False,
            cache_dir="./",
            pad_idx=0,
            markup="IO",
            device="cuda"
    ):
        args = copy(locals())
        tokenizer_args = if_none(tokenizer_args, {})
        tokenizer = tokenizer_cls.from_pretrained(model_name_or_path, **tokenizer_args)
        if tokenizer_cls.__name__ not in vars(tokenization_auto):
            logger.warning("Note, your specify tokenizer, that is not from transformers.")
        if online:
            features = None
            if dictionaries_path is None and dictionaries is None:
                raise ValueError("You should specify dictionaries_path or dictionaries while online mode")
            if dictionaries_path is not None and dictionaries is None:
                dictionaries = load_pkl(dictionaries_path)
        else:
            if dictionaries_path is None and dictionaries is None:
                raise ValueError("You should specify dictionaries_path or dictionaries.")
            if dictionaries_path is not None and dictionaries is None:
                if os.path.exists(dictionaries_path):
                    dictionaries = load_pkl(dictionaries_path)
            cached_name = get_hash(args)
            cached_path = os.path.join(cache_dir, "{}_{}.pkl".format(df_path, cached_name))
            if df_path is not None and (clear_cache or not os.path.exists(cached_path)):
                logger.info("Creating dataset from path {}...".format(df_path))
                features, dictionaries = cls.build_features(tokenizer, df_path, dictionaries, markup, max_tokens)
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                save_pkl(features, cached_path)
            if clear_cache or not os.path.exists(dictionaries_path):
                save_pkl(dictionaries, dictionaries_path)
            if os.path.exists(cached_path):
                features = load_pkl(cached_path)
            else:
                raise FileNotFoundError("File {} not found".format(df_path))

        self = cls(tokenizer, dictionaries, args, features)
        return self

    def __init__(self, tokenizer, dictionaries, args, features=None):
        super(TransformersDataset, self).__init__()
        self.tokenizer = tokenizer
        self.dictionaries = dictionaries
        self.inv_dictionaries = {}
        for key in dictionaries:
            self.inv_dictionaries[key] = [0] * len(dictionaries[key])
            for target, idx in dictionaries[key].items():
                self.inv_dictionaries[key][idx] = target
        self.args = args
        self.features = features
        self._len = 0 if features is None else len(features)

    def __getitem__(self, index):
        return {
            "id": index,
            "size": self.features[index]["meta"]["size"],
            "net_input": self.features[index]["net_input"],
            "target": self.features[index]["target"]
        }

    def __len__(self):
        return self._len

    def collater(self, samples):
        samples = default_collate(samples)
        samples["net_input"] = {
            key: collate_tokens(samples["net_input"][key], self.args["pad_idx"]) for key in samples["net_input"]}
        samples["target"] = {
            key: collate_tokens(
                samples["target"][key]) if key == "ner" else samples["target"][key].view(-1)
            for key in samples["target"]}
        # Now we don't provide multitask
        if len(samples["target"]) > 1:
            raise ValueError("Now we don't provide multitask targets.")
        if self.args["device"] == "cuda":
            samples["net_input"] = {
                key: samples["net_input"][key].cuda() for key in samples["net_input"]}
            samples["target"] = {key: samples["target"][key].cuda() for key in samples["target"]}

        samples["n_samples"] = len(samples)
        samples["n_tokens"] = sum(samples["size"])
        return samples
