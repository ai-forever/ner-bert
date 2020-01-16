from transformers import tokenization_auto
from .dataset import TransformersDataset
from torch.utils.data import DataLoader


class TransformerData(object):

    @classmethod
    def create(
            cls,
            model_name,
            train_df_path=None,
            valid_df_path=None,
            test_df_path=None,
            dictionaries=None,
            dictionaries_path=None,
            tokenizer_cls=tokenization_auto.AutoTokenizer,
            tokenizer_args=None,
            max_tokens=512,
            clear_cache=False,
            online=False,
            shuffle=True,
            cache_dir="./",
            pad_idx=0,
            markup="IO",
            batch_size=32
    ):
        train_dataset = TransformersDataset.create(
            model_name,
            df_path=train_df_path,
            dictionaries=dictionaries,
            dictionaries_path=dictionaries_path,
            tokenizer_cls=tokenizer_cls,
            tokenizer_args=tokenizer_args,
            max_tokens=max_tokens,
            clear_cache=clear_cache,
            online=online,
            cache_dir=cache_dir,
            pad_idx=pad_idx,
            markup=markup
        )
        train_dl = None
        if len(train_dataset):
            train_dl = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=train_dataset.collater)
        dictionaries = train_dataset.dictionaries
        valid_dataset = None
        valid_dl = None
        if valid_df_path is not None:
            valid_dataset = TransformersDataset.create(
                model_name,
                df_path=valid_df_path,
                dictionaries=dictionaries,
                dictionaries_path=dictionaries_path,
                tokenizer_cls=tokenizer_cls,
                tokenizer_args=tokenizer_args,
                max_tokens=max_tokens,
                clear_cache=clear_cache,
                online=online,
                cache_dir=cache_dir,
                pad_idx=pad_idx,
                markup=markup
            )
            assert len(valid_dataset)
            if len(valid_dataset):
                valid_dl = DataLoader(
                    valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=valid_dataset.collater)
        test_dataset = None
        test_dl = None
        if test_df_path is not None:
            test_dataset = TransformersDataset.create(
                model_name,
                df_path=test_df_path,
                dictionaries=dictionaries,
                dictionaries_path=dictionaries_path,
                tokenizer_cls=tokenizer_cls,
                tokenizer_args=tokenizer_args,
                max_tokens=max_tokens,
                clear_cache=clear_cache,
                online=online,
                cache_dir=cache_dir,
                pad_idx=pad_idx,
                markup=markup
            )
            assert len(test_dataset)
            if len(test_dataset):
                test_dl = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collater)
        datasets = {
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset
        }
        dataloaders = {
            "train": train_dl,
            "valid": valid_dl,
            "test": test_dl
        }
        return cls(datasets, dataloaders, dictionaries)

    def __init__(self, datasets, dataloaders, dictionaries):
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.dictionaries = dictionaries
