from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from modules import tokenization
import torch
import pandas as pd


id2label = ["<pad>", "[CLS]", "[SEP]", 'B_O', 'I_O', 'B_LOC', 'I_LOC', 'B_ORG', 'I_ORG', 'B_PER', 'I_PER']
label2idx = {l: idx for idx, l in enumerate(id2label)}


class BertDataSet(Dataset):

    def __init__(self, data):
        """
        Создает pytorch data set.

        Parameters
        -----------
        transforms : list | Nine
            Список трансформаций. Не добавлять ToTensor (применяется автоматически).
        """
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

    def __init__(self, data_set, cuda, **kwargs):
        super(DataLoaderHelper, self).__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            **kwargs
        )
        self.cuda = cuda

    def collate_fn(self, data):
        input_ids = []
        input_mask = []
        input_type_ids = []
        labels_ids = []
        lens = list(map(lambda x: len(x.tokens), data))
        max_len = max(lens)
        for f in data:
            input_ids.append(f.input_ids[:max_len])
            input_mask.append(f.input_mask[:max_len])
            input_type_ids.append(f.input_type_ids[:max_len])
            labels_ids.append(f.labels_ids[:max_len])
        if self.cuda:
            return (torch.LongTensor(input_ids).cuda(),
                    torch.LongTensor(input_mask).cuda(),
                    torch.LongTensor(input_type_ids).cuda(),
                    torch.LongTensor(labels_ids).cuda())
        return (torch.LongTensor(input_ids),
                torch.LongTensor(input_mask),
                torch.LongTensor(input_type_ids),
                torch.LongTensor(labels_ids))


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, labels, labels_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.labels = labels
        self.labels_ids = labels_ids


def get_bert_data(df, tokenizer, label2idx, max_seq_len=424, pad="<pad>"):
    orig_to_tok_map = []
    features = []
    for idx, (text, labels) in enumerate(zip(df["1"].tolist(), df["0"].tolist())):
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
        features.append(InputFeatures(
            unique_id=idx, 
            tokens=bert_tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            labels=bert_labels,
            labels_ids=bert_labels_ids))
        orig_to_tok_map.append(tok_map)
    return features, orig_to_tok_map, label2idx


def get_data_loaders(train, valid, vocab_file, label2idx=label2idx, batch_size=16, cuda=True):
    train = pd.read_csv(train)
    valid = pd.read_csv(valid)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    train_f, train_orig_to_tok_map, label2idx = get_bert_data(train, tokenizer, label2idx)
    dl_train = DataLoaderHelper(train_f, batch_size=batch_size, shuffle=True, cuda=cuda)
    valid_f, valid_orig_to_tok_map, label2idx = get_bert_data(valid, tokenizer, label2idx)
    dl_valid = DataLoaderHelper(valid_f, batch_size=batch_size, cuda=cuda)
    return dl_train, train_orig_to_tok_map, dl_valid, valid_orig_to_tok_map, tokenizer, label2idx


def get_data_loader_for_predict(path, learner):
    df = pd.read_csv(path)
    tokenizer = learner.data.tokenizer
    f, orig_to_tok_map, _ = get_bert_data(df, learner.data.tokenizer, learner.data.label2idx)
    dl = DataLoaderHelper(f, batch_size=learner.data.batch_size, shuffle=False, cuda=learner.model.use_cuda)

    return dl, orig_to_tok_map


class NerData(object):

    def __init__(self, dl_train, train_orig_to_tok_map, dl_valid, valid_orig_to_tok_map,
                 tokenizer, label2idx=label2idx, batch_size=16, cuda=True):
        self.dl_train = dl_train
        self.train_orig_to_tok_map = train_orig_to_tok_map
        self.valid_orig_to_tok_map = valid_orig_to_tok_map
        self.dl_valid = dl_valid
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.batch_size = batch_size
        self.cuda = cuda
        self.id2label = sorted(label2idx.keys(), key=lambda x: label2idx[x])

    @staticmethod
    def create(train_path, valid_path, vocab_file, label2idx=label2idx, batch_size=16, cuda=True):

        return NerData(*get_data_loaders(train_path, valid_path, vocab_file, label2idx), batch_size, cuda)
