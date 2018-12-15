import numpy as np
import codecs
import json
from torch.utils.data import DataLoader
import os
import torch
import pandas as pd
from modules.utils.utils import ipython_info
from tqdm._tqdm_notebook import tqdm_notebook
from tqdm import tqdm


def read_list(sents, max_chars=None):
    """
    read raw text file. The format of the input is like, one sentence per line
    words are separated by '\t'
    :param sents:
    :param max_chars: int, the number of maximum characters in a word, this
      parameter is used when the model is configured with CNN word encoder.
    :return:
    """
    dataset = []
    textset = []
    for sent in sents:
        data = ['<bos>']
        text = []
        for token in sent:
            text.append(token)
            if max_chars is not None and len(token) + 2 > max_chars:
                token = token[:max_chars - 2]
            data.append(token)
        data.append('<eos>')
        dataset.append(data)
        textset.append(text)
    return dataset, textset


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            # Elmo data
            input_ids, char_ids,
            # Origin data
            tokens, labels, labels_ids, cls=None, cls_idx=None):
        """
        Data has the following structure.
        data[0]: list, tokens ids
        ...
        data[-1]: list, labels ids
        """
        self.data = []
        # Elmo data
        self.input_ids = input_ids
        self.data.append(input_ids)

        # Origin data
        self.tokens = tokens
        self.labels = labels
        self.char_ids = char_ids
        if char_ids is not None:
            self.data.append(char_ids)
        # Used for joint model
        self.cls = cls
        self.cls_idx = cls_idx
        if cls is not None:
            self.data.append(cls_idx)
        # Labels data

        self.labels_ids = labels_ids
        self.data.append(labels_ids)


class DataLoaderForTrain(DataLoader):

    def __init__(self, data_set, w_pad_id, c_pad_id, max_chars, cuda, **kwargs):
        super(DataLoaderForTrain, self).__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            **kwargs
        )
        self.w_pad_id = w_pad_id
        self.c_pad_id = c_pad_id
        self.l_pad_id = 0
        self.max_chars = max_chars
        self.cuda = cuda

    def collate_fn(self, data):
        batch_size = len(data)
        lens = [len(x.labels) for x in data]
        max_len = max(lens)
        sorted_idx = np.argsort(lens)[::-1]
        # Words prc
        batch_w = None
        if data[0].input_ids is not None:
            batch_w = torch.LongTensor(batch_size, max_len).fill_(self.w_pad_id)
            for i, idx in enumerate(sorted_idx):
                x_i = data[idx].input_ids
                for j, x_ij in enumerate(x_i):
                    batch_w[i][j] = x_ij
            if self.cuda:
                batch_w = batch_w.cuda()
        # Chars prc
        batch_c = None
        if data[0].char_ids is not None:
            batch_c = torch.LongTensor(batch_size, max_len, self.max_chars).fill_(self.c_pad_id)
            for i, idx in enumerate(sorted_idx):
                x_i = data[idx].char_ids
                for j, x_ij in enumerate(x_i):
                    for k, c in enumerate(x_ij):
                        batch_c[i][j][k] = c
            if self.cuda:
                batch_c = batch_c.cuda()

        # Masks prc
        masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]

        for i, idx in enumerate(sorted_idx):
            x_i = data[idx].input_ids
            for j in range(len(x_i)):
                masks[0][i][j] = 1
                if j + 1 < len(x_i):
                    masks[1].append(i * max_len + j)
                if j > 0:
                    masks[2].append(i * max_len + j)

        assert len(masks[1]) <= batch_size * max_len
        assert len(masks[2]) <= batch_size * max_len

        masks[1] = torch.LongTensor(masks[1])
        masks[2] = torch.LongTensor(masks[2])
        if self.cuda:
            masks[0] = masks[0].cuda()
            masks[1] = masks[1].cuda()
            masks[2] = masks[2].cuda()

        # Labels prc
        batch_l = torch.LongTensor(batch_size, max_len).fill_(self.l_pad_id)
        for i, idx in enumerate(sorted_idx):
            x_i = data[idx].labels_ids
            for j, x_ij in enumerate(x_i):
                batch_l[i][j] = x_ij
        if self.cuda:
            batch_l = batch_l.cuda()

        if data[0].cls_idx is not None:
            batch_cls = torch.LongTensor([data[idx].cls_idx for idx in sorted_idx])
            if self.cuda:
                batch_cls = batch_cls.cuda()
            return batch_w, batch_c, masks, batch_cls, masks[0], batch_l
        return batch_w, batch_c, masks, masks[0], batch_l


def get_data(df, config, label2idx=None, oov='<oov>', pad='<pad>', cls2idx=None, is_cls=False,
             word_lexicon=None, char_lexicon=None, max_seq_len=424):
    if label2idx is None:
        label2idx = {pad: 0, '<bos>': 1, '<eos>': 2}
    features = []
    if is_cls:
        # Use joint model
        if cls2idx is None:
            cls2idx = dict()
        zip_args = zip(df["1"].tolist(), df["0"].tolist(), df["2"].tolist())
    else:
        zip_args = zip(df["1"].tolist(), df["0"].tolist())
    cls = None
    total = len(df["0"].tolist())
    for args in tqdm_notebook(enumerate(zip_args), total=total, leave=False):
        if is_cls:
            idx, (text, labels, cls) = args
        else:
            idx, (text, labels) = args
        text = text.split()
        text = text[:max_seq_len - 2]
        labels = labels.split()[:max_seq_len - 2]
        labels = ['<bos>'] + labels + ['<eos>']
        if config['token_embedder']['name'].lower() == 'cnn':
            tokens, text = read_list([text], config['token_embedder']['max_characters_per_token'])
        else:
            tokens, text = read_list([text])
        tokens, text = tokens[0], text[0]
        input_ids = None
        if word_lexicon is not None:
            oov_id, pad_id = word_lexicon.get(oov, None), word_lexicon.get(pad, None)
            assert oov_id is not None and pad_id is not None
            input_ids = [word_lexicon.get(x, oov_id) for x in tokens]
        char_ids = None
        # get a batch of character id whose size is (batch x max_len x max_chars)
        if char_lexicon is not None:
            char_ids = []
            bow_id, eow_id, oov_id, pad_id = [char_lexicon.get(key, None) for key in ('<eow>', '<bow>', oov, pad)]

            assert bow_id is not None and eow_id is not None and oov_id is not None and pad_id is not None

            if config['token_embedder']['name'].lower() == 'cnn':
                max_chars = config['token_embedder']['max_characters_per_token']
                assert max([len(w) for w in tokens]) + 2 <= max_chars
            elif config['token_embedder']['name'].lower() == 'lstm':
                # counting the <bow> and <eow>
                pass
            else:
                raise ValueError('Unknown token_embedder: {0}'.format(config['token_embedder']['name']))
            for token in tokens:
                chars = [bow_id]
                if token == '<bos>' or token == '<eos>':
                    chars.append(char_lexicon.get(token))
                    chars.append(eow_id)
                else:
                    for c in token:
                        chars.append(char_lexicon.get(c, oov_id))
                    chars.append(eow_id)
                char_ids.append(chars)

        for l in labels:
            if l not in label2idx:
                label2idx[l] = len(label2idx)
        labels_ids = [label2idx[l] for l in labels]
        # For joint model
        cls_idx = None
        if is_cls:
            if cls not in cls2idx:
                cls2idx[cls] = len(cls2idx)
            cls_idx = cls2idx[cls]
        features.append(InputFeatures(input_ids, char_ids, tokens, labels, labels_ids, cls=cls, cls_idx=cls_idx))
    if is_cls:
        return features, (label2idx, cls2idx)
    return features, label2idx


def get_elmo_data_loaders(train, valid, model_dir, config_name, batch_size, cuda, is_cls,
                          oov='<oov>', pad='<pad>'):
    train = pd.read_csv(train)
    valid = pd.read_csv(valid)
    with open(os.path.join(model_dir, config_name), 'r') as fin:
        config = json.load(fin)
    c_pad_id = None
    char_lexicon = None
    # For the model trained with character-based word encoder.
    if config['token_embedder']['char_dim'] > 0:
        char_lexicon = {}
        with codecs.open(os.path.join(model_dir, 'char.dic'), 'r', encoding='utf-8') as fpi:
            for line in fpi:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')
                token, i = tokens
                char_lexicon[token] = int(i)
        c_pad_id = char_lexicon.get(pad)
    w_pad_id = None
    word_lexicon = None
    # For the model trained with word form word encoder.
    if config['token_embedder']['word_dim'] > 0:
        word_lexicon = {}
        with codecs.open(os.path.join(model_dir, 'word.dic'), 'r', encoding='utf-8') as fpi:
            for line in fpi:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')
                token, i = tokens
                word_lexicon[token] = int(i)
        w_pad_id = word_lexicon.get(pad)

    max_chars = None
    if config['token_embedder']['name'].lower() == 'cnn':
        max_chars = config['token_embedder']['max_characters_per_token']
    elif config['token_embedder']['name'].lower() == 'lstm':
        # counting the <bow> and <eow>
        pass
    else:
        raise ValueError('Unknown token_embedder: {0}'.format(config['token_embedder']['name']))

    # Get train dataset
    train_f, label2idx = get_data(
        train, config, oov=oov, pad=pad, is_cls=is_cls, word_lexicon=word_lexicon, char_lexicon=char_lexicon)
    cls2idx = None
    if is_cls:
        label2idx, cls2idx = label2idx
    # Get train dataloader
    train_dl = DataLoaderForTrain(
        train_f, w_pad_id, c_pad_id, max_chars, batch_size=batch_size, shuffle=True, cuda=cuda)

    # Get valid dataset
    valid_f, label2idx = get_data(
        valid, config, oov=oov, pad=pad, is_cls=is_cls, cls2idx=cls2idx,
        word_lexicon=word_lexicon, char_lexicon=char_lexicon)
    cls2idx = None
    if is_cls:
        label2idx, cls2idx = label2idx
    # Get valid dataloader
    valid_dl = DataLoaderForTrain(
        valid_f, w_pad_id, c_pad_id, max_chars, batch_size=batch_size, shuffle=False, cuda=cuda)
    if is_cls:
        return train_dl, valid_dl, label2idx, word_lexicon, char_lexicon, cls2idx
    return train_dl, valid_dl, label2idx, word_lexicon, char_lexicon


class DataLoaderForPredict(DataLoader):

    def __init__(self, data_set, w_pad_id, c_pad_id, max_chars, cuda, **kwargs):
        super(DataLoaderForPredict, self).__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            **kwargs
        )
        self.w_pad_id = w_pad_id
        self.c_pad_id = c_pad_id
        self.l_pad_id = 0
        self.max_chars = max_chars
        self.cuda = cuda

    def collate_fn(self, data):
        batch_size = len(data)
        lens = [len(x.labels) for x in data]
        max_len = max(lens)
        sorted_idx = np.argsort(lens)[::-1]
        # Words prc
        batch_w = None
        if data[0].input_ids is not None:
            batch_w = torch.LongTensor(batch_size, max_len).fill_(self.w_pad_id)
            for i, idx in enumerate(sorted_idx):
                x_i = data[idx].input_ids
                for j, x_ij in enumerate(x_i):
                    batch_w[i][j] = x_ij
            if self.cuda:
                batch_w = batch_w.cuda()
        # Chars prc
        batch_c = None
        if data[0].char_ids is not None:
            batch_c = torch.LongTensor(batch_size, max_len, self.max_chars).fill_(self.c_pad_id)
            for i, idx in enumerate(sorted_idx):
                x_i = data[idx].char_ids
                for j, x_ij in enumerate(x_i):
                    for k, c in enumerate(x_ij):
                        batch_c[i][j][k] = c
            if self.cuda:
                batch_c = batch_c.cuda()

        # Masks prc
        masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]

        for i, idx in enumerate(sorted_idx):
            x_i = data[idx].input_ids
            for j in range(len(x_i)):
                masks[0][i][j] = 1
                if j + 1 < len(x_i):
                    masks[1].append(i * max_len + j)
                if j > 0:
                    masks[2].append(i * max_len + j)

        assert len(masks[1]) <= batch_size * max_len
        assert len(masks[2]) <= batch_size * max_len

        masks[1] = torch.LongTensor(masks[1])
        masks[2] = torch.LongTensor(masks[2])
        if self.cuda:
            masks[0] = masks[0].cuda()
            masks[1] = masks[1].cuda()
            masks[2] = masks[2].cuda()

        # Labels prc
        batch_l = torch.LongTensor(batch_size, max_len).fill_(self.l_pad_id)
        for i, idx in enumerate(sorted_idx):
            x_i = data[idx].labels_ids
            for j, x_ij in enumerate(x_i):
                batch_l[i][j] = x_ij
        sorted_idx = torch.LongTensor(list(sorted_idx))
        if self.cuda:
            batch_l = batch_l.cuda()
            sorted_idx = sorted_idx.cuda()

        if data[0].cls_idx is not None:
            batch_cls = torch.LongTensor([data[idx].cls_idx for idx in sorted_idx])
            if self.cuda:
                batch_cls = batch_cls.cuda()
            return batch_w, batch_c, masks, batch_cls, masks[0], batch_l
        return (batch_w, batch_c, masks, masks[0], batch_l), sorted_idx


def get_elmo_data_loader_for_predict(
        valid, learner, oov='<oov>', pad='<pad>'):
    valid = pd.read_csv(valid)
    c_pad_id = None
    char_lexicon = learner.data.char2idx
    # For the model trained with character-based word encoder.
    if char_lexicon is not None:
        c_pad_id = char_lexicon.get(pad)
    w_pad_id = None
    word_lexicon = learner.data.word2idx
    # For the model trained with word form word encoder.
    if word_lexicon is not None:
        w_pad_id = word_lexicon.get(pad)

    max_chars = learner.data.train_dl.max_chars
    cls2idx = learner.data.cls2idx
    config = learner.model.encoder.embeddings.config
    is_cls = learner.data.is_cls
    cuda = learner.data.cuda
    batch_size = learner.data.batch_size

    # Get valid dataset
    valid_f, label2idx = get_data(
        valid, config, oov=oov, pad=pad, is_cls=is_cls, cls2idx=cls2idx,
        word_lexicon=word_lexicon, char_lexicon=char_lexicon)
    # Get valid dataloader
    valid_dl = DataLoaderForPredict(
        valid_f, w_pad_id, c_pad_id, max_chars, batch_size=batch_size, shuffle=False, cuda=cuda)
    return valid_dl


class ElmoNerData(object):

    def __init__(self, train_dl, valid_dl, label2idx,
                 word2idx=None, char2idx=None,
                 cls2idx=None, batch_size=16, cuda=True):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.label2idx = label2idx
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.cls2idx = cls2idx
        self.batch_size = batch_size
        self.cuda = cuda
        self.id2label = sorted(label2idx.keys(), key=lambda x: label2idx[x])
        if word2idx is not None:
            self.idx2word = sorted(word2idx.keys(), key=lambda x: word2idx[x])
        if char2idx is not None:
            self.idx2char = sorted(char2idx.keys(), key=lambda x: char2idx[x])
        self.is_cls = False
        if cls2idx is not None:
            self.is_cls = True
            self.id2cls = sorted(cls2idx.keys(), key=lambda x: cls2idx[x])

    @classmethod
    def create(cls,
               train_path, valid_path, model_dir, config_name, batch_size=16, cuda=True, is_cls=False,
               oov='<oov>', pad='<pad>'):
        if ipython_info():
            global tqdm_notebook
            tqdm_notebook = tqdm
        fn = get_elmo_data_loaders
        return cls(*fn(
            train_path, valid_path, model_dir, config_name, batch_size, cuda, is_cls, oov, pad),
                   batch_size=batch_size, cuda=cuda)
