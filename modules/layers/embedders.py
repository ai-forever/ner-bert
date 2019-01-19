from modules.layers import bert_modeling
import torch
from gensim.models import KeyedVectors
import os
import codecs
import logging
import json
from torch import nn
from elmoformanylangs.modules.embedding_layer import EmbeddingLayer
from elmoformanylangs.frontend import Model


# TODO: add from_config to other embedders
class BertEmbedder(nn.Module):

    # @property
    def get_config(self):
        config = {
            "name": "BertEmbedder",
            "params": {
                "bert_config_file": self.bert_config_file,
                "init_checkpoint_pt": self.init_checkpoint_pt,
                "freeze": self.is_freeze,
                "embedding_dim": self.embedding_dim,
                "use_cuda": self.use_cuda,
                "bert_mode": self.bert_mode
            }
        }
        return config

    def __init__(self, model, bert_config_file, init_checkpoint_pt,
                 freeze=True, embedding_dim=768, use_cuda=True, bert_mode="weighted",):
        super(BertEmbedder, self).__init__()
        self.bert_config_file = bert_config_file
        self.init_checkpoint_pt = init_checkpoint_pt
        self.is_freeze = freeze
        self.embedding_dim = embedding_dim
        self.model = model
        self.use_cuda = use_cuda
        self.bert_mode = bert_mode
        if self.bert_mode == "weighted":
            self.bert_weights = nn.Parameter(torch.FloatTensor(12, 1))
            self.bert_gamma = nn.Parameter(torch.FloatTensor(1, 1))

        if use_cuda:
            self.cuda()

        self.init_weights()

    @classmethod
    def from_config(cls, config):
        return cls.create(**config)

    def init_weights(self):
        if self.bert_mode == "weighted":
            nn.init.xavier_normal(self.bert_gamma)
            nn.init.xavier_normal(self.bert_weights)

    def forward(self, *batch):
        input_ids, input_mask, input_type_ids = batch[:3]
        all_encoder_layers, _ = self.model(input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        if self.bert_mode == "last":
            return all_encoder_layers[-1]
        elif self.bert_mode == "weighted":
            all_encoder_layers = torch.stack([a * b for a, b in zip(all_encoder_layers, self.bert_weights)])
            return self.bert_gamma * torch.sum(all_encoder_layers, dim=0)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_to(self, to=-1):
        idx = 0
        if to < 0:
            to = len(self.model.encoder.layer) + to + 1
        for idx in range(to):
            for param in self.model.encoder.layer[idx].parameters():
                param.requires_grad = False
        print("Embeddings freezed to {}".format(to))
        to = len(self.model.encoder.layer)
        for idx in range(idx, to):
            for param in self.model.encoder.layer[idx].parameters():
                param.requires_grad = True

    def get_n_trainable_params(self):
        pp = 0
        for p in list(self.parameters()):
            if p.requires_grad:
                num = 1
                for s in list(p.size()):
                    num = num * s
                pp += num
        return pp

    @classmethod
    def create(cls,
               bert_config_file, init_checkpoint_pt, embedding_dim=768, use_cuda=True, bert_mode="weighted",
               freeze=True):
        bert_config = bert_modeling.BertConfig.from_json_file(bert_config_file)
        model = bert_modeling.BertModel(bert_config)
        if use_cuda:
            device = torch.device("cuda")
            map_location = "cuda"
        else:
            map_location = "cpu"
            device = torch.device("cpu")
        model.load_state_dict(torch.load(init_checkpoint_pt, map_location=map_location))
        model = model.to(device)
        model = cls(model=model, embedding_dim=embedding_dim, use_cuda=use_cuda, bert_mode=bert_mode,
                    bert_config_file=bert_config_file, init_checkpoint_pt=init_checkpoint_pt, freeze=freeze)
        if freeze:
            model.freeze()
        return model


class Word2VecEmbedder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim=300,
                 padding_idx=0,
                 trainable=True,
                 normalize=True):
        super(Word2VecEmbedder, self).__init__()
        self.pad_id = padding_idx
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.model = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad_id)

        self.trainable = trainable
        self.normalize = normalize

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2, 1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

        if trainable:
            self.embedding.weight.requires_grad = True
        else:
            self.embedding.weight.requires_grad = False
        self.loaded = False
        self.path = None
        self.init_weights()

    def init_weights(self):
        for p in self.embedding.parameters():
            torch.nn.init.xavier_normal(p)

    def forward(self, *batch):
        input_ids = batch[0]
        return self.model(input_ids)

    def load_gensim_word2vec(self, path, words, binary=False):
        self.loaded = True
        word_vectors = KeyedVectors.load_word2vec_format(path, binary=binary)
        for word, idx in words.items():
            if word in word_vectors:
                if idx < self.vocab_size:
                    self.embedding.weight.data[idx].set_(torch.FloatTensor(word_vectors[word]))
        return self

    @classmethod
    def create(cls, path, words, binary=False, embedding_dim=300, padding_idx=0, trainable=True, normalize=True):
        model = cls(
            vocab_size=len(words), embedding_dim=embedding_dim, padding_idx=padding_idx,
            trainable=trainable, normalize=normalize)
        model = model.load_gensim_word2vec(path, words, binary)
        return model


class ElmoEmbedder(nn.Module):
    def __init__(self, model, config, embedding_dim=1024, use_cuda=True, elmo_mode="avg"):
        super(ElmoEmbedder, self).__init__()
        self.model = model
        self.embedding_dim = embedding_dim
        self.model = model
        self.use_cuda = use_cuda
        self.config = config
        self.elmo_mode = elmo_mode

        if self.elmo_mode == "weighted":
            self.elmo_weights = nn.Parameter(torch.FloatTensor(3, 1))
            self.elmo_gamma = nn.Parameter(torch.FloatTensor(1, 1))

        if use_cuda:
            self.cuda()

        self.init_weights()

    def init_weights(self):
        if self.elmo_mode == "weighted":
            nn.init.xavier_normal(self.elmo_weights)
            nn.init.xavier_normal(self.elmo_gamma)

    def forward(self, *batch):
        w, c, masks = batch[:3]
        all_encoder_layers = self.model.forward(w, c, masks)
        if self.elmo_mode == "avg":
            return all_encoder_layers.mean(0)
        elif self.bert_mode == "weighted":
            all_encoder_layers = torch.stack([a * b for a, b in zip(all_encoder_layers, self.elmo_weights)])
            return self.elmo_gamma * torch.sum(all_encoder_layers, dim=0)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    @classmethod
    def create(
            cls, model_dir, config_name, embedding_dim=1024, use_cuda=True, elmo_mode="avg", freeze=True):
        with open(os.path.join(model_dir, config_name), 'r') as fin:
            config = json.load(fin)
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
            char_emb_layer = EmbeddingLayer(
                config['token_embedder']['char_dim'], char_lexicon, fix_emb=False, embs=None)
            logging.info('char embedding size: ' +
                         str(len(char_emb_layer.word2id)))
        else:
            char_emb_layer = None

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
            word_emb_layer = EmbeddingLayer(
                config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=None)
            logging.info('word embedding size: ' +
                         str(len(word_emb_layer.word2id)))
        else:
            word_emb_layer = None

        # instantiate the model
        model = Model(config, word_emb_layer, char_emb_layer, use_cuda)

        model.load_model(model_dir)

        model.eval()
        model = cls(model, config, embedding_dim, use_cuda, elmo_mode=elmo_mode)
        if freeze:
            model.freeze()
        return model
