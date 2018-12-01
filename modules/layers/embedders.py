from modules.layers import modeling
import torch
from torch import nn
from gensim.models import KeyedVectors
from collections import defaultdict


class BertEmbedder(nn.Module):

    def __init__(self, model, embedding_dim=768, use_cuda=True, bert_mode="weighted"):
        super(BertEmbedder, self).__init__()
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
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        model = modeling.BertModel(bert_config)
        if use_cuda:
            device = torch.device("cuda")
            map_location = "cuda"
        else:
            map_location = "cpu"
            device = torch.device("cpu")
        model.load_state_dict(torch.load(init_checkpoint_pt, map_location=map_location))
        model = model.to(device)
        model = cls(model=model, embedding_dim=embedding_dim, use_cuda=use_cuda, bert_mode=bert_mode)
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
