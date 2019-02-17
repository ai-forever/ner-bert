from modules.layers import bert_modeling
import torch
from torch import nn


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
        self.model.eval()

    def unfreeze(self):
        self.model.train()

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
               bert_config_file, init_checkpoint_pt, embedding_dim=768, use_cuda=True,
               bert_mode="weighted", freeze=True):
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
