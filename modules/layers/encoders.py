from torch import nn
import torch


class BertMetaBiLSTMEncoder(nn.Module):

    def __init__(self, embeddings, meta_embeddings=None,
                 hidden_dim=128, rnn_layers=1, dropout=0.5, use_cuda=True):
        super(BertMetaBiLSTMEncoder, self).__init__()
        self.embeddings = embeddings
        self.meta_embeddings = meta_embeddings
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.use_cuda = use_cuda
        self.dropout = nn.Dropout(dropout)
        meta_dim = 0
        if self.meta_embeddings:
            meta_dim = meta_embeddings.embedding_dim
        self.lstm = nn.LSTM(
            self.embeddings.embedding_dim + meta_dim,
            hidden_dim // 2,
            rnn_layers, batch_first=True, bidirectional=True)
        if use_cuda:
            self.cuda()
        self.init_weights()
        self.output_dim = hidden_dim
        self.hidden = None

    def init_weights(self):
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def forward(self, batch):
        input_mask = batch[1]
        output = self.embeddings(*batch)
        if self.meta_embeddings:
            output = torch.cat((output, self.meta_embeddings(*batch)), dim=-1)
        output = self.dropout(output)
        lens = input_mask.sum(-1)
        output = nn.utils.rnn.pack_padded_sequence(
            output, lens.tolist(), batch_first=True)
        output, self.hidden = self.lstm(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = torch.cat((output, batch[3]), dim=-1)
        return output, self.hidden

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
    def create(cls, embeddings, meta_embeddings=None,
               hidden_dim=128, rnn_layers=1, dropout=0.5, use_cuda=True):
        model = cls(
            embeddings, meta_embeddings, hidden_dim, rnn_layers, dropout, use_cuda=use_cuda)
        return model
