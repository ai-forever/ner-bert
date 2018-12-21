from torch import nn
import torch


class BertBiLSTMEncoder(nn.Module):

    def __init__(self, embeddings,
                 hidden_dim=128, rnn_layers=1, use_cuda=True):
        super(BertBiLSTMEncoder, self).__init__()
        self.embeddings = embeddings
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(
            self.embeddings.embedding_dim, hidden_dim // 2,
            rnn_layers, batch_first=True, bidirectional=True)
        self.hidden = None
        if use_cuda:
            self.cuda()
        self.init_weights()
        self.output_dim = hidden_dim

    def init_weights(self):
        # for p in self.lstm.parameters():
        #    nn.init.xavier_normal(p)
        pass

    def forward(self, batch):
        input, input_mask = batch[0], batch[1]
        output = self.embeddings(*batch)
        # output = self.dropout(output)
        lens = input_mask.sum(-1)
        output = nn.utils.rnn.pack_padded_sequence(
            output, lens.tolist(), batch_first=True)
        output, self.hidden = self.lstm(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
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
    def create(cls, embeddings, hidden_dim=128, rnn_layers=1, use_cuda=True):
        model = cls(
            embeddings=embeddings, hidden_dim=hidden_dim, rnn_layers=rnn_layers, use_cuda=use_cuda)
        return model


class ElmoBiLSTMEncoder(nn.Module):

    def __init__(self, embeddings,
                 hidden_dim=128, rnn_layers=1, use_cuda=True):
        super(ElmoBiLSTMEncoder, self).__init__()
        self.embeddings = embeddings
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(
            self.embeddings.embedding_dim, hidden_dim // 2,
            rnn_layers, batch_first=True, bidirectional=True)
        self.hidden = None
        if use_cuda:
            self.cuda()
        self.init_weights()
        self.output_dim = hidden_dim

    def init_weights(self):
        # for p in self.lstm.parameters():
        #    nn.init.xavier_normal(p)
        pass

    def forward(self, batch):
        input, input_mask = batch[0], batch[-2]
        output = self.embeddings(*batch)
        # output = self.dropout(output)
        lens = input_mask.sum(-1)
        output = nn.utils.rnn.pack_padded_sequence(
            output, lens.tolist(), batch_first=True)
        output, self.hidden = self.lstm(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
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
    def create(cls, embeddings, hidden_dim=128, rnn_layers=1, use_cuda=True):
        model = cls(
            embeddings=embeddings, hidden_dim=hidden_dim, rnn_layers=rnn_layers, use_cuda=use_cuda)
        return model


class BertMetaBiLSTMEncoder(nn.Module):

    def __init__(self, embeddings, meta_dim,
                 hidden_dim=128, rnn_layers=1, use_cuda=True):
        super(BertMetaBiLSTMEncoder, self).__init__()
        self.embeddings = embeddings
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(
            self.embeddings.embedding_dim, hidden_dim // 2,
            rnn_layers, batch_first=True, bidirectional=True)
        self.hidden = None
        if use_cuda:
            self.cuda()
        self.init_weights()
        self.meta_dim = meta_dim
        self.output_dim = hidden_dim + meta_dim

    def init_weights(self):
        # for p in self.lstm.parameters():
        #    nn.init.xavier_normal(p)
        pass

    def forward(self, batch):
        input, input_mask = batch[0], batch[1]
        output = torch.cat((self.embeddings(*batch), batch[3]), axis=-1)
        # output = self.dropout(output)
        lens = input_mask.sum(-1)
        output = nn.utils.rnn.pack_padded_sequence(
            output, lens.tolist(), batch_first=True)
        output, self.hidden = self.lstm(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
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
    def create(cls, embeddings, meta_dim, hidden_dim=128, rnn_layers=1, use_cuda=True):
        model = cls(
            embeddings=embeddings, meta_dim=meta_dim,
            idden_dim=hidden_dim, rnn_layers=rnn_layers, use_cuda=use_cuda)
        return model
