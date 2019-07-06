import torch
from torch.nn import functional
from torch.autograd import Variable
from torch import nn
from .layers import Linears
from .crf import CRF
from .ncrf import NCRF


class CRFDecoder(nn.Module):
    def __init__(self, crf, label_size, input_dim, input_dropout=0.5):
        super(CRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.linear = Linears(in_features=input_dim,
                              out_features=label_size,
                              hiddens=[input_dim // 2])
        self.crf = crf
        self.label_size = label_size

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        output = inputs.contiguous().view(-1, self.input_dim)
        output = self.input_dropout(output)
        # Fully-connected layer
        output = self.linear.forward(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, labels_mask):
        lens = labels_mask.sum(-1)
        logits = self.forward_model(inputs)
        logits = self.crf.pad_logits(logits)
        scores, preds = self.crf.viterbi_decode(logits, lens)
        return preds

    def score(self, inputs, labels_mask, labels):
        lens = labels_mask.sum(-1)
        logits = self.forward_model(inputs)
        logits = self.crf.pad_logits(logits)
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score
        return -loglik.mean()

    @classmethod
    def create(cls, label_size, input_dim, input_dropout=0.5):
        return cls(CRF(label_size + 2), label_size, input_dim, input_dropout)


class NMTDecoder(nn.Module):
    def __init__(self,
                 label_size,
                 embedding_dim=64, hidden_dim=256, rnn_layers=1,
                 dropout_p=0.1, pad_idx=0):
        super(NMTDecoder, self).__init__()
        self.slot_size = label_size
        self.pad_idx = pad_idx
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(self.slot_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim + self.hidden_dim * 2,
                            self.hidden_dim, self.rnn_layers,
                            batch_first=True)
        self.attn = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.slot_out = nn.Linear(self.hidden_dim * 2, self.slot_size)

        self.loss = nn.CrossEntropyLoss(ignore_index=pad_idx)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal(self.embedding.weight)
        nn.init.xavier_normal(self.attn.weight)
        nn.init.xavier_normal(self.slot_out.weight)

    def attention(self, hidden, encoder_outputs, input_mask):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        input_mask : B,T # ByteTensor
        """
        input_mask = input_mask == 0
        hidden = hidden.squeeze(0).unsqueeze(2)

        # B
        batch_size = encoder_outputs.size(0)
        # T
        max_len = encoder_outputs.size(1)
        # B*T,D -> B*T,D
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1))
        energies = energies.view(batch_size, max_len, -1)
        # B,T,D * B,D,1 --> B,1,T
        attn_energies = energies.bmm(hidden).transpose(1, 2)
        # PAD masking
        attn_energies = attn_energies.squeeze(1).masked_fill(input_mask, -1e12)

        # B,T
        alpha = functional.softmax(attn_energies)
        # B,1,T
        alpha = alpha.unsqueeze(1)
        # B,1,T * B,T,D => B,1,D
        context = alpha.bmm(encoder_outputs)
        # B,1,D
        return context

    def forward_model(self, encoder_outputs, input_mask):
        real_context = []

        for idx, o in enumerate(encoder_outputs):
            real_length = input_mask[idx].sum().cpu().data.tolist()
            real_context.append(o[real_length - 1])
        context = torch.cat(real_context).view(encoder_outputs.size(0), -1).unsqueeze(1)

        batch_size = encoder_outputs.size(0)

        input_mask = input_mask == 0
        # Get the embedding of the current input word

        embedded = Variable(torch.zeros(batch_size, self.embedding_dim))
        if self.use_cuda:
            embedded = embedded.cuda()
        embedded = embedded.unsqueeze(1)
        decode = []
        aligns = encoder_outputs.transpose(0, 1)
        length = encoder_outputs.size(1)
        for i in range(length):
            # B,1,D
            aligned = aligns[i].unsqueeze(1)
            # input, context, aligned encoder hidden, hidden
            _, hidden = self.lstm(torch.cat((embedded, context, aligned), 2))

            # print(hidden[0].shape, context.transpose(0, 1).shape)

            concated = torch.cat((hidden[0], context.transpose(0, 1)), 2)
            score = self.slot_out(concated.squeeze(0))
            softmaxed = functional.log_softmax(score)
            decode.append(softmaxed)
            _, input = torch.max(softmaxed, 1)
            embedded = self.embedding(input.unsqueeze(1))

            context = self.attention(hidden[0], encoder_outputs, input_mask)
        slot_scores = torch.cat(decode, 1)

        return slot_scores.view(batch_size, length, -1)

    def forward(self, encoder_outputs, input_mask):
        scores = self.forward_model(encoder_outputs, input_mask)
        return scores.argmax(-1)

    def score(self, encoder_outputs, input_mask, labels_ids):
        scores = self.forward_model(encoder_outputs, input_mask)
        batch_size = encoder_outputs.shape[0]
        len_ = encoder_outputs.shape[1]
        return self.loss(scores.view(batch_size * len_, -1), labels_ids.view(-1))

    @classmethod
    def create(cls, label_size,
               embedding_dim=64, hidden_dim=256, rnn_layers=1, dropout_p=0.1, pad_idx=0):
        return cls(label_size=label_size,
                   embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                   rnn_layers=rnn_layers, dropout_p=dropout_p, pad_idx=pad_idx)


class PoolingLinearClassifier(nn.Module):
    """Create a linear classifier with pooling."""

    def __init__(self, input_dim, intent_size, input_dropout=0.5):
        super(PoolingLinearClassifier, self).__init__()
        self.input_dim = input_dim
        self.intent_size = intent_size
        self.input_dropout = input_dropout
        self.dropout = nn.Dropout(p=input_dropout)
        self.linear = Linears(input_dim * 3, intent_size, [input_dim // 2], activation="relu")

    @staticmethod
    def pool(x, bs, is_max):
        """Pool the tensor along the seq_len dimension."""
        f = functional.adaptive_max_pool1d if is_max else functional.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(bs, -1)

    def forward(self, output):
        output = self.dropout(output).transpose(0, 1)
        sl, bs, _ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        return self.linear(x)


class NMTJointDecoder(nn.Module):
    def __init__(self,
                 label_size, intent_size,
                 embedding_dim=64, hidden_dim=256, rnn_layers=1,
                 dropout_p=0.1, pad_idx=0):
        super(NMTJointDecoder, self).__init__()
        self.slot_size = label_size
        self.intent_size = intent_size
        self.pad_idx = pad_idx
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(self.slot_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim + self.hidden_dim * 2,
                            self.hidden_dim, self.rnn_layers,
                            batch_first=True)
        self.attn = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.slot_out = nn.Linear(self.hidden_dim * 2, self.slot_size)

        self.loss = nn.CrossEntropyLoss(ignore_index=pad_idx)

        self.intent_loss = nn.CrossEntropyLoss()
        self.intent_out = Linears(
            in_features=self.hidden_dim * 2,
            out_features=self.intent_size,
            hiddens=[hidden_dim // 2],
            activation="relu")

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal(self.embedding.weight)
        nn.init.xavier_normal(self.attn.weight)
        nn.init.xavier_normal(self.slot_out.weight)

    def attention(self, hidden, encoder_outputs, input_mask):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        input_mask : B,T # ByteTensor
        """
        input_mask = input_mask == 0
        hidden = hidden.squeeze(0).unsqueeze(2)

        # B
        batch_size = encoder_outputs.size(0)
        # T
        max_len = encoder_outputs.size(1)
        # B*T,D -> B*T,D
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1))
        energies = energies.view(batch_size, max_len, -1)
        # B,T,D * B,D,1 --> B,1,T
        attn_energies = energies.bmm(hidden).transpose(1, 2)
        # PAD masking
        attn_energies = attn_energies.squeeze(1).masked_fill(input_mask, -1e12)

        # B,T
        alpha = functional.softmax(attn_energies)
        # B,1,T
        alpha = alpha.unsqueeze(1)
        # B,1,T * B,T,D => B,1,D
        context = alpha.bmm(encoder_outputs)
        # B,1,D
        return context

    def forward_model(self, encoder_outputs, input_mask):
        real_context = []

        for idx, o in enumerate(encoder_outputs):
            real_length = input_mask[idx].sum().cpu().data.tolist()
            real_context.append(o[real_length - 1])
        context = torch.cat(real_context).view(encoder_outputs.size(0), -1).unsqueeze(1)

        batch_size = encoder_outputs.size(0)

        input_mask = input_mask == 0
        # Get the embedding of the current input word

        embedded = Variable(torch.zeros(batch_size, self.embedding_dim))
        if self.use_cuda:
            embedded = embedded.cuda()
        embedded = embedded.unsqueeze(1)
        decode = []
        aligns = encoder_outputs.transpose(0, 1)
        length = encoder_outputs.size(1)
        intent_score = None
        for i in range(length):
            # B,1,D
            aligned = aligns[i].unsqueeze(1)
            # input, context, aligned encoder hidden, hidden
            _, hidden = self.lstm(torch.cat((embedded, context, aligned), 2))

            # for Intent Detection
            if i == 0:
                intent_hidden = hidden[0].clone()
                # 1,B,D
                intent_context = self.attention(intent_hidden, encoder_outputs, input_mask)
                concated = torch.cat((intent_hidden, intent_context.transpose(0, 1)), 2)
                # B,D
                intent_score = self.intent_out(concated.squeeze(0))

            concated = torch.cat((hidden[0], context.transpose(0, 1)), 2)
            score = self.slot_out(concated.squeeze(0))
            softmaxed = functional.log_softmax(score)
            decode.append(softmaxed)
            _, input = torch.max(softmaxed, 1)
            embedded = self.embedding(input.unsqueeze(1))

            context = self.attention(hidden[0], encoder_outputs, input_mask)
        slot_scores = torch.cat(decode, 1)

        return slot_scores.view(batch_size, length, -1), intent_score

    def forward(self, encoder_outputs, input_mask):
        scores, intent_score = self.forward_model(encoder_outputs, input_mask)
        return scores.argmax(-1), intent_score.argmax(-1)

    def score(self, encoder_outputs, input_mask, labels_ids, cls_ids):
        scores, intent_score = self.forward_model(encoder_outputs, input_mask)
        batch_size = encoder_outputs.shape[0]
        len_ = encoder_outputs.shape[1]
        return self.loss(scores.view(batch_size * len_, -1), labels_ids.view(-1)) + self.intent_loss(
            intent_score, cls_ids)

    @classmethod
    def create(cls, label_size, intent_size,
               embedding_dim=64, hidden_dim=256, rnn_layers=1, dropout_p=0.1, pad_idx=0):
        return cls(label_size=label_size, intent_size=intent_size,
                   embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                   rnn_layers=rnn_layers, dropout_p=dropout_p, pad_idx=pad_idx)


class NCRFDecoder(nn.Module):

    def __init__(self,
                 crf, label_size, input_dim, input_dropout=0.5, nbest=8):
        super(NCRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(input_dropout)
        self.linear = Linears(in_features=input_dim,
                              out_features=label_size,
                              hiddens=[input_dim // 2])
        self.nbest = nbest
        self.crf = crf
        self.label_size = label_size

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        inputs = self.dropout(inputs)

        output = inputs.contiguous().view(-1, self.input_dim)
        # Fully-connected layer
        output = self.linear.forward(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, labels_mask):
        logits = self.forward_model(inputs)
        _, preds = self.crf._viterbi_decode_nbest(logits, labels_mask, self.nbest)
        preds = preds[:, :, 0]
        return preds

    def score(self, inputs, labels_mask, labels):
        logits = self.forward_model(inputs)
        crf_score = self.crf.neg_log_likelihood_loss(logits, labels_mask, labels) / logits.size(0)
        return crf_score

    @classmethod
    def from_config(cls, config):
        return cls.create(**config)

    @classmethod
    def create(cls, label_size, input_dim, input_dropout=0.5, nbest=8, device="cuda"):
        return cls(NCRF(label_size, device), label_size + 2, input_dim, input_dropout, nbest)


class ClassDecoder(nn.Module):

    def __init__(self, intent_size, input_dim, input_dropout=0.3):
        super(ClassDecoder, self).__init__()
        self.intent_loss = nn.CrossEntropyLoss()
        self.intent_size = intent_size
        self.input_dropout = input_dropout
        self.input_dim = input_dim
        self.intent_out = PoolingLinearClassifier(input_dim, intent_size, input_dropout)

    def forward(self, inputs):
        return self.intent_out(inputs).argmax(-1)

    def score(self, inputs, cls_ids):
        return self.intent_loss(self.intent_out(inputs), cls_ids)
