from torch.nn import functional
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils import rnn as rnn_utils
import math


class BiLSTM(nn.Module):

    def __init__(self, embedding_size=768, hidden_dim=512, rnn_layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_dim // 2,
            rnn_layers, batch_first=True, bidirectional=True)

    def forward(self, input_, input_mask):
        length = input_mask.sum(-1)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_ = input_[sorted_idx]
        packed_input = rnn_utils.pack_padded_sequence(input_, sorted_lengths.data.tolist(), batch_first=True)
        output, (hidden, _) = self.lstm(packed_input)
        padded_outputs = rnn_utils.pad_packed_sequence(output, batch_first=True)[0]
        _, reversed_idx = torch.sort(sorted_idx)
        return padded_outputs[reversed_idx], hidden[:, reversed_idx]

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class Linear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        init.orthogonal_(self.weight)


class Linears(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hiddens,
                 bias=True,
                 activation='tanh'):
        super(Linears, self).__init__()
        assert len(hiddens) > 0

        self.in_features = in_features
        self.out_features = self.output_size = out_features

        in_dims = [in_features] + hiddens[:-1]
        self.linears = nn.ModuleList([Linear(in_dim, out_dim, bias=bias)
                                      for in_dim, out_dim
                                      in zip(in_dims, hiddens)])
        self.output_linear = Linear(hiddens[-1], out_features, bias=bias)
        self.activation = getattr(functional, activation)

    def forward(self, inputs):
        linear_outputs = inputs
        for linear in self.linears:
            linear_outputs = linear.forward(linear_outputs)
            linear_outputs = self.activation(linear_outputs)
        return self.output_linear.forward(linear_outputs)


# Reused from https://github.com/JayParks/transformer/
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v] note: (len_k == len_v)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor  # attn: [b_size x len_q x len_k]
        if attn_mask is not None:
            print(attn_mask.size(), attn.size())
            assert attn_mask.size() == attn.size()
            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        outputs = torch.bmm(attn, v) # outputs: [b_size x len_q x d_v]

        return outputs, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d_hid), requires_grad=True)
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean.expand_as(z)) / (std.expand_as(z) + self.eps)
        ln_out = self.gamma.expand_as(ln_out) * ln_out + self.beta.expand_as(ln_out)

        return ln_out


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_k, dropout)

        init.xavier_normal(self.w_q)
        init.xavier_normal(self.w_k)
        init.xavier_normal(self.w_v)

    def forward(self, q, k, v, attn_mask=None):
        (d_k, d_v, d_model, n_heads) = (self.d_k, self.d_v, self.d_model, self.n_heads)
        b_size = k.size(0)

        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_q x d_model]
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_k x d_model]
        v_s = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_v x d_model]

        q_s = torch.bmm(q_s, self.w_q).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_k x d_k]
        v_s = torch.bmm(v_s, self.w_v).view(b_size * n_heads, -1, d_v)  # [b_size * n_heads x len_v x d_v]

        # perform attention, result_size = [b_size * n_heads x len_q x d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_heads, 1, 1)
        outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)

        # return a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        return torch.split(outputs, b_size, dim=0), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.attention = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.proj = Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # outputs: a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        outputs, attn = self.attention(q, k, v, attn_mask=attn_mask)
        # concatenate 'n_heads' multi-head attentions
        outputs = torch.cat(outputs, dim=-1)
        # project back to residual size, result_size = [b_size x len_q x d_model]
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(residual + outputs), attn


class _BahdanauAttention(nn.Module):
    def __init__(self, method, hidden_size):
        super(_BahdanauAttention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param mask:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        """
        max_len = encoder_outputs.size(0)
        # this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        # [B*T*H]
        encoder_outputs = encoder_outputs.transpose(0, 1)
        # compute attention score
        attn_energies = self.score(H, encoder_outputs)
        if mask is not None:
            attn_energies = attn_energies.masked_fill(mask, -1e18)
        # normalize with softmax
        return functional.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = functional.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        # [B*H*T]
        energy = energy.transpose(2, 1)
        # [B*1*H]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)
        # [B*1*T]
        energy = torch.bmm(v, energy)
        # [B*T]
        return energy.squeeze(1)


class BahdanauAttention(nn.Module):
    """Reused from https://github.com/chrisbangun/pytorch-seq2seq_with_attention/"""

    def __init__(self, hidden_dim=128, query_dim=128, memory_dim=128):
        super(BahdanauAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.sofmax = nn.Softmax()

        self.query_layer = nn.Linear(query_dim, hidden_dim, bias=False)
        self.memory_layer = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.alignment_layer = nn.Linear(hidden_dim, 1, bias=False)

    def alignment_score(self, query, keys):
        query = self.query_layer(query)
        keys = self.memory_layer(keys)

        extendded_query = query.unsqueeze(1)
        alignment = self.alignment_layer(functional.tanh(extendded_query + keys))
        return alignment.squeeze(2)

    def forward(self, query, keys):
        alignment_score = self.alignment_score(query, keys)
        weight = functional.softmax(alignment_score)
        context = weight.unsqueeze(2) * keys
        total_context = context.sum(1)
        return total_context, alignment_score
