from .bert_models import BERTNerModel
from modules.layers.decoders import *
from modules.layers.embedders import *
from modules.layers.layers import *


class BERTLinearsClassifier(BERTNerModel):

    def __init__(self, embeddings, linear, dropout, activation, device="cuda"):
        super(BERTLinearsClassifier, self).__init__()
        self.embeddings = embeddings
        self.linear = linear
        self.dropout = dropout
        self.activation = activation
        self.intent_loss = nn.CrossEntropyLoss()
        self.to(device)

    @staticmethod
    def pool(x, bs, is_max):
        """Pool the tensor along the seq_len dimension."""
        f = functional.adaptive_max_pool1d if is_max else functional.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(bs, -1)

    def forward(self, batch):
        input_embeddings = self.embeddings(batch)
        output = self.dropout(input_embeddings).transpose(0, 1)
        sl, bs, _ = output.size()
        output = self.pool(output, bs, True)
        output = self.linear(output)
        return self.activation(output).argmax(-1)

    def score(self, batch):
        input_embeddings = self.embeddings(batch)
        output = self.dropout(input_embeddings).transpose(0, 1)
        sl, bs, _ = output.size()
        output = self.pool(output, bs, True)
        output = self.linear(output)
        return self.intent_loss(self.activation(output), batch[-1])

    @classmethod
    def create(cls,
               intent_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # Decoder params
               embedding_size=768, clf_dropout=0.3, num_hiddens=2,
               activation="tanh",
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        linear = Linears(embedding_size, intent_size, [embedding_size // 2**idx for idx in range(num_hiddens)])
        dropout = nn.Dropout(clf_dropout)
        activation = getattr(functional, activation)
        return cls(embeddings, linear, dropout, activation, device)


class BERTLinearClassifier(BERTNerModel):

    def __init__(self, embeddings, linear, dropout, activation, device="cuda"):
        super(BERTLinearClassifier, self).__init__()
        self.embeddings = embeddings
        self.linear = linear
        self.dropout = dropout
        self.activation = activation
        self.intent_loss = nn.CrossEntropyLoss()
        self.to(device)

    @staticmethod
    def pool(x, bs, is_max):
        """Pool the tensor along the seq_len dimension."""
        f = functional.adaptive_max_pool1d if is_max else functional.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(bs, -1)

    def forward(self, batch):
        input_embeddings = self.embeddings(batch)
        output = self.dropout(input_embeddings).transpose(0, 1)
        sl, bs, _ = output.size()
        output = self.pool(output, bs, True)
        output = self.linear(output)
        return self.activation(output).argmax(-1)

    def score(self, batch):
        input_embeddings = self.embeddings(batch)
        output = self.dropout(input_embeddings).transpose(0, 1)
        sl, bs, _ = output.size()
        output = self.pool(output, bs, True)
        output = self.linear(output)
        return self.intent_loss(self.activation(output), batch[-1])

    @classmethod
    def create(cls,
               intent_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # Decoder params
               embedding_size=768, clf_dropout=0.3,
               activation="sigmoid",
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        linear = Linear(embedding_size, intent_size)
        dropout = nn.Dropout(clf_dropout)
        activation = getattr(functional, activation)
        return cls(embeddings, linear, dropout, activation, device)


class BERTBaseClassifier(BERTNerModel):

    def __init__(self, embeddings, clf, device="cuda"):
        super(BERTBaseClassifier, self).__init__()
        self.embeddings = embeddings
        self.clf = clf
        self.to(device)

    def forward(self, batch):
        input_embeddings = self.embeddings(batch)
        return self.clf(input_embeddings)

    def score(self, batch):
        input_, labels_mask, input_type_ids, cls_ids = batch
        input_embeddings = self.embeddings(batch)
        return self.clf.score(input_embeddings, cls_ids)

    @classmethod
    def create(cls,
               intent_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # Decoder params
               embedding_size=768, clf_dropout=0.3,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        clf = ClassDecoder(intent_size, embedding_size, clf_dropout)
        return cls(embeddings, clf, device)


class BERTBiLSTMAttnClassifier(BERTNerModel):

    def __init__(self, embeddings, lstm, attn, clf, device="cuda"):
        super(BERTBiLSTMAttnClassifier, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.attn = attn
        self.clf = clf
        self.to(device)

    def forward(self, batch):
        input_, labels_mask, input_type_ids = batch[:3]
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.clf(output)

    def score(self, batch):
        input_, labels_mask, input_type_ids = batch[:3]
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.clf.score(output, batch[-1])

    @classmethod
    def create(cls,
               intent_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # Decoder params
               clf_dropout=0.3,
               # BiLSTM
               hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # Attn params
               embedding_size=768, key_dim=64, val_dim=64, num_heads=3, attn_dropout=0.3,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
            embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, attn_dropout)
        clf = ClassDecoder(intent_size, embedding_size, clf_dropout)
        return cls(embeddings, lstm, attn, clf, device)
