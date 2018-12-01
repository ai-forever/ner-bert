from modules.layers.encoders import *
from modules.layers.decoders import *
from modules.layers.embedders import *
import abc


class NerModel(nn.Module, metaclass=abc.ABCMeta):
    """Base class for all Models"""
    def __init__(self, encoder, decoder, use_cuda=True):
        super(NerModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if use_cuda:
            self.cuda()

    @abc.abstractmethod
    def forward(self, *batch):
        # return self.decoder(self.encoder(batch))
        raise NotImplementedError("abstract method forward must be implemented")

    @abc.abstractmethod
    def score(self, *batch):
        # return self.decoder.score(self.encoder(batch))
        raise NotImplementedError("abstract method score must be implemented")

    @abc.abstractmethod
    def create(self, *args):
        raise NotImplementedError("abstract method create must be implemented")


class BertBiLSTMCRF(NerModel):

    def forward(self, batch):
        output, _ = self.encoder(batch)
        return self.decoder(output, batch[-2])

    def score(self, batch):
        output, _ = self.encoder(batch)
        return self.decoder.score(output, batch[-2], batch[-1])

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               bert_config_file, init_checkpoint_pt, embedding_dim=768, bert_mode="weighted",
               freeze=True,
               # BiLSTMEncoder params
               enc_hidden_dim=128, rnn_layers=1,
               # CRFDecoder params
               input_dropout=0.5,
               # Global params
               use_cuda=True):
        embedder = BertEmbedder.create(
            bert_config_file, init_checkpoint_pt, embedding_dim, use_cuda, bert_mode, freeze)
        encoder = BiLSTMEncoder.create(embedder, enc_hidden_dim, rnn_layers, use_cuda)
        decoder = CRFDecoder.create(label_size, encoder.output_dim, input_dropout)
        return cls(encoder, decoder, use_cuda)
