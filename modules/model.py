from modules.crf import CRF
from torch import nn
import torch
from modules import modeling
from modules import extract_features


class NerModel(nn.Module):
    def __init__(self, tagset_size, encoder, embedding_dim=768, use_lstm=True, use_hidden=True,
                 hidden_dim=128, rnn_layers=1, dropout_ratio=0.4, use_cuda=True):
        super(NerModel, self).__init__()
        self.encoder = encoder
        self.tagset_size = tagset_size
        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                                num_layers=rnn_layers, bidirectional=True, dropout=dropout_ratio)
        else:
            hidden_dim = embedding_dim
        self.use_hidden = use_hidden
        if self.use_hidden:
            self.hidden_layer = nn.Linear(hidden_dim, self.tagset_size)
        self.activation = nn.Tanh()
        self.dropout2 = nn.Dropout(p=dropout_ratio)
        self.crf = CRF(tagset_size)
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def forward(self, batch):
        """
        args:
            sentence (batch_size, word_seq_len) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size), hidden
        """
        output, input_mask, input_type_ids = batch
        output = self.get_bert_output(output, input_mask, input_type_ids, self.encoder)
        output = output.transpose(0, 1)
        input_mask = input_mask.transpose(0, 1)
        output = self.dropout1(output)
        if self.use_lstm:
            output, hidden = self.lstm(output)
        if self.use_hidden:
            output = self.hidden_layer(output)
        output = self.activation(output)
        return self.crf.decode(self.dropout2(output), mask=input_mask)
    
    def score(self, batch):
        output, input_mask, input_type_ids, labels_ids = batch
        output = self.get_bert_output(output, input_mask, input_type_ids, self.encoder)
        output = output.transpose(0, 1)
        input_mask = input_mask.transpose(0, 1)
        labels_ids = labels_ids.transpose(0, 1)
        output = self.dropout1(output)
        if self.use_lstm:
            output, hidden = self.lstm.forward(output)
        if self.use_hidden:
            output = self.hidden_layer(output)
        output = self.activation(output)
        output = self.dropout2(output)
        # - masked_cross_entropy(output, tags, mask)
        return -self.crf(output, labels_ids, mask=input_mask)
    
    def freeze_encoder(self):
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def unfreeze_encoder(self):
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = True

    def get_n_trainable_params(self):
        pp=0
        for p in list(self.parameters()):
            if p.requires_grad == True:
                nn=1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
        return pp

    @staticmethod
    def get_bert_output(input_ids, input_mask, input_type_ids, bert_model, mode="last"):
        all_encoder_layers, _ = bert_model(input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        if mode == "last":
            return all_encoder_layers[-1]

    @staticmethod
    def create(bert_config_file, init_checkpoint_pt, tag_size, use_hidden=True, use_lstm=True, use_cuda=True, freeze_enc=True, hidden_dim=128):
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        bert_model = extract_features.BertModel(bert_config)
        if use_cuda:
            map_location = "cuda"
        else:
            map_location = "cpu"
        bert_model.load_state_dict(torch.load(init_checkpoint_pt, map_location=map_location))
        if use_cuda:
            device = torch.device("cuda")
        bert_model = bert_model.to(device)
        model = NerModel(tag_size, encoder=bert_model, use_lstm=use_lstm, use_hidden=use_hidden, use_cuda=use_cuda, hidden_dim=hidden_dim)
        if freeze_enc:
            model.freeze_encoder()
            print("freeze_encoder")
        return model
