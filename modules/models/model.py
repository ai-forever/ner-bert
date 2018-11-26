from modules.layers.crf import CRF
from torch import nn
import torch
from modules.layers.layers import MultiHeadAttention
from modules.layers import modeling


class NerModel(nn.Module):
    def __init__(self,
                 tagset_size, encoder, embedding_dim=768, dropout_ratio=0.1,
                 # For rnn
                 use_rnn=True, hidden_dim=128, rnn_layers=1,
                 use_cuda=True,
                 # For attention
                 num_heads=3, dropout_attn=0.1, key_dim=64, val_dim=64, use_attn=False,
                 bert_mode="last"):
        super(NerModel, self).__init__()
        self.encoder = encoder
        self.bert_mode = bert_mode
        if self.bert_mode == "weighted":
            self.bert_weights = nn.Parameter(torch.FloatTensor(12, 1))
            self.bert_gamma = nn.Parameter(torch.FloatTensor(1, 1))
            nn.init.xavier_normal(self.bert_gamma)
            nn.init.xavier_normal(self.bert_weights)
        self.tagset_size = tagset_size
        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.use_rnn = use_rnn
        self.use_attn = use_attn
        rnn_dim = embedding_dim
        if self.use_rnn:
            print("Use rnn of type: {} with n_layers: {}".format("lstm", rnn_layers))
            self.rnn = nn.LSTM(rnn_dim, hidden_dim // 2, rnn_layers, bidirectional=True)
        else:
            hidden_dim = embedding_dim
        if self.use_attn:
            self.attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, dropout_attn)
            print("Use MultiHeadAttention layer with num_heads {}".format(num_heads))
        self.use_hidden = True

        print("Use fully-connected layer before crf.")
        self.hidden_layer = nn.Linear(hidden_dim, self.tagset_size)

        self.activation = nn.Tanh()
        self.dropout2 = nn.Dropout(p=dropout_ratio)
        self.crf = CRF(tagset_size)
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def get_logits(self, output, input_mask, input_type_ids):
        output = self.get_bert_output(output, input_mask, input_type_ids)
        output = self.dropout1(output)
        if self.use_rnn:
            output = output.transpose(0, 1)
            output, _ = self.rnn(output)
            output = output.transpose(0, 1)
        if self.use_attn:
            output, _ = self.attn(output, output, output, None)
            # output = torch.cat([output, attn_output], -1)
        output = output.transpose(0, 1)
        output = self.dropout2(output)
        return output

    def forward(self, batch):
        output, input_mask, input_type_ids = batch
        output = self.get_logits(output, input_mask, input_type_ids)
        output = self.hidden_layer(output)
        output = self.activation(output)
        input_mask = input_mask.transpose(0, 1)
        return self.crf.decode(output, mask=input_mask)

    def score(self, batch):
        output, input_mask, input_type_ids, labels_ids = batch
        output = self.get_logits(output, input_mask, input_type_ids)
        output = self.hidden_layer(output)
        output = self.activation(output)
        input_mask = input_mask.transpose(0, 1)
        labels_ids = labels_ids.transpose(0, 1)
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

    def freeze_encoder_to(self, to=-1):
        if to < 0:
            to = len(self.encoder.encoder.layer) + to + 1
        for idx in range(to):
            for param in self.encoder.encoder.layer[idx].parameters():
                    param.requires_grad = False
        print("Encoder freezed to {}".format(to))
        to = len(self.encoder.encoder.layer)
        for idx in range(idx, to):
            for param in self.encoder.encoder.layer[idx].parameters():
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

    def get_bert_output(self, input_ids, input_mask, input_type_ids):
        all_encoder_layers, _ = self.encoder(input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        if self.bert_mode == "last":
            return all_encoder_layers[-1]
        elif self.bert_mode == "weighted":
            all_encoder_layers = torch.stack([a * b for a, b in zip(all_encoder_layers, self.bert_weights)])
            return self.bert_gamma * torch.sum(all_encoder_layers, dim=0)

    @staticmethod
    def create(
        bert_config_file, init_checkpoint_pt,
        tagset_size, embedding_dim=768, dropout_ratio=0.4,
        # For rnn
        use_rnn=True, hidden_dim=128, rnn_layers=1,
        use_cuda=True,
        # For attention
        num_heads=3, dropout_attn=0.1, key_dim=64, val_dim=64, use_attn=False,
        freeze_enc=True, bert_mode="last"
    ):
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        bert_model = modeling.BertModel(bert_config)
        if use_cuda:
            device = torch.device("cuda")
            map_location = "cuda"
        else:
            map_location = "cpu"
            device = torch.device("cpu")
        bert_model.load_state_dict(torch.load(init_checkpoint_pt, map_location=map_location))
        bert_model = bert_model.to(device)
        model = NerModel(
            tagset_size=tagset_size, encoder=bert_model, embedding_dim=embedding_dim, dropout_ratio=dropout_ratio,
            use_rnn=use_rnn, hidden_dim=hidden_dim, rnn_layers=rnn_layers,
            use_cuda=use_cuda,
            num_heads=num_heads, dropout_attn=dropout_attn, key_dim=key_dim, val_dim=val_dim, use_attn=use_attn,
            bert_mode=bert_mode
        )
        if freeze_enc:
            model.freeze_encoder()
            print("freeze_encoder")
        return model
