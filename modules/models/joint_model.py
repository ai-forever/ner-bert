from modules.layers.crf import CRF
from torch import nn
import torch
from modules.layers import MultiHeadAttention, modeling
from .model import NerModel


# TODO: split model to Encoder and NER and CLS decoders
class JointModel(NerModel):
    def __init__(self,
                 tagset_size, encoder, cls_size, embedding_dim=768, dropout_ratio=0.1,
                 # For rnn
                 use_rnn=True, hidden_dim=128, rnn_layers=1,
                 use_cuda=True,
                 # For attention
                 num_heads=3, dropout_attn=0.1, key_dim=64, val_dim=64, use_attn=False,
                 bert_mode="last"):
        super(JointModel, self).__init__(
            tagset_size, encoder, embedding_dim, dropout_ratio,
            use_rnn, hidden_dim, rnn_layers, use_cuda,
            num_heads, dropout_attn, key_dim, val_dim, use_attn, bert_mode)

        self.cls_loss = nn.NLLLoss()

        self.cls_size = cls_size
        self.cls_hidden_layer = nn.Linear(self.hidden_dim, self.cls_size)

    def _forward_cls(self, output, input_mask, from_logits=True):
        if not from_logits:
            output, input_type_ids = output
            output = self.get_logits(output, input_mask, input_type_ids)
        output = torch.mean(output * input_mask, 0)
        output = self.cls_hidden_layer(output)
        return nn.functional.log_softmax(output)

    def _forward_tag(self, output, input_mask, from_logits=True):
        if not from_logits:
            output, input_type_ids = output
            output = self.get_logits(output, input_mask.transpose(0, 1), input_type_ids)
        output = self.hidden_layer(output)
        output = self.activation(output)
        return self.crf.decode(output, mask=input_mask)

    def forward(self, batch):
        output, input_mask, input_type_ids = batch
        output = self.get_logits(output, input_mask, input_type_ids)
        input_mask = input_mask.transpose(0, 1)
        return self._forward_cls(output, input_mask), self._forward_tag(output, input_mask)

    def _score_tag(self, output, input_mask, labels_ids, from_logits=True):
        if not from_logits:
            output, input_type_ids = output
            output = self.get_logits(output, input_mask.transpose(0, 1), input_type_ids)
        output = self.hidden_layer(output)
        output = self.activation(output)
        labels_ids = labels_ids.transpose(0, 1)
        return -self.crf(output, labels_ids, mask=input_mask)

    def _score_cls(self, output, input_mask, cls_ids, from_logits=True):
        cls_preds = self._forward_cls(output, input_mask, from_logits)
        return self.cls_loss(cls_preds, cls_ids)

    def score(self, batch):
        output, input_mask, input_type_ids, labels_ids, cls_ids = batch
        output = self.get_logits(output, input_mask, input_type_ids)
        input_mask = input_mask.transpose(0, 1)
        return self._score_cls(output, input_mask, cls_ids) +\
               self._score_tag(output, input_mask, labels_ids)

    @staticmethod
    def create(
            bert_config_file, init_checkpoint_pt,
            tagset_size, cls_size, embedding_dim=768, dropout_ratio=0.4,
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
            bert_mode=bert_mode, cls_size=cls_size
        )
        if freeze_enc:
            model.freeze_encoder()
            print("freeze_encoder")
        return model
