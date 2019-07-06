from pytorch_pretrained_bert import BertModel
import torch


class BERTEmbedder(torch.nn.Module):
    def __init__(self, model, config):
        super(BERTEmbedder, self).__init__()
        self.config = config
        self.model = model
        if self.config["mode"] == "weighted":
            self.bert_weights = torch.nn.Parameter(torch.FloatTensor(12, 1))
            self.bert_gamma = torch.nn.Parameter(torch.FloatTensor(1, 1))
        self.init_weights()

    def init_weights(self):
        if self.config["mode"] == "weighted":
            torch.nn.init.xavier_normal(self.bert_gamma)
            torch.nn.init.xavier_normal(self.bert_weights)

    @classmethod
    def create(
            cls, model_name='bert-base-multilingual-cased',
            device="cuda", mode="weighted",
            is_freeze=True):
        config = {
            "model_name": model_name,
            "device": device,
            "mode": mode,
            "is_freeze": is_freeze
        }
        model = BertModel.from_pretrained(model_name)
        model.to(device)
        model.train()
        self = cls(model, config)
        if is_freeze:
            self.freeze()
        return self

    @classmethod
    def from_config(cls, config):
        return cls.create(**config)

    def forward(self, batch):
        """
        batch has the following structure:
            data[0]: list, tokens ids
            data[1]: list, tokens mask
            data[2]: list, tokens type ids (for bert)
            data[3]: list, bert labels ids
        """
        encoded_layers, _ = self.model(
            input_ids=batch[0],
            token_type_ids=batch[2],
            attention_mask=batch[1],
            output_all_encoded_layers=self.config["mode"] == "weighted")
        if self.config["mode"] == "weighted":
            encoded_layers = torch.stack([a * b for a, b in zip(encoded_layers, self.bert_weights)])
            return self.bert_gamma * torch.sum(encoded_layers, dim=0)
        return encoded_layers

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
