from .bert_models import BERTNerModel
from modules.layers.decoders import *
from modules.layers.embedders import *


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
