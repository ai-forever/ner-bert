from transformers import (
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelWithLMHead
)
from torch.nn import Module


MODEL_TYPES = {
    "base": AutoModel,
    "qa": AutoModelForQuestionAnswering,
    "sc": AutoModelForSequenceClassification,
    "wlmh": AutoModelWithLMHead
}


class GeneralModel(Module):

    @classmethod
    def create(cls, model_name, model_type):
        model = MODEL_TYPES[model_type].from_pretrained(model_name)
        return cls(model)

    def __init__(self, model):
        super(GeneralModel, self).__init__()
        self.model = model

    def forward(self, samples):
        res = self.model(**samples["net_input"])
        if isinstance(res, tuple):
            res = res[0]
        return {"cls": res}
