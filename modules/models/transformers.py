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
    "cls": AutoModelForSequenceClassification,
    "wlmh": AutoModelWithLMHead
}


class GeneralModel(Module):

    @classmethod
    def create(cls, model_name, model_type, **model_args):
        model = MODEL_TYPES[model_type].from_pretrained(model_name, **model_args)
        return cls(model, model_type)

    def __init__(self, model, model_type):
        super(GeneralModel, self).__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, samples):
        res = self.model(**samples["net_input"])
        if isinstance(res, tuple):
            res = res[0]
        return {self.model_type: res}
