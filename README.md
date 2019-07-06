## 0. Papers
There are two solutions based on this architecture.
1. [BSNLP 2019 ACL workshop](http://bsnlp.cs.helsinki.fi/shared_task.html): [solution](https://github.com/king-menin/slavic-ner) and [paper](https://arxiv.org/abs/1906.09978) on multilingual shared task.
2. The second place [solution](https://github.com/king-menin/AGRR-2019) of [Dialogue AGRR-2019](https://github.com/dialogue-evaluation/AGRR-2019) task and [paper](http://www.dialog-21.ru/media/4679/emelyanov-artemova-gapping_parsing_using_pretrained_embeddings__attention_mechanisn_and_ncrf.pdf).

## Description
This repository contains solution of NER task based on PyTorch [reimplementation](https://github.com/huggingface/pytorch-pretrained-BERT) of [Google's TensorFlow repository for the BERT model](https://github.com/google-research/bert) that was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

This implementation can load any pre-trained TensorFlow checkpoint for BERT (in particular [Google's pre-trained models](https://github.com/google-research/bert)).

Old version is in "old" branch.

## 2. Usage
### 2.1 Create data
```
from modules.data import bert_data
data = bert_data.LearnData.create(
    train_df_path=train_df_path,
    valid_df_path=valid_df_path,
    idx2labels_path="/path/to/vocab",
    clear_cache=True
)
```

### 2.2 Create model
```
from modules.models.bert_models import BERTBiLSTMAttnCRF
model = BERTBiLSTMAttnCRF.create(len(data.train_ds.idx2label))
```

### 2.3 Create Learner
```
from modules.train.train import NerLearner
num_epochs = 100
learner = NerLearner(
    model, data, "/path/for/save/best/model", t_total=num_epochs * len(data.train_dl))
```

### 2.4 Predict
```
from modules.data.bert_data import get_data_loader_for_predict
learner.load_model()
dl = get_data_loader_for_predict(data, df_path="/path/to/df/for/predict")
preds = learner.predict(dl)
```

### 2.5 Evaluate
```
from sklearn_crfsuite.metrics import flat_classification_report
from modules.analyze_utils.utils import bert_labels2tokens, voting_choicer
from modules.analyze_utils.plot_metrics import get_bert_span_report
from modules.analyze_utils.main_metrics import precision_recall_f1


pred_tokens, pred_labels = bert_labels2tokens(dl, preds)
true_tokens, true_labels = bert_labels2tokens(dl, [x.bert_labels for x in dl.dataset])
tokens_report = flat_classification_report(true_labels, pred_labels, digits=4)
print(tokens_report)

results = precision_recall_f1(true_labels, pred_labels)
```

## 3. Results
We didn't search best parametres and obtained the following results.

| Model | Data set | Dev F1 tok | Dev F1 span | Test F1 tok | Test F1 span
|-|-|-|-|-|-|
|**OURS**||||||
| M-BERTCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8543 | 0.8409
| M-BERTNCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8637 | 0.8516
| M-BERTBiLSTMCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8835 | **0.8718**
| M-BERTBiLSTMNCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8632 | 0.8510
| M-BERTAttnCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8503 | 0.8346
| M-BERTBiLSTMAttnCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | **0.8839** | 0.8716
| M-BERTBiLSTMAttnNCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8807 | 0.8680
| M-BERTBiLSTMAttnCRF-fit_BERT-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - |  0.8823 | 0.8709
| M-BERTBiLSTMAttnNCRF-fit_BERT-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - |  0.8583 | 0.8456
|-|-|-|-|-|-|
| BERTBiLSTMCRF-IO | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.9629 | - | 0.9221 | -
| B-BERTBiLSTMCRF-IO | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.9635 | - | 0.9229 | -
| B-BERTBiLSTMAttnCRF-IO | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.9614 | - | 0.9237 | -
| B-BERTBiLSTMAttnNCRF-IO | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.9631 | - | **0.9249** | -
|**Current SOTA**||||||
| DeepPavlov-RuBERT-NER | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | - | **0.8266**
| CSE | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | - | - | **0.931** | -
| BERT-LARGE | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.966 | - | 0.928 | -
| BERT-BASE | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.964 | - | 0.924 | -
