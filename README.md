## 0. Papers
There are two solutions based on this architecture.
1. [BSNLP 2019 ACL workshop](http://bsnlp.cs.helsinki.fi/shared_task.html): [solution](https://github.com/king-menin/slavic-ner) and [paper](https://arxiv.org/abs/1906.09978) on multilingual shared task.
2. The second place [solution](https://github.com/king-menin/AGRR-2019) of [Dialogue AGRR-2019](https://github.com/dialogue-evaluation/AGRR-2019) task.

## Description
This repository contains solution of NER task based on PyTorch [reimplementation](https://github.com/huggingface/pytorch-pretrained-BERT) of [Google's TensorFlow repository for the BERT model](https://github.com/google-research/bert) that was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

This implementation can load any pre-trained TensorFlow checkpoint for BERT (in particular [Google's pre-trained models](https://github.com/google-research/bert)) and a conversion script is provided (see below).

## 2. Results
We didn't search best parametres and obtained the following results.

| Model | Data set | Dev F1 tok | Dev F1 span | Test F1 tok | Test F1 span
|-|-|-|-|-|-|
|**OURS**||||||
| M-BERTCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8598 | 0.7676
| M-BERTNCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8603 | 0.7783
| M-BERTBiLSTMCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8780 | 0.8108
| M-BERTBiLSTMCRF-BIO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8263 | 0.8051
| M-BERTBiLSTMNCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8594 | 0.7842
| M-BERTAttnCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8630 | 0.7879
| M-BERTBiLSTMAttnCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8851 | 0.8244
| M-BERTBiLSTMAttnNCRF-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | 0.8609 | 0.7869
| M-BERTBiLSTMAttnNCRF-fit_BERT-IO | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - |  0.8739 | 0.8201
|-|-|-|-|-|-|
| BERTBiLSTMCRF-IO | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.9624 | 0.9273 | - | -
| BERTBiLSTMCRF-BIO | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.9530 | 0.9236 | - | -
| B-BERTBiLSTMCRF-IO | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.9635 | 0.9277 | - | -
| B-BERTBiLSTMCRF-BIO | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.9536 | 0.9156 | - | -
| B-BERTBiLSTMAttnCRF-IO | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.9571 | 0.9114 | - | -
| B-BERTBiLSTMAttnNCRF-IO | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.9631 | 0.9197 | - | -
|**Current SOTA**||||||
| DeepPavlov-RuBERT-NER | [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | - | - | - | **0.8266**
| CSE | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | - | - | **0.931** | -
| BERT-LARGE | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.966 | - | 0.928 | -
| BERT-BASE | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) | 0.964 | - | 0.924 | -
