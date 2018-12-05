# PyTorch solution of NER task with Google AI's BERT model
## 0. Introduction

This repository contains solution of NER task based on PyTorch [reimplementation](https://github.com/huggingface/pytorch-pretrained-BERT) of [Google's TensorFlow repository for the BERT model](https://github.com/google-research/bert) that was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

This implementation can load any pre-trained TensorFlow checkpoint for BERT (in particular [Google's pre-trained models](https://github.com/google-research/bert)) and a conversion script is provided (see below).

## 1. Loading a TensorFlow checkpoint (e.g. [Google's pre-trained models](https://github.com/google-research/bert#pre-trained-models))

You can convert any TensorFlow checkpoint for BERT (in particular [the pre-trained models released by Google](https://github.com/google-research/bert#pre-trained-models)) in a PyTorch save file by using the [`convert_tf_checkpoint_to_pytorch.py`](convert_tf_checkpoint_to_pytorch.py) script.

This script takes as input a TensorFlow checkpoint (three files starting with `bert_model.ckpt`) and the associated configuration file (`bert_config.json`), and creates a PyTorch model for this configuration, loads the weights from the TensorFlow checkpoint in the PyTorch model and saves the resulting model in a standard PyTorch save file that can be imported using `torch.load()`.

You only need to run this conversion script **once** to get a PyTorch model. You can then disregard the TensorFlow checkpoint (the three files starting with `bert_model.ckpt`) but be sure to keep the configuration file (`bert_config.json`) and the vocabulary file (`vocab.txt`) as these are needed for the PyTorch model too.

To run this specific conversion script you will need to have TensorFlow and PyTorch installed (`pip install tensorflow`). The rest of the repository only requires PyTorch.

Here is an example of the conversion process for a pre-trained `BERT-Base Uncased` model:

```shell
export BERT_BASE_DIR=/path/to/bert/multilingual_L-12_H-768_A-12

python3 convert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
    --bert_config_file $BERT_BASE_DIR/bert_config.json \
    --pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin
```

You can download Google's pre-trained models for the conversion [here](https://github.com/google-research/bert#pre-trained-models).

There is used the [BERT-Base, Multilingual](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip) and [BERT-Cased, Multilingual](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) (recommended) in this solution.

## 2. Results
We didn't search best parametres and obtained the following results for no more than <b>10 epochs</b>.

### Only NER models
#### Model: `BertBiLSTMAttnCRF`.

| Dataset | Lang | IOB precision | Span precision | Total spans in test set | Notebook
|-|-|-|-|-|-|
| [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | ru | <b>0.937</b> | <b>0.883</b> | 4 | [factrueval.ipynb](factrueval.ipynb)
| [Atis](https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data) | en | 0.852 | 0.787 | 65 | [conll-2003.ipynb](conll-2003.ipynb)
| [Conll-2003](https://github.com/kyzhouhzau/BERT-NER/tree/master/NERdata) | en | <b>0.945</b> | 0.858 | 5 | [atis.ipynb](atis.ipynb)

* Factrueval (f1): 0.9163±0.006, best **0.926**.
* Atis (f1): 0.882±0.02, best **0.896**
* Conll-2003 (f1, dev): 0.949±0.002, best **0.951**; 0.892 (f1, test).

#### Model: `BertBiLSTMAttnNMT`.

| Dataset | Lang | IOB precision | Span precision | Total spans in test set | Notebook
|-|-|-|-|-|-|
| [FactRuEval](https://github.com/dialogue-evaluation/factRuEval-2016) | ru | 0.925 | 0.827 | 4 | [factrueval-nmt.ipynb](factrueval-nmt.ipynb)
| [Atis](https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data) | en | <b>0.919</b> | <b>0.829</b> | 65 | [atis-nmt.ipynb](atis-nmt.ipynb)
| [Conll-2003](https://github.com/kyzhouhzau/BERT-NER/tree/master/NERdata) | en | 0.936 | <b>0.900</b> | 5 | [conll-2003-nmt.ipynb](conll-2003-nmt.ipynb)

### Joint Models
#### Model: `BertBiLSTMAttnCRFJoint`

| Dataset | Lang | IOB precision | Span precision | Clf precision | Total spans in test set | Total classes | Notebook
|-|-|-|-|-|-|-|-|
| [Atis](https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data) | en | 0.877 | 0.824 | 0.894 | 65 | 17 | [atis-joint.ipynb](atis-joint.ipynb)

#### Model: `BertBiLSTMAttnNMTJoint`

| Dataset | Lang | IOB precision | Span precision | Clf precision | Total spans in test set | Total classes | Notebook
|-|-|-|-|-|-|-|-|
| [Atis](https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data) | en | 0.913 | 0.820 | 0.888 | 65 | 17 | [atis-joint-nmt.ipynb](atis-joint-nmt.ipynb)

## 3. Installation, requirements, test

This code was tested on Python 3.5+. The requirements are:

- PyTorch (>= 0.4.1)
- tqdm
- tensorflow (for convertion)

To install the dependencies:

````bash
pip install -r ./requirements.txt
````

## PyTorch neural network models

All models are organized as `Encoder`-`Decoder`. `Encoder` is a freezed and <i>weighted</i> (as proposed in [elmo](https://allennlp.org/elmo)) bert output from 12 layers. There are three models that is obtained by using different `Decoder`.

`Encoder`: BertBiLSTM

1. `BertBiLSTMCRF`: `Encoder` + `Decoder` (BiLSTM + CRF)
2. `BertBiLSTMAttnCRF`: `Encoder` + `Decoder` (BiLSTM + MultiHead Attention + CRF)
3. `BertBiLSTMAttnNMT`: `Encoder` + `Decoder` (LSTM + Bahdanau Attention - NMT Decode)
4. `BertBiLSTMAttnCRFJoint`: `Encoder` + `Decoder` (BiLSTM + MultiHead Attention + CRF) + (PoolingLinearClassifier - for classification) - joint model with classification.
5. `BertBiLSTMAttnNMTJoint`: `Encoder` + `Decoder` (LSTM + Bahdanau Attention - NMT Decode) + (LinearClassifier - for classification) - joint model with classification.


## Usage

### 1. Loading data:

```from modules.bert_data import BertNerData as NerData```

```data = NerData.create(train_path, valid_path, vocab_file)```

### 2. Create model:

```from modules.bert_models import BertBiLSTMCRF```

```model = BertBiLSTMCRF.create(len(data.label2idx), bert_config_file, init_checkpoint_pt, enc_hidden_dim=256)```

### 3. Create learner:

```from modules.train import NerLearner```

```learner = NerLearner(model, data, best_model_path="/datadrive/models/factrueval/exp_final.cpt", base_lr=0.0001, lr_max=0.005, clip=5.0, use_lr_scheduler=True, sup_labels=data.id2label[5:])```

### 4. Learn your NER model:

```learner.fit(2, target_metric='prec')```

### 5. Predict on new data:

```from modules.data.bert_data import get_bert_data_loader_for_predict```

```dl = get_bert_data_loader_for_predict(data_path + "valid.csv", learner)```

```learner.load_model(best_model_path)```

```preds = learner.predict(dl)```

For more detailed instructions see [samples.ipynb](samples.ipynb)
