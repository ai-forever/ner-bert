# PyTorch solution of NER task with Google AI's BERT model
## Introduction

This repository contains solution of NER task based on PyTorch [reimplementation](https://github.com/huggingface/pytorch-pretrained-BERT) of [Google's TensorFlow repository for the BERT model](https://github.com/google-research/bert) that was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

This implementation can load any pre-trained TensorFlow checkpoint for BERT (in particular [Google's pre-trained models](https://github.com/google-research/bert)) and a conversion script is provided (see below).

## Loading a TensorFlow checkpoint (e.g. [Google's pre-trained models](https://github.com/google-research/bert#pre-trained-models))

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

There is used the [BERT-Base, Multilingual](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip) in this solution.

## Installation, requirements, test

This code was tested on Python 3.5+. The requirements are:

- PyTorch (>= 0.4.1)
- tqdm
- tensorflow (for convertion)

To install the dependencies:

````bash
pip install -r ./requirements.txt
````

## Usage

### 1. Loading data:

```from modules.data import NerData```

```data = NerData.create(train_path, valid_path, vocab_file)```

### 2. Create model:

```from modules.models import BertBiLSTMCRF```

```model = BertBiLSTMCRF.create(len(data.label2idx), bert_config_file, init_checkpoint_pt, enc_hidden_dim=256)```

### 3. Create learner:

```from modules.train import NerLearner```

```learner = NerLearner(model, data, best_model_path="/datadrive/models/factrueval/exp_final.cpt", base_lr=0.0001, lr_max=0.005, clip=5.0, use_lr_scheduler=True, sup_labels=data.id2label[5:])```

### 4. Learn your NER model:

```learner.fit(2, target_metric='prec')```

### 5. Predict on new data:

```from modules.data.data import get_bert_data_loader_for_predict```

```dl = get_bert_data_loader_for_predict(data_path + "valid.csv", learner)```

```learner.load_model(best_model_path)```

```preds = learner.predict(dl)```

For more detailed instructions see [samples.ipynb](https://github.com/king-menin/ner-bert/blob/master/samples.ipynb)

### TODO:
1. Add tests
2. Add searcher of best params
3. Improve model:
    a. Add pos tags;
    b. Refactor attention model