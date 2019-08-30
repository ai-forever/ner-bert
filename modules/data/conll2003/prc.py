import pandas as pd
from modules import tqdm
import argparse
import codecs
import os


def conll2003_preprocess(
        data_dir, train_name="eng.train", dev_name="eng.testa", test_name="eng.testb"):
    train_f = read_data(os.path.join(data_dir, train_name))
    dev_f = read_data(os.path.join(data_dir, dev_name))
    test_f = read_data(os.path.join(data_dir, test_name))

    train = pd.DataFrame({"labels": [x[0] for x in train_f], "text": [x[1] for x in train_f]})
    train["cls"] = train["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    train.to_csv(os.path.join(data_dir, "{}.train.csv".format(train_name)), index=False, sep="\t")

    dev = pd.DataFrame({"labels": [x[0] for x in dev_f], "text": [x[1] for x in dev_f]})
    dev["cls"] = dev["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    dev.to_csv(os.path.join(data_dir, "{}.dev.csv".format(dev_name)), index=False, sep="\t")

    test_ = pd.DataFrame({"labels": [x[0] for x in test_f], "text": [x[1] for x in test_f]})
    test_["cls"] = test_["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    test_.to_csv(os.path.join(data_dir, "{}.dev.csv".format(test_name)), index=False, sep="\t")


def read_data(input_file):
    """Reads a BIO data."""
    with codecs.open(input_file, "r", encoding="utf-8") as f:
        lines = []
        words = []
        labels = []
        f_lines = f.readlines()
        for line in tqdm(f_lines, total=len(f_lines), desc="Process {}".format(input_file)):
            contends = line.strip()
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue

            if len(contends) == 0 and not len(words):
                words.append("")

            if len(contends) == 0 and words[-1] == '.':
                lbl = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([lbl, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label.replace("-", "_"))
        return lines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--train_name', type=str, default="eng.train")
    parser.add_argument('--dev_name', type=str, default="eng.testa")
    parser.add_argument('--test_name', type=str, default="eng.testb")
    return vars(parser.parse_args())


if __name__ == "__main__":
    conll2003_preprocess(**parse_args())
