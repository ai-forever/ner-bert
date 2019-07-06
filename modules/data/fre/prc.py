from modules.data.fre.reader import Reader
import pandas as pd
from modules import tqdm
import argparse


def fact_ru_eval_preprocess(dev_dir, test_dir, dev_df_path, test_df_path):
    dev_reader = Reader(dev_dir)
    dev_reader.read_dir()
    dev_texts, dev_tags = dev_reader.split()
    res_tags = []
    res_tokens = []
    for tag, tokens in tqdm(zip(dev_tags, dev_texts), total=len(dev_tags), desc="Process FactRuEval2016 dev set."):
        if len(tag):
            res_tags.append(tag)
            res_tokens.append(tokens)
    dev = pd.DataFrame({"labels": list(map(" ".join, res_tags)), "text": list(map(" ".join, res_tokens))})
    dev["clf"] = dev["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    dev.to_csv(dev_df_path, index=False, sep="\t")

    test_reader = Reader(test_dir)
    test_reader.read_dir()
    test_texts, test_tags = test_reader.split()
    res_tags = []
    res_tokens = []
    for tag, tokens in tqdm(zip(test_tags, test_texts), total=len(test_tags), desc="Process FactRuEval2016 test set."):
        if len(tag):
            res_tags.append(tag)
            res_tokens.append(tokens)
    valid = pd.DataFrame({"labels": list(map(" ".join, res_tags)), "text": list(map(" ".join, res_tokens))})
    valid["clf"] = valid["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    valid.to_csv(test_df_path, index=False, sep="\t")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--dev_dir', type=str)
    parser.add_argument('-td', '--test_dir', type=str)
    parser.add_argument('-ddp', '--dev_df_path', type=str)
    parser.add_argument('-tdp', '--test_df_path', type=str)
    return vars(parser.parse_args())


if __name__ == "__main__":
    fact_ru_eval_preprocess(**parse_args())
