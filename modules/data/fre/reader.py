# -*- coding: utf-8 -*-
import pandas as pd
from .utils import get_file_names
from .entity.document import Document


class Reader(object):

    def __init__(self,
                 dir_path,
                 document_creator=Document,
                 get_file_names_=get_file_names,
                 tagged=True):
        self.path = dir_path
        self.tagged = tagged
        self.documents = []
        self.document_creator = document_creator
        self.get_file_names = get_file_names_

    def split(self, use_morph=False):
        res_texts = []
        res_tags = []
        for doc in self.documents:
            sent_tokens = []
            sent_tags = []
            for token in doc.tagged_tokens:
                if token.get_tag() == "O" and token.text == ".":
                    res_texts.append(tuple(sent_tokens))
                    res_tags.append(tuple(sent_tags))
                    sent_tokens = []
                    sent_tags = []
                else:
                    text = token.text
                    sent_tokens.append(text)
                    sent_tags.append(token.get_tag())
        if use_morph:
            return res_texts, res_tags
        return res_texts, res_tags

    def to_data_frame(self, split=False):
        if split:
            docs = self.split()
        else:
            docs = []
            for doc in self.documents:
                docs.append([(token.text, token.get_tag()) for token in doc.tagged_tokens])

        texts = []
        tags = []
        for sent in docs:
            sample_text = []
            sample_tag = []
            for text, tag in sent:
                sample_text.append(text)
                sample_tag.append(tag)
            texts.append(" ".join(sample_text))
            tags.append(" ".join(sample_tag))
        return pd.DataFrame({"texts": texts, "tags": tags}, columns=["texts", "tags"])

    def read_dir(self):
        for path in self.get_file_names(self.path):
            self.documents.append(self.document_creator(path, self.tagged))

    def get_text_tokens(self):
        return [doc.to_text_tokens() for doc in self.documents]

    def get_text_tags(self):
        return [doc.get_tags() for doc in self.documents]
