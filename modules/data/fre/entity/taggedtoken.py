# -*- coding: utf-8 -*-


class TaggedToken(object):

    @property
    def text(self):
        return self.__token.text

    def __init__(self, tag, token):
        self.__tag = tag
        self.__token = token

    def get_token(self):
        return self.__token

    def get_tag(self):
        return self.__tag

    def __repr__(self):
        if self.__tag:
            return "<" + self.__tag + "_" + str(self.__token) + ">"
        else:
            return "<None_" + str(self.__token) + ">"
