# -*- coding: utf-8 -*-


class Token(object):
    __token_id__ = 0

    @property
    def length(self):
        return self.__length

    @property
    def position(self):
        return self.__position

    @property
    def id(self):
        return self.__id

    @property
    def text(self):
        return self.__text

    @property
    def all(self):
        return self.__id, self.__position, self.__length, self.__text

    @property
    def tag(self):
        return self.tag

    def __init__(self, token_id=None, position=None, length=None, text=None):
        self.__id = token_id
        if token_id is None:
            self.__id = Token.__token_id__
            Token.__token_id__ += 1
        self.__position = position
        self.__length = length
        self.__text = text
        self.__tag = None

    def from_sting(self, string):
        self.__id, self.__position, self.__length, self.__text = string
        return self

    def __len__(self):
        return self.__length

    def __str__(self):
        return self.__text

    def __repr__(self):
        return "<<" + self.__id + "_" + self.__text + ">>"
