import codecs
from .token import Token
from .taggedtoken import TaggedToken
from collections import defaultdict
from ..bilou import to_bilou


class Document(object):
    def __init__(self, path, tagged=True, encoding="utf-8"):
        self.path = path
        self.tagged = tagged
        self.encoding = encoding
        self.tokens = []
        self.tagged_tokens = []
        self.load()

    def to_text_tokens(self):
        return [token.text for token in self.tokens]

    def get_tags(self):
        return [token.get_tag() for token in self.tagged_tokens]

    def load(self):
        self.tokens = self.__get_tokens_from_file()
        if self.tagged:
            self.tagged_tokens = self.__get_tagged_tokens_from()
        else:
            self.tagged_tokens = [TaggedToken(None, token) for token in self.tokens]
        return self

    def parse_file(self, path):
        with codecs.open(path, 'r', encoding=self.encoding, errors="ignore") as file:
            rows = file.read().split('\n')
        return [row.split(' # ')[0].split() for row in rows if len(row) != 0]

    def __get_tokens_from_file(self):
        rows = self.parse_file(self.path + '.tokens')
        tokens = []
        for token_str in rows:
            tokens.append(Token().from_sting(token_str))
        return tokens

    def __get_tagged_tokens_from(self):
        span_dict = self.__span_id2token_ids(self.path + '.spans', [token.id for token in self.tokens])
        object_dict = self.__to_dict_of_objects(self.path + '.objects')
        dict_of_nes = self.__merge(object_dict, span_dict, self.tokens)
        return to_bilou.get_tagged_tokens_from(dict_of_nes, self.tokens)

    def __span_id2token_ids(self, span_file, token_ids):
        span_list = self.parse_file(span_file)
        dict_of_spans = {}
        for span in span_list:
            span_id = span[0]
            span_start = span[4]
            span_length_in_tokens = int(span[5])
            list_of_token_of_spans = self.__find_tokens_for(span_start, span_length_in_tokens, token_ids)
            dict_of_spans[span_id] = list_of_token_of_spans
        return dict_of_spans

    @staticmethod
    def __find_tokens_for(start, length, token_ids):
        list_of_tokens = []
        index = token_ids.index(start)
        for i in range(length):
            list_of_tokens.append(token_ids[index + i])
        return list_of_tokens

    def __to_dict_of_objects(self, object_file):
        object_list = self.parse_file(object_file)
        dict_of_objects = {}
        for obj in object_list:
            object_id = obj[0]
            object_tag = obj[1]
            object_spans = obj[2:]
            dict_of_objects[object_id] = {'tag': object_tag, 'spans': object_spans}
        return dict_of_objects

    def __merge(self, object_dict, span_dict, tokens):
        ne_dict = self.__get_dict_of_nes(object_dict, span_dict)
        return self.__clean(ne_dict, tokens)

    @staticmethod
    def __get_dict_of_nes(object_dict, span_dict):
        ne_dict = defaultdict(set)
        for obj_id, obj_values in object_dict.items():
            for span in obj_values['spans']:
                ne_dict[(obj_id, obj_values['tag'])].update(span_dict[span])
        for ne in ne_dict:
            ne_dict[ne] = sorted(list(set([int(i) for i in ne_dict[ne]])))
        return ne_dict

    def __clean(self, ne_dict, tokens):
        sorted_nes = sorted(ne_dict.items(), key=self.__sort_by_tokens)
        dict_of_tokens_by_id = {}
        for i in range(len(tokens)):
            dict_of_tokens_by_id[tokens[i].id] = i
        result_nes = {}
        if len(sorted_nes) != 0:
            start_ne = sorted_nes[0]
            for ne in sorted_nes:
                if self.__not_intersect(start_ne[1], ne[1]):
                    result_nes[start_ne[0][0]] = {
                        'tokens_list': self.__check_order(start_ne[1], dict_of_tokens_by_id, tokens),
                        'tag': start_ne[0][1]}
                    start_ne = ne
                else:
                    result_tokens_list = self.__check_normal_form(start_ne[1], ne[1])
                    start_ne = (start_ne[0], result_tokens_list)
            result_nes[start_ne[0][0]] = {
                'tokens_list': self.__check_order(start_ne[1], dict_of_tokens_by_id, tokens),
                'tag': start_ne[0][1]}
        return result_nes

    @staticmethod
    def __sort_by_tokens(tokens):
        ids_as_int = [int(token_id) for token_id in tokens[1]]
        return min(ids_as_int), -max(ids_as_int)

    @staticmethod
    def __not_intersect(start_ne, current_ne):
        intersection = set.intersection(set(start_ne), set(current_ne))
        return intersection == set()

    def __check_normal_form(self, start_ne, ne):
        all_tokens = set.union(set(start_ne), set(ne))
        return self.__find_all_range_of_tokens(all_tokens)

    @staticmethod
    def __find_all_range_of_tokens(tokens):
        tokens = sorted(tokens)
        if (tokens[-1] - tokens[0] - len(tokens)) < 5:
            return list(range(tokens[0], tokens[-1] + 1))
        else:
            return tokens

    def __check_order(self, list_of_tokens, dict_of_tokens_by_id, tokens):
        list_of_tokens = [str(i) for i in self.__find_all_range_of_tokens(list_of_tokens)]
        result = []
        for token in list_of_tokens:
            if token in dict_of_tokens_by_id:
                result.append((token, dict_of_tokens_by_id[token]))
        result = sorted(result, key=self.__sort_by_position)
        result = self.__add_quotation_marks(result, tokens)
        return [r[0] for r in result]

    @staticmethod
    def __sort_by_position(result_tuple):
        return result_tuple[1]

    @staticmethod
    def __add_quotation_marks(result, tokens):
        result_tokens_texts = [tokens[token[1]].text for token in result]
        prev_pos = result[0][1] - 1
        next_pos = result[-1][1] + 1

        if prev_pos >= 0 and tokens[prev_pos].text == '«' \
                and '»' in result_tokens_texts and '«' not in result_tokens_texts:
            result = [(tokens[prev_pos].id, prev_pos)] + result

        if next_pos < len(tokens) and tokens[next_pos].text == '»' \
                and '«' in result_tokens_texts and '»' not in result_tokens_texts:
            result = result + [(tokens[next_pos].id, next_pos)]

        return result
