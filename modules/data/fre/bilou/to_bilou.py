# -*- coding: utf-8 -*-
from ..entity.taggedtoken import TaggedToken


def get_tagged_tokens_from(dict_of_nes, token_list):
    list_of_tagged_tokens = [TaggedToken('O', token_list[i]) for i in range(len(token_list))]
    dict_of_tokens_with_indexes = {token_list[i].id: i for i in range(len(token_list))}

    for ne in dict_of_nes.values():
        for tokenid in ne['tokens_list']:
            try:
                tag = format_tag(tokenid, ne)
            except ValueError:
                tag = "O"
            id_in_token_tuple = dict_of_tokens_with_indexes[tokenid]
            token = token_list[id_in_token_tuple]
            list_of_tagged_tokens[id_in_token_tuple] = TaggedToken(tag, token)
    return list_of_tagged_tokens


def format_tag(tokenid, ne):
    bilou = __choose_bilou_tag_for(tokenid, ne['tokens_list'])
    formatted_tag = __tag_to_fact_ru_eval_format(ne['tag'])
    return "{}_{}".format(bilou, formatted_tag)


def __choose_bilou_tag_for(token_id, token_list):
    if len(token_list) == 1:
        return 'B'
    elif len(token_list) > 1:
        if token_list.index(token_id) == 0:
            return 'B'
        else:
            return 'I'


def __tag_to_fact_ru_eval_format(tag):
    if tag == 'Person':
        return 'PER'
    elif tag == 'Org':
        return 'ORG'
    elif tag == 'Location':
        return 'LOC'
    elif tag == 'LocOrg':
        return 'LOC'
    elif tag == 'Project':
        return 'ORG'
    else:
        raise ValueError('tag ' + tag + " is not the right tag")
