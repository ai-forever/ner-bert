# -*- coding: utf-8 -*-


def untag(list_of_tags, list_of_tokens):
    """
    :param list_of_tags:
    :param list_of_tokens:
    :return:
    """
    if len(list_of_tags) == len(list_of_tokens):
        dict_of_final_ne = {}
        ne_words = []
        ne_tag = None

        for index in range(len(list_of_tokens)):
            if not ((ne_tag is not None) ^ (ne_words != [])):
                current_tag = list_of_tags[index]
                current_token = list_of_tokens[index]

                if current_tag.startswith('B') or current_tag.startswith('I'):
                    dict_of_final_ne, ne_words, ne_tag = __check_bi(
                        dict_of_final_ne, ne_words, ne_tag, current_tag, current_token)
                elif current_tag.startswith('L'):
                    dict_of_final_ne, ne_words, ne_tag = __check_l(
                        dict_of_final_ne, ne_words, ne_tag, current_tag, current_token)
                elif current_tag.startswith('O'):
                    dict_of_final_ne, ne_words, ne_tag = __finish_ne_if_required(dict_of_final_ne, ne_words, ne_tag)

                elif current_tag.startswith('U'):
                    dict_of_final_ne, ne_words, ne_tag = __check_u(dict_of_final_ne, ne_words, ne_tag, current_tag,
                                                                   current_token)
                else:
                    raise ValueError("tag contains no BILOU tags")
            else:
                if ne_tag is None:
                    raise Exception('Somehow ne_tag is None and ne_words is not None')
                else:
                    raise Exception('Somehow ne_words is None and ne_tag is not None')

        dict_of_final_ne, ne_words, ne_tag = __finish_ne_if_required(dict_of_final_ne, ne_words, ne_tag)
        return __to_output_format(dict_of_final_ne)
    else:
        raise ValueError('lengths are not equal')


def __check_bi(dict_of_final_ne, ne_words, ne_tag, current_tag, current_token):
    if ne_tag is None and ne_words == []:
        ne_tag = current_tag[1:]
        ne_words = [current_token]
    else:
        if current_tag.startswith('I') and ne_tag == current_tag[1:]:
            ne_words.append(current_token)
        else:
            dict_of_final_ne, ne_words, ne_tag = __replace_by_new(dict_of_final_ne, ne_words, ne_tag, current_tag,
                                                                  current_token)
    return dict_of_final_ne, ne_words, ne_tag


def __check_l(dict_of_final_ne, ne_words, ne_tag, current_tag, current_token):
    if ne_tag == current_tag[1:]:
        dict_of_final_ne, ne_words, ne_tag = __finish_ne_if_required(dict_of_final_ne, ne_words+[current_token], ne_tag)
    else:
        dict_of_final_ne, ne_words, ne_tag = __finish_ne_if_required(dict_of_final_ne, ne_words, ne_tag)
        dict_of_final_ne, ne_words, ne_tag = __finish_ne_if_required(dict_of_final_ne, [current_token], current_tag[1:])
    return dict_of_final_ne, ne_words, ne_tag


def __check_u(dict_of_final_ne, ne_words, ne_tag, current_tag, current_token):
    dict_of_final_ne, ne_words, ne_tag = __finish_ne_if_required(dict_of_final_ne, ne_words, ne_tag)
    return __finish_ne_if_required(dict_of_final_ne, [current_token], current_tag[1:])


def __replace_by_new(dict_of_final_ne, ne_words, ne_tag, current_tag, current_token):
    dict_of_final_ne, ne_words, ne_tag = __finish_ne_if_required(dict_of_final_ne, ne_words, ne_tag)
    ne_tag = current_tag[1:]
    ne_words = [current_token]
    return dict_of_final_ne, ne_words, ne_tag


def __finish_ne_if_required(dict_of_final_ne, ne_words, ne_tag):
    if ne_tag is not None and ne_words != []:
        dict_of_final_ne[tuple(ne_words)] = ne_tag
        ne_tag = None
        ne_words = []
    return dict_of_final_ne, ne_words, ne_tag


def __to_output_format(dict_nes):
    """
    :param dict_nes:
    :return:
    """
    list_of_results_for_output = []

    for tokens_tuple, tag in dict_nes.items():
        position = int(tokens_tuple[0].get_position())
        length = int(tokens_tuple[-1].get_position()) + int(tokens_tuple[-1].get_length()) - position
        list_of_results_for_output.append([tag, position, length])

    return list_of_results_for_output
