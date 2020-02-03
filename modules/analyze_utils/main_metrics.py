# This code is reused from https://github.com/deepmipt/DeepPavlov/blob/master/deeppavlov/metrics/fmeasure.py
import itertools
from collections import OrderedDict


def chunk_finder(current_token, previous_token, tag):
    current_tag = current_token.split('_', 1)[-1]
    previous_tag = previous_token.split('_', 1)[-1]
    if previous_tag != tag:
        previous_tag = 'O'
    if current_tag != tag:
        current_tag = 'O'
    if (previous_tag == 'O' and current_token == 'B_' + tag) or \
            (previous_token == 'I_' + tag and current_token == 'B_' + tag) or \
            (previous_token == 'B_' + tag and current_token == 'B_' + tag) or \
            (previous_tag == 'O' and current_token == 'I_' + tag):
        create_chunk = True
    else:
        create_chunk = False

    if (previous_token == 'I_' + tag and current_token == 'B_' + tag) or \
            (previous_token == 'B_' + tag and current_token == 'B_' + tag) or \
            (current_tag == 'O' and previous_token == 'I_' + tag) or \
            (current_tag == 'O' and previous_token == 'B_' + tag):
        pop_out = True
    else:
        pop_out = False
    return create_chunk, pop_out


def _global_stats_f1(results):
    total_true_entities = 0
    total_predicted_entities = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_correct = 0
    for tag in results:
        if tag == '__total__':
            continue

        n_pred = results[tag]['n_pred']
        n_true = results[tag]['n_true']
        total_correct += results[tag]['tp']
        total_true_entities += n_true
        total_predicted_entities += n_pred
        total_precision += results[tag]['precision'] * n_pred
        total_recall += results[tag]['recall'] * n_true
        total_f1 += results[tag]['f1'] * n_true
    if total_true_entities > 0:
        accuracy = total_correct / total_true_entities * 100
        total_recall = total_recall / total_true_entities
    else:
        accuracy = 0
        total_recall = 0
    if total_predicted_entities > 0:
        total_precision = total_precision / total_predicted_entities
    else:
        total_precision = 0

    if total_precision + total_recall > 0:
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    else:
        total_f1 = 0

    total_res = {'n_predicted_entities': total_predicted_entities,
                 'n_true_entities': total_true_entities,
                 'precision': total_precision,
                 'recall': total_recall,
                 'f1': total_f1}
    return total_res, accuracy, total_true_entities, total_predicted_entities, total_correct


def precision_recall_f1(y_true, y_pred, print_results=True, short_report=False, entity_of_interest=None):
    y_true = list(itertools.chain(*y_true))
    y_pred = list(itertools.chain(*y_pred))
    # Find all tags
    tags = set()
    for tag in itertools.chain(y_true, y_pred):
        if tag not in ["O", "I_O", "B_O"]:
            current_tag = tag[2:]
            tags.add(current_tag)
    tags = sorted(list(tags))

    results = OrderedDict()
    for tag in tags:
        results[tag] = OrderedDict()
    results['__total__'] = OrderedDict()
    n_tokens = len(y_true)
    # Firstly we find all chunks in the ground truth and prediction
    # For each chunk we write starting and ending indices

    for tag in tags:
        count = 0
        true_chunk = []
        pred_chunk = []
        y_true = [str(y) for y in y_true]
        y_pred = [str(y) for y in y_pred]
        prev_tag_true = 'O'
        prev_tag_pred = 'O'
        while count < n_tokens:
            yt = y_true[count]
            yp = y_pred[count]

            create_chunk_true, pop_out_true = chunk_finder(yt, prev_tag_true, tag)
            if pop_out_true:
                true_chunk[-1] = (true_chunk[-1], count - 1)
            if create_chunk_true:
                true_chunk.append(count)

            create_chunk_pred, pop_out_pred = chunk_finder(yp, prev_tag_pred, tag)
            if pop_out_pred:
                pred_chunk[-1] = (pred_chunk[-1], count - 1)
            if create_chunk_pred:
                pred_chunk.append(count)
            prev_tag_true = yt
            prev_tag_pred = yp
            count += 1

        if len(true_chunk) > 0 and not isinstance(true_chunk[-1], tuple):
            true_chunk[-1] = (true_chunk[-1], count - 1)
        if len(pred_chunk) > 0 and not isinstance(pred_chunk[-1], tuple):
            pred_chunk[-1] = (pred_chunk[-1], count - 1)

        # Then we find all correctly classified intervals
        # True positive results
        tp = len(set(pred_chunk).intersection(set(true_chunk)))
        # And then just calculate errors of the first and second kind
        # False negative
        fn = len(true_chunk) - tp
        # False positive
        fp = len(pred_chunk) - tp
        if tp + fp > 0:
            precision = tp / (tp + fp) * 100
        else:
            precision = 0
        if tp + fn > 0:
            recall = tp / (tp + fn) * 100
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        results[tag]['precision'] = precision
        results[tag]['recall'] = recall
        results[tag]['f1'] = f1
        results[tag]['n_pred'] = len(pred_chunk)
        results[tag]['n_true'] = len(true_chunk)
        results[tag]['tp'] = tp
        results[tag]['fn'] = fn
        results[tag]['fp'] = fp

    results['__total__'], accuracy, total_true_entities, total_predicted_entities, total_correct = _global_stats_f1(results)
    results['__total__']['n_pred'] = total_predicted_entities
    results['__total__']['n_true'] = total_true_entities
    results['__total__']["n_tokens"] = n_tokens
    if print_results:
        _print_conll_report(results, short_report, entity_of_interest)
    return results


def _print_conll_report(results, short_report=False, entity_of_interest=None):
    _, accuracy, total_true_entities, total_predicted_entities, total_correct = _global_stats_f1(results)
    n_tokens = results['__total__']["n_tokens"]
    tags = list(results.keys())

    s = 'processed {len} tokens ' \
        'with {tot_true} phrases; ' \
        'found: {tot_pred} phrases;' \
        ' correct: {tot_cor}.\n\n'.format(len=n_tokens,
                                          tot_true=total_true_entities,
                                          tot_pred=total_predicted_entities,
                                          tot_cor=total_correct)

    s += 'precision:  {tot_prec:.2f}%; ' \
         'recall:  {tot_recall:.2f}%; ' \
         'FB1:  {tot_f1:.2f}\n\n'.format(acc=accuracy,
                                         tot_prec=results['__total__']['precision'],
                                         tot_recall=results['__total__']['recall'],
                                         tot_f1=results['__total__']['f1'])

    if not short_report:
        for tag in tags:
            if entity_of_interest is not None:
                if entity_of_interest in tag:
                    s += '\t' + tag + ': precision:  {tot_prec:.2f}%; ' \
                                      'recall:  {tot_recall:.2f}%; ' \
                                      'F1:  {tot_f1:.2f} ' \
                                      '{tot_predicted}\n\n'.format(tot_prec=results[tag]['precision'],
                                                                   tot_recall=results[tag]['recall'],
                                                                   tot_f1=results[tag]['f1'],
                                                                   tot_predicted=results[tag]['n_pred'])
            elif tag != '__total__':
                s += '\t' + tag + ': precision:  {tot_prec:.2f}%; ' \
                                  'recall:  {tot_recall:.2f}%; ' \
                                  'F1:  {tot_f1:.2f} ' \
                                  '{tot_predicted}\n\n'.format(tot_prec=results[tag]['precision'],
                                                               tot_recall=results[tag]['recall'],
                                                               tot_f1=results[tag]['f1'],
                                                               tot_predicted=results[tag]['n_pred'])
    elif entity_of_interest is not None:
        s += '\t' + entity_of_interest + ': precision:  {tot_prec:.2f}%; ' \
                          'recall:  {tot_recall:.2f}%; ' \
                          'F1:  {tot_f1:.2f} ' \
                          '{tot_predicted}\n\n'.format(tot_prec=results[entity_of_interest]['precision'],
                                                       tot_recall=results[entity_of_interest]['recall'],
                                                       tot_f1=results[entity_of_interest]['f1'],
                                                       tot_predicted=results[entity_of_interest]['n_pred'])
    print(s)
