from cntk.layers import *
from cntk.layers.blocks import _INFERRED

import tsv2ctf


def BiRNN(hidden_dim, num_layers=1, recurrent_op='gru', use_cudnn=True, name=''):
    if use_cudnn:
        W = C.parameter(_INFERRED + (hidden_dim,), init=C.glorot_uniform())

        @C.Function
        def cuDNN_bigru(x):
            return C.optimized_rnnstack(x, W, hidden_dim, num_layers, True, recurrent_op='gru',
                                        name=name)

        return cuDNN_bigru
    else:
        cell = {
            'gru': [GRU(hidden_dim), GRU(hidden_dim)],
            'lstm': [LSTM(hidden_dim), LSTM(hidden_dim)]
        }
        if recurrent_op == 'gru':
            fwd = GRU(hidden_dim)
            bwd = GRU(hidden_dim)
        elif recurrent_op == 'lstm':
            fwd = LSTM(hidden_dim)
            bwd = LSTM(hidden_dim)
        else:
            raise ValueError('no such recurrent_op!')

        return Sequential([
            (Recurrence(fwd),
             Recurrence(bwd, go_backwards=True)),
            splice
        ], name=name)


def seq_loss(prob, y):
    return -C.log(C.sequence.last(C.sequence.gather(prob, y)))


def all_spans_loss(start_logits, start_y, end_logits, end_y):
    # this works as follows:
    # let end_logits be A, B, ..., Y, Z
    # let start_logits be a, b, ..., y, z
    # the tricky part is computing log sum (i<=j) exp(start_logits[i] + end_logits[j])
    # we break this problem as follows
    # x = logsumexp(A, B, ..., Y, Z), logsumexp(B, ..., Y, Z), ..., logsumexp(Y, Z), Z
    # y = a + logsumexp(A, B, ..., Y, Z), b + logsumexp(B, ..., Y, Z), ..., y + logsumexp(Y, Z), z + Z
    # now if we exponentiate each element in y we have all the terms we need. We just need to sum those exponentials...
    # logZ = last(sequence.logsumexp(y))
    x = C.layers.Recurrence(C.log_add_exp, go_backwards=True, initial_state=-1e+30)(end_logits)

    y = start_logits + x
    logZ = C.layers.Fold(C.log_add_exp, initial_state=-1e+30)(y)
    return logZ - C.sequence.last(C.sequence.gather(start_logits, start_y)) - C.sequence.last(
        C.sequence.gather(end_logits, end_y))

def seq_hardmax(logits):
    seq_max = C.layers.Fold(C.element_max, initial_state=C.constant(-1e+30, logits.shape))(logits)
    s = C.equal(logits, C.sequence.broadcast_as(seq_max, logits))
    s_acc = C.layers.Recurrence(C.plus)(s)
    return s * C.equal(s_acc, 1) # only pick the first one

# map from token to char offset
def w2c_map(s, words):
    w2c = []
    rem = s
    offset = 0
    for i, w in enumerate(words):
        cidx = rem.find(w)
        assert (cidx >= 0)
        w2c.append(cidx + offset)
        offset += cidx + len(w)
        rem = rem[cidx + len(w):]
    return w2c


# get phrase from string based on tokens and their offsets


def get_answer(raw_text, tokens, start, end):
    w2c = w2c_map(raw_text, tokens)
    return raw_text[w2c[start]:w2c[end] + len(tokens[end])]


def symbolic_best_span(begin, end):
    running_max_begin = C.layers.Recurrence(
        C.element_max, initial_state=-float("inf"))(begin)
    return C.layers.Fold(C.element_max, initial_state=C.constant(-1e+30))(running_max_begin + end)


def create_tsv_reader(func, tsv_file, model, seqs, num_workers,  misc=None):
    with open(tsv_file, 'r', encoding='utf-8') as f:
        eof = False
        batch_count = 0
        while not (eof and (batch_count % num_workers) == 0):
            batch_count += 1
            batch = {'cwids': [], 'qwids': [], 'baidx': [],'eaidx': []}

            while not eof and len(batch['cwids']) < seqs:
                line = f.readline()
                if not line:
                    eof = True
                    break

                if misc is not None:
                    import re
                    misc['uid'].append(re.match('^([^\t]*)', line).groups()[0])

                    title, ctokens, qtokens, atokens, cwids, qwids, awids, baidx, eaidx = tsv2ctf.tsv_iter(
                        line, model.vocab, model.chars, False, misc)

                batch['cwids'].append(cwids)
                batch['qwids'].append(qwids)
                batch['baidx'].append(baidx)
                batch['eaidx'].append(eaidx)

            if len(batch['cwids']) > 0:
                context_g_words = C.Value.one_hot(
                    [[C.Value.ONE_HOT_SKIP if i >= model.wg_dim else i for i in cwids] for cwids in batch['cwids']],
                    model.wg_dim)
                context_ng_words = C.Value.one_hot(
                    [[C.Value.ONE_HOT_SKIP if i < model.wg_dim else i - model.wg_dim for i in cwids] for cwids in
                     batch['cwids']], model.wn_dim)
                query_g_words = C.Value.one_hot(
                    [[C.Value.ONE_HOT_SKIP if i >= model.wg_dim else i for i in qwids] for qwids in batch['qwids']],
                    model.wg_dim)
                query_ng_words = C.Value.one_hot(
                    [[C.Value.ONE_HOT_SKIP if i < model.wg_dim else i - model.wg_dim for i in qwids] for qwids in
                     batch['qwids']], model.wn_dim)

                yield {
                    argument_by_name(func, 'passage_gw'): context_g_words,
                    argument_by_name(func, 'question_gw'): query_g_words,
                    argument_by_name(func, 'passage_nw'): context_ng_words,
                    argument_by_name(func, 'question_nw'): query_ng_words,
                }
            else:
                yield {}  # need to generate empty batch for distributed training


def argument_by_name(func, name):
    found = [arg for arg in func.arguments if arg.name == name]
    if len(found) == 0:
        raise ValueError('no matching names in arguments')
    elif len(found) > 1:
        raise ValueError('multiple matching names in arguments')
    else:
        return found[0]
