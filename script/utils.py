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
    # s_acc = C.layers.Recurrence(C.plus)(s)
    # result=s * C.equal(s_acc, 1) # only pick the first one
    return s

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


# def tsv_iter(line, vocab, chars, is_test=False, misc={}):
#     EMPTY_TOKEN = '<NULL>'
#     unk = '<UNK>'
#     unk_w = vocab[unk]
#
#     if is_test:
#         uid, title, context, query = line.split('\t')
#         answer = ''
#         begin_answer, end_answer = '0', '1'
#     else:
#         uid, title, context, query, answer, raw_context, begin_answer, end_answer, raw_answer = line.split('\t')
#         # uid, title, context, query, begin_answer, end_answer, answer = line.split('\t')
#
#     ctokens = context.split(' ')
#     qtokens = query.split(' ')
#     atokens = answer.split(' ')
#
#     ba, ea = int(begin_answer), int(end_answer) - 1  # the end from tsv is exclusive
#
#     if ba > ea:
#         raise ValueError('answer problem with input line:\n%s' % line)
#
#     if not is_test:
#         mtokens = ctokens[ba:ea + 1]
#
#     # replace EMPTY_TOKEN with ''
#     ctokens = [t if t != EMPTY_TOKEN else '' for t in ctokens]
#     qtokens = [t if t != EMPTY_TOKEN else '' for t in qtokens]
#     atokens = [t if t != EMPTY_TOKEN else '' for t in atokens]
#
#     cwids = [vocab.get(t.lower(), unk_w) for t in ctokens]
#     qwids = [vocab.get(t.lower(), unk_w) for t in qtokens]
#     awids = [vocab.get(t.lower(), unk_w) for t in atokens]
#     # ccids = [[chars.get(c, unk_c) for c in t][:word_size] for t in ctokens]  # clamp at word_size
#     # qcids = [[chars.get(c, unk_c) for c in t][:word_size] for t in qtokens]
#     # acids = [[chars.get(c, unk_c) for c in t][:word_size] for t in atokens]
#
#     baidx = [0 if i != ba else 1 for i, t in enumerate(ctokens)]
#     eaidx = [0 if i != ea else 1 for i, t in enumerate(ctokens)]
#
#     if not is_test and sum(eaidx) == 0:
#         raise ValueError('problem with input line:\n%s' % line)
#     if misc.keys():
#         misc['answer'] += [answer]
#         misc['rawctx'] += [context]
#         misc['ctoken'] += [ctokens]
#
#     return title, ctokens, qtokens, atokens, cwids, qwids, awids, baidx, eaidx


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

def create_mb_and_map(func, data_file, ee_model, randomize=True, repeat=True):
    mb_source = C.io.MinibatchSource(
        C.io.CTFDeserializer(
            data_file,
            C.io.StreamDefs(
                context_g_words=C.io.StreamDef(
                    'cgw', shape=ee_model.wg_dim, is_sparse=True),
                query_g_words=C.io.StreamDef(
                    'qgw', shape=ee_model.wg_dim, is_sparse=True),
                context_ng_words=C.io.StreamDef(
                    'cnw', shape=ee_model.wn_dim, is_sparse=True),
                query_ng_words=C.io.StreamDef(
                    'qnw', shape=ee_model.wn_dim, is_sparse=True),
                answer_begin=C.io.StreamDef(
                    'ab', shape=ee_model.a_dim, is_sparse=False),
                answer_end=C.io.StreamDef(
                    'ae', shape=ee_model.a_dim, is_sparse=False),
            )
        ),
        randomize=randomize,
        max_sweeps=C.io.INFINITELY_REPEAT if repeat else 1
    )

    input_map = {
        argument_by_name(func, 'passage_gw'): mb_source.streams.context_g_words,
        argument_by_name(func, 'question_gw'): mb_source.streams.query_g_words,
        argument_by_name(func, 'passage_nw'): mb_source.streams.context_ng_words,
        argument_by_name(func, 'question_nw'): mb_source.streams.query_ng_words,
        argument_by_name(func, 'begin'): mb_source.streams.answer_begin,
        argument_by_name(func, 'end'): mb_source.streams.answer_end
    }
    return mb_source, input_map

def validate(test_data, model, ee_model):
    begin_prob = model.outputs[0]
    end_prob = model.outputs[1]
    loss = model.outputs[2]
    root = C.as_composite(loss.owner)
    mb_source, input_map = create_mb_and_map(
        root, test_data, ee_model, randomize=False, repeat=False)
    begin_label = argument_by_name(root, 'begin')
    end_label = argument_by_name(root, 'end')


    # rouge-L
    begin_prediction = C.sequence.input_variable(
        1, sequence_axis=begin_label.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input_variable(
        1, sequence_axis=end_label.dynamic_axes[1], needs_gradient=True)

    best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    predicted_span = C.layers.Recurrence(C.plus)(
        begin_prediction - C.sequence.past_value(end_prediction))
    true_span = C.layers.Recurrence(C.plus)(
        begin_label - C.sequence.past_value(end_label))
    common_span = C.element_min(predicted_span, true_span)
    begin_match = C.sequence.reduce_sum(
        C.element_min(begin_prediction, begin_label))
    end_match = C.sequence.reduce_sum(C.element_min(end_prediction, end_label))

    predicted_len = C.sequence.reduce_sum(predicted_span)
    true_len = C.sequence.reduce_sum(true_span)
    common_len = C.sequence.reduce_sum(common_span)
    f1 = 2 * common_len / (predicted_len + true_len)
    exact_match = C.element_min(begin_match, end_match)
    precision = common_len / predicted_len
    recall = common_len / true_len
    overlap = C.greater(common_len, 0)

    def s(x):
        return C.reduce_sum(x, axis=C.Axis.all_axes())

    stats = C.splice(s(f1), s(exact_match), s(precision), s(
        recall), s(overlap), s(begin_match), s(end_match))

    # Evaluation parameters
    minibatch_size = 64
    num_sequences = 0

    stat_sum = 0
    loss_sum = 0
    while True:
        data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
        if not data or not (begin_label in data) or data[begin_label].num_sequences == 0:
            break
        out = model.eval(
            data, outputs=[begin_prob, end_prob, loss], as_numpy=False)
        testloss = out[loss]
        other_input_map = {begin_prediction: out[begin_prob], end_prediction: out[end_prob],
                           begin_label: data[begin_label], end_label: data[end_label]}
        stat_sum += stats.eval((other_input_map))
        loss_sum += np.sum(testloss.asarray())
        num_sequences += data[begin_label].num_sequences

    stat_avg = stat_sum / num_sequences
    loss_avg = loss_sum / num_sequences

    print(
        "Validated {} sequences, loss {:.4f}, F1 {:.4f}, EM {:.4f}, precision {:4f}, recall {:4f} hasOverlap {:4f}, start_match {:4f}, end_match {:4f}".format(
            num_sequences,
            loss_avg,
            stat_avg[0],
            stat_avg[1],
            stat_avg[2],
            stat_avg[3],
            stat_avg[4],
            stat_avg[5],
            stat_avg[6]))

    return loss_sum


def argument_by_name(func, name):
    found = [arg for arg in func.arguments if arg.name == name]
    if len(found) == 0:
        raise ValueError('no matching names in arguments')
    elif len(found) > 1:
        raise ValueError('multiple matching names in arguments')
    else:
        return found[0]
