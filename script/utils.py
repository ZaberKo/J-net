from cntk.layers import *
from cntk.layers.blocks import _inject_name

import tsv2ctf
from cntk.layers.blocks import _INFERRED

def BiGRU(hidden_dim, num_layers=1, use_cudnn=True, name=''):
    if use_cudnn:
        W = C.parameter(_INFERRED + (hidden_dim,), init=C.glorot_uniform())
        @C.Function
        def cuDNN_bigru(x):
            return C.optimized_rnnstack(x, W, hidden_dim, num_layers, True, recurrent_op='gru',
                                        name=name)

        return cuDNN_bigru
    else:
        return Sequential([
            (Recurrence(GRU(hidden_dim)),
             Recurrence(GRU(hidden_dim), go_backwards=True)),
            splice
        ], name=name)


def MyAttentionModel(attention_dim,
                     init=default_override_or(glorot_uniform()),
                     go_backwards=default_override_or(False),
                     enable_self_stabilization=default_override_or(True), name=''):
    '''
    AttentionModel(attention_dim, attention_span=None, attention_axis=None, init=glorot_uniform(), go_backwards=False, enable_self_stabilization=True, name='')

    Layer factory function to create a function object that implements an attention model
    as described in Bahdanau, et al., "Neural machine translation by jointly learning to align and translate."
    '''

    init = get_default_override(AttentionModel, init=init)
    go_backwards = get_default_override(AttentionModel, go_backwards=go_backwards)
    enable_self_stabilization = get_default_override(AttentionModel,
                                                     enable_self_stabilization=enable_self_stabilization)
    # model parameters
    with default_options(bias=False):  # all the projections have no bias
        attn_proj_enc = Stabilizer(enable_self_stabilization=enable_self_stabilization) >> Dense(attention_dim,
                                                                                                 init=init,
                                                                                                 input_rank=1)  # projects input hidden state, keeping span axes intact
        attn_proj_dec = Stabilizer(enable_self_stabilization=enable_self_stabilization) >> Dense(attention_dim,
                                                                                                 init=init,
                                                                                                 input_rank=1)  # projects decoder hidden state, but keeping span and beam-search axes intact
        attn_proj_tanh = Stabilizer(enable_self_stabilization=enable_self_stabilization) >> Dense(1, init=init,
                                                                                                  input_rank=1)  # projects tanh output, keeping span and beam-search axes intact
    attn_final_stab = Stabilizer(enable_self_stabilization=enable_self_stabilization)

    @Function
    def new_attention(encoder_hidden_state, decoder_hidden_state):
        # encode_hidden_state: [#, e] [h]
        # decoder_hidden_state: [#, d] [H]
        unpacked_encoder_hidden_state, valid_mask = C.sequence.unpack(encoder_hidden_state, padding_value=0).outputs
        # unpacked_encoder_hidden_state: [#] [*=e, h]
        # valid_mask: [#] [*=e]
        projected_encoder_hidden_state = C.sequence.broadcast_as(attn_proj_enc(unpacked_encoder_hidden_state),
                                                                 decoder_hidden_state)
        # projected_encoder_hidden_state: [#, d] [*=e, attention_dim]
        broadcast_valid_mask = C.sequence.broadcast_as(C.reshape(valid_mask, (1,), 1), decoder_hidden_state)
        # broadcast_valid_mask: [#, d] [*=e]
        projected_decoder_hidden_state = attn_proj_dec(decoder_hidden_state)
        # projected_decoder_hidden_state: [#, d] [attention_dim]
        tanh_output = C.tanh(projected_decoder_hidden_state + projected_encoder_hidden_state)
        # tanh_output: [#, d] [*=e, attention_dim]
        attention_logits = attn_proj_tanh(tanh_output)
        # attention_logits = [#, d] [*=e, 1]
        minus_inf = C.constant(-1e+30)
        masked_attention_logits = C.element_select(broadcast_valid_mask, attention_logits, minus_inf)
        # masked_attention_logits = [#, d] [*=e]
        attention_weights = C.softmax(masked_attention_logits, axis=0)
        attention_weights = Label('attention_weights')(attention_weights)
        # attention_weights = [#, d] [*=e]
        attended_encoder_hidden_state = C.reduce_sum(
            attention_weights * C.sequence.broadcast_as(unpacked_encoder_hidden_state, attention_weights), axis=0)
        # attended_encoder_hidden_state = [#, d] [1, h]
        output = attn_final_stab(C.reshape(attended_encoder_hidden_state, (), 0, 1))
        # output = [#, d], [h]
        return output

    return _inject_name(new_attention, name)


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


def create_tsv_reader(func, tsv_file, polymath, seqs, num_workers, is_test=False, misc=None):
    with open(tsv_file, 'r', encoding='utf-8') as f:
        eof = False
        batch_count = 0
        while not (eof and (batch_count % num_workers) == 0):
            batch_count += 1
            batch = {'cwids': [], 'qwids': [], 'baidx': [],
                     'eaidx': [], 'ccids': [], 'qcids': []}

            while not eof and len(batch['cwids']) < seqs:
                line = f.readline()
                if not line:
                    eof = True
                    break

                if misc is not None:
                    import re
                    misc['uid'].append(re.match('^([^\t]*)', line).groups()[0])

                ctokens, qtokens, atokens, cwids, qwids, baidx, eaidx, ccids, qcids = tsv2ctf.tsv_iter(
                    line, polymath.vocab, polymath.chars, is_test, misc)

                batch['cwids'].append(cwids)
                batch['qwids'].append(qwids)
                batch['baidx'].append(baidx)
                batch['eaidx'].append(eaidx)
                batch['ccids'].append(ccids)
                batch['qcids'].append(qcids)

            if len(batch['cwids']) > 0:
                context_g_words = C.Value.one_hot(
                    [[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in cwids] for cwids in batch['cwids']],
                    polymath.wg_dim)
                context_ng_words = C.Value.one_hot(
                    [[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in cwids] for cwids in
                     batch['cwids']], polymath.wn_dim)
                query_g_words = C.Value.one_hot(
                    [[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in qwids] for qwids in batch['qwids']],
                    polymath.wg_dim)
                query_ng_words = C.Value.one_hot(
                    [[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in qwids] for qwids in
                     batch['qwids']], polymath.wn_dim)
                context_chars = [np.asarray([[[c for c in cc + [0] * max(0, polymath.word_size - len(cc))]]
                                             for cc in ccid], dtype=np.float32) for ccid in batch['ccids']]
                query_chars = [np.asarray([[[c for c in qc + [0] * max(0, polymath.word_size - len(qc))]]
                                           for qc in qcid], dtype=np.float32) for qcid in batch['qcids']]
                answer_begin = [np.asarray(ab, dtype=np.float32)
                                for ab in batch['baidx']]
                answer_end = [np.asarray(ae, dtype=np.float32)
                              for ae in batch['eaidx']]

                yield {argument_by_name(func, 'cgw'): context_g_words,
                       argument_by_name(func, 'qgw'): query_g_words,
                       argument_by_name(func, 'cnw'): context_ng_words,
                       argument_by_name(func, 'qnw'): query_ng_words,
                       argument_by_name(func, 'cc'): context_chars,
                       argument_by_name(func, 'qc'): query_chars,
                       argument_by_name(func, 'ab'): answer_begin,
                       argument_by_name(func, 'ae'): answer_end}
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
