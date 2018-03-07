import importlib
import os
import pickle

from cntk.layers import *

from utils import *


class EvidenceExtractionModel(object):
    def __init__(self, config_file):
        config = importlib.import_module(config_file)
        data_config = config.data_config
        model_config = config.evidence_extraction_model

        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_file = os.path.join(self.abs_path, 'data', data_config['pickle_file'])
        with open(pickle_file, 'rb') as vf:
            known, vocab, chars, known_npglove_matrix = pickle.load(vf)

        self.vocab=vocab
        self.chars=chars
        self.known_npglove_matrix = known_npglove_matrix
        # self.vocab_dim = len(vocab)
        self.wg_dim = known
        self.wn_dim = len(vocab) - known
        self.char_dim = len(chars)
        self.a_dim = 1
        self.word_size = data_config['word_size']
        self.word_emb_dim = model_config['word_emb_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.attention_dim = model_config['attention_dim']
        self.dropout = model_config['dropout']
        self.use_cuDNN = model_config['use_cuDNN']
        self.use_sparse = model_config['use_sparse']
        self.use_harmax=model_config['use_harmax']
        self.question_seq_axis = C.Axis.new_unique_dynamic_axis('questionAxis')
        self.passage_seq_axis = C.Axis.new_unique_dynamic_axis('passageAxis')

    # def charcnn(self, x):
    #     conv_out = C.layers.Sequential([
    #         C.layers.Embedding(self.char_emb_dim),
    #         C.layers.Dropout(self.dropout),
    #         C.layers.Convolution2D((5, self.char_emb_dim), self.char_convs, activation=C.relu, init=C.glorot_uniform(),
    #                                bias=True, init_bias=0, name='charcnn_conv')])(x)
    #     return C.reshape(C.reduce_max(conv_out, axis=1),
    #                      self.char_convs)  # workaround cudnn failure in GlobalMaxPooling

    def embed(self):
        # load glove
        glove = C.constant(self.known_npglove_matrix, name='Untrainable_Glove')
        nonglove = C.parameter(shape=(self.wn_dim, self.word_emb_dim), init=C.glorot_uniform(),
                               name='Trainable_Glove')

        @C.Function
        def func(wg, wn):
            return C.times(wg, glove) + C.times(wn, nonglove)

        return func

    def encoder_factory(self, name=''):
        with default_options(enable_self_stabilization=True):
            model = Sequential([
                Stabilizer(),
                BiRNN(self.hidden_dim, use_cudnn=self.use_cuDNN),
                Dropout(self.dropout)
            ], name=name)
        return model

    def input_layer(self, question_gw, question_nw, passage_gw, passage_nw):
        qgw_ph = C.placeholder()
        qnw_ph = C.placeholder()
        pgw_ph = C.placeholder()
        pnw_ph = C.placeholder()
        question_encoder = self.encoder_factory('question_encoder')
        passage_encoder = self.encoder_factory('passage_encoder')

        # input_chars = C.placeholder(shape=(1, self.word_size, self.char_dim))
        input_glove_words = C.placeholder(shape=(self.wg_dim,))
        input_nonglove_words = C.placeholder(shape=(self.wn_dim,))
        embedded = self.embed()(input_glove_words, input_nonglove_words)
        q_emb = embedded.clone(C.CloneMethod.share, {input_glove_words: qgw_ph, input_nonglove_words: qnw_ph})
        p_emb = embedded.clone(C.CloneMethod.share, {input_glove_words: pgw_ph, input_nonglove_words: pnw_ph})

        U_Q = question_encoder(q_emb)
        U_P = passage_encoder(p_emb)

        return C.as_block(
            C.combine([U_Q, U_P]),
            [(pgw_ph, passage_gw), (pnw_ph, passage_nw), (qgw_ph, question_gw), (qnw_ph, question_nw), ],
            'input_layer',
            'input_layer'
        )

    def soft_alignment(self, U_Q, U_P):
        U_Q_ph = C.placeholder(shape=self.hidden_dim * 2)
        U_P_ph = C.placeholder(shape=self.hidden_dim * 2)

        # C_Q_gru = GRU(self.hidden_dim, enable_self_stabilization=True)
        C_Q_att_layer = AttentionModel(self.attention_dim, name='C_Q_att_layer')
        rnn = Sequential([
            BiRNN(self.hidden_dim),
            Stabilizer(),
            # Dropout(self.dropout)
        ], name='soft_alignment_rnn')
        gate_layer = Dense(self.hidden_dim * 4, activation=sigmoid, bias=False, name='att_gate')

        @C.Function
        def soft_alignment_layer(U_Q: SequenceOver[self.question_seq_axis], U_P: SequenceOver[self.passage_seq_axis]):
            C_Q = C_Q_att_layer(U_Q, U_P)
            att_mod = C.splice(U_P, C_Q)
            processed_attention = C.element_times(gate_layer(att_mod), att_mod)
            V_P = rnn(processed_attention)

            # r_Q = r_Q_att_layer(U_Q, C.sequence.last(V_P))
            return V_P

        V_P = soft_alignment_layer(U_Q_ph, U_P_ph)

        return C.as_block(
            V_P,
            [(U_Q_ph, U_Q), (U_P_ph, U_P)],
            'soft_alignment',
            'soft_alignment'
        )

    def poniter_init_attention(self, encoder_hidden_states):
        encoder_h_ph = C.placeholder(self.hidden_dim * 2)
        init = glorot_uniform()
        with default_options(bias=False, enable_self_stabilization=True):  # all the projections have no bias
            attn_proj_enc = Stabilizer() >> Dense(self.attention_dim,
                                                  init=init,
                                                  input_rank=1)  # projects input hidden state, keeping span axes intact
            attn_proj_tanh = Stabilizer() >> Dense(1, init=init,
                                                   input_rank=1)  # projects tanh output, keeping span and beam-search axes intact
            attn_final_stab = Stabilizer()
        decoder_hidden_state = C.parameter(self.attention_dim)

        @C.Function
        def attention_layer(encoder_hidden_state):
            projected_encoder_hidden_state = attn_proj_enc(encoder_hidden_state)
            projected_decoder_hidden_state = C.sequence.broadcast_as(decoder_hidden_state,
                                                                     encoder_hidden_state)
            attention_logits = attn_proj_tanh(projected_decoder_hidden_state + projected_encoder_hidden_state)
            attention_weights = C.sequence.softmax(attention_logits)
            # attention_weights = Label('attention_weights')(attention_weights)
            attended_encoder_hidden_state = C.sequence.reduce_sum(
                C.element_times(attention_weights, encoder_hidden_state)
            )
            output = attn_final_stab(attended_encoder_hidden_state)
            return output

        return C.as_block(
            attention_layer(encoder_h_ph),
            [(encoder_h_ph, encoder_hidden_states)],
            'poniter_init_attention',
            'poniter_init_attention'
        )

    def pointer_network(self, V_P, r_Q):
        V_P_ph = C.placeholder(shape=self.hidden_dim * 2)
        r_Q_ph = C.placeholder(shape=self.hidden_dim * 2)

        init = glorot_uniform()
        with default_options(bias=False, enable_self_stabilization=True):  # all the projections have no bias
            attn_proj_enc = Stabilizer() >> Dense(self.attention_dim, init=init,
                                                  input_rank=1)  # projects input hidden state, keeping span axes intact
            attn_proj_dec = Stabilizer() >> Dense(self.attention_dim, init=init,
                                                  input_rank=1)  # projects decoder hidden state, but keeping span and beam-search axes intact
            attn_proj_tanh = Stabilizer() >> Dense(1, init=init, activation=C.tanh,
                                                   input_rank=1)  # projects tanh output, keeping span and beam-search axes intact
            attn_final_stab = Stabilizer()
        C_gru = GRU(self.hidden_dim * 2, enable_self_stabilization=True)

        @C.Function
        def pointer_network_layer(V_P: SequenceOver[self.passage_seq_axis], r_Q):
            encoder_hidden_state = V_P

            def H_A_gru_cell(hidden_prev):
                decoder_hidden_state = hidden_prev

                # attention layer
                # ============================
                projected_encoder_hidden_state = attn_proj_enc(encoder_hidden_state)
                projected_decoder_hidden_state = C.sequence.broadcast_as(attn_proj_dec(decoder_hidden_state),
                                                                         encoder_hidden_state)
                attention_logits = attn_proj_tanh(projected_decoder_hidden_state + projected_encoder_hidden_state)
                attention_weights = C.sequence.softmax(attention_logits)
                # attention_weights = Label('attention_weights')(attention_weights)
                attended_encoder_hidden_state = C.sequence.reduce_sum(
                    C.element_times(attention_weights, encoder_hidden_state)
                )
                output = attn_final_stab(attended_encoder_hidden_state)
                # ============================
                c = output
                hidden = C_gru(hidden_prev, c)
                return (attention_weights, hidden)

            # h1:[300]
            p1, h1 = H_A_gru_cell(r_Q)
            p2, h2 = H_A_gru_cell(h1)

            return C.combine([p1, p2])

        return C.as_block(
            pointer_network_layer(V_P_ph, r_Q_ph),
            [(V_P_ph, V_P), (r_Q_ph, r_Q)],
            'pointer_network',
            'pointer_network'
        )

    def criterion(self, begin, end, begin_label, end_label):
        b_ph = C.placeholder()
        e_ph = C.placeholder()
        bl_ph = C.placeholder()
        el_ph = C.placeholder()
        if self.use_harmax:

            b=seq_hardmax(b_ph)
            e=C.hardmax(e_ph)
        else:
            b=b_ph
            e=e_ph
        loss_raw = C.plus(C.binary_cross_entropy(b, bl_ph), C.binary_cross_entropy(e, el_ph))
        # # print(loss_raw)
        loss = C.sequence.reduce_sum(loss_raw)

        # start_loss = seq_loss(b_ph, bl_ph)
        # end_loss = seq_loss(e_ph,el_ph)
        # paper_loss = start_loss + end_loss
        # loss,x,y,logZ=all_spans_loss(b_ph,bl_ph,e_ph,el_ph)

        return C.as_block(
            loss,
            [(b_ph, begin), (e_ph, end), (bl_ph, begin_label), (el_ph, end_label)],
            'criterion',
            'criterion'
        )

    def model(self):
        question_gw = C.sequence.input_variable(self.wg_dim, sequence_axis=self.question_seq_axis,
                                                is_sparse=self.use_sparse, name='question_gw')
        question_nw = C.sequence.input_variable(self.wn_dim, sequence_axis=self.question_seq_axis,
                                                is_sparse=self.use_sparse, name='question_nw')
        passage_gw = C.sequence.input_variable(self.wg_dim, sequence_axis=self.passage_seq_axis,
                                               is_sparse=self.use_sparse, name='passage_gw')
        passage_nw = C.sequence.input_variable(self.wn_dim, sequence_axis=self.passage_seq_axis,
                                               is_sparse=self.use_sparse, name='passage_nw')
        begin = C.sequence.input_variable(self.a_dim, sequence_axis=self.passage_seq_axis, name='begin')
        end = C.sequence.input_variable(self.a_dim, sequence_axis=self.passage_seq_axis, name='end')

        # soft_alignment = self.soft_alignment_factory()


        # Output('input_layer', [#, questionAxis], [300])
        # Output('input_layer', [#, passageAxis], [300])
        U_Q, U_P = self.input_layer(question_gw, question_nw, passage_gw, passage_nw).outputs

        # ('soft_alignment', [#, passageAxis], [300])
        V_P = self.soft_alignment(U_Q, U_P)
        # print(V_P.output)

        # ('poniter_init_attention', [#], [300])
        r_Q = self.poniter_init_attention(U_Q)

        # print(r_Q.output)
        model = self.pointer_network(V_P, r_Q)
        # Output('pointer_network', [#, passageAxis], [1])
        # Output('pointer_network', [#, passageAxis], [1])
        p1, p2 = model.outputs
        # print(p1, p2)

        loss = self.criterion(p1, p2, begin, end)

        return model, loss


a = EvidenceExtractionModel('config')

model,loss=a.model()
# root=loss
# begin_label = argument_by_name(root, 'begin')
# end_label = argument_by_name(root, 'end')
# print(type(begin_label))
# print(C.combine([model,loss]))
# print(loss.arguments)
# print(a.model()[1])
