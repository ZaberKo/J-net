import importlib
import os
import pickle

from cntk.layers import *

from utils import BiGRU, MyAttentionModel


class EvidenceExtractionModel(object):
    def __init__(self, config_file):
        config = importlib.import_module(config_file)
        data_config = config.data_config
        model_config = config.evidence_extraction_model

        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_file = os.path.join(self.abs_path, 'data', data_config['pickle_file'])
        with open(pickle_file, 'rb') as vf:
            known, vocab, chars, known_npglove_matrix = pickle.load(vf)

        self.known_npglove_matrix = known_npglove_matrix
        # self.vocab_dim = len(vocab)
        self.wg_dim = known
        self.wn_dim = len(vocab) - known
        self.char_dim = len(chars)
        self.word_size = data_config['word_size']
        self.word_emb_dim = model_config['word_emb_dim']
        self.char_emb_dim = model_config['char_emb_dim']
        self.char_convs = model_config['char_convs']
        self.hidden_dim = model_config['hidden_dim']
        self.attention_dim = model_config['attention_dim']
        self.dropout = model_config['dropout']
        self.use_cuDNN = model_config['use_cuDNN']
        self.use_sparse = model_config['use_sparse']
        self.a_dim = 1
        self.question_seq_axis = C.Axis.new_unique_dynamic_axis('questionAxis')
        self.passage_seq_axis = C.Axis.new_unique_dynamic_axis('passageAxis')

    def charcnn(self, x):
        conv_out = C.layers.Sequential([
            C.layers.Embedding(self.char_emb_dim),
            C.layers.Dropout(self.dropout),
            C.layers.Convolution2D((5, self.char_emb_dim), self.char_convs, activation=C.relu, init=C.glorot_uniform(),
                                   bias=True, init_bias=0, name='charcnn_conv')])(x)
        return C.reshape(C.reduce_max(conv_out, axis=1),
                         self.char_convs)  # workaround cudnn failure in GlobalMaxPooling

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
                BiGRU(self.hidden_dim, use_cudnn=self.use_cuDNN),
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
        r_Q_att_layer = MyAttentionModel(self.attention_dim, self.hidden_dim, name='r_Q_att_layer')
        rnn = Sequential([
            BiGRU(self.hidden_dim),
            Stabilizer(),
            # Dropout(self.dropout)
        ], name='soft_alignment_rnn')

        @C.Function
        def soft_alignment_layer(U_Q: SequenceOver[self.question_seq_axis], U_P: SequenceOver[self.passage_seq_axis]):
            # U_Q, U_P = input_layer(question, passage).outputs

            # @C.Function
            # def gru_with_attention(hidden_prev, x):
            #     # todo: structure changed, rewrite new att_layer
            #     C_Q = C_Q_att_layer(U_Q, C.splice(x,hidden_prev))
            #     hidden = C_Q_gru(hidden_prev, C.splice(C_Q, x))
            #     return hidden
            C_Q = C_Q_att_layer(U_Q, U_P)
            V_P = rnn(C_Q)
            r_Q = r_Q_att_layer(U_Q)
            # r_Q = r_Q_att_layer(U_Q, C.sequence.last(V_P))
            return C.combine([V_P, r_Q])

        return C.as_block(
            soft_alignment_layer(U_Q_ph, U_P_ph),
            [(U_Q_ph, U_Q), (U_P_ph, U_P)],
            'soft_alignment',
            'soft_alignment'
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
        attn_proj_tanh = Stabilizer() >> Dense(1, init=init, bias=True, activation=C.tanh,
                                               input_rank=1)  # projects tanh output, keeping span and beam-search axes intact
        attn_final_stab = Stabilizer(enable_self_stabilization=True)
        C_gru = GRU(self.hidden_dim * 2, enable_self_stabilization=True)

        @C.Function
        def pointer_network_layer(V_P: SequenceOver[self.passage_seq_axis], r_Q):
            encoder_hidden_state = V_P

            # r_Q: 'r_Q_att_layer', [#], [300]
            @C.Function
            def H_A_gru_cell(hidden_prev):
                decoder_hidden_state = hidden_prev

                # copy from cntk source code
                # ============================
                unpacked_encoder_hidden_state, valid_mask = C.sequence.unpack(encoder_hidden_state,
                                                                              padding_value=0).outputs

                projected_encoder_hidden_state = C.sequence.broadcast_as(attn_proj_enc(unpacked_encoder_hidden_state),
                                                                         decoder_hidden_state)
                broadcast_valid_mask = C.sequence.broadcast_as(C.reshape(valid_mask, (1,), 1), decoder_hidden_state)
                projected_decoder_hidden_state = attn_proj_dec(decoder_hidden_state)
                tanh_output = C.tanh(projected_decoder_hidden_state + projected_encoder_hidden_state)
                attention_logits = attn_proj_tanh(tanh_output)
                minus_inf = C.constant(-1e+30)
                masked_attention_logits = C.element_select(broadcast_valid_mask, attention_logits, minus_inf)
                attention_weights = C.softmax(masked_attention_logits, axis=0)
                attention_weights = Label('attention_weights')(attention_weights)
                attended_encoder_hidden_state = C.reduce_sum(
                    attention_weights * C.sequence.broadcast_as(unpacked_encoder_hidden_state, attention_weights),
                    axis=0)
                output = attn_final_stab(C.reshape(attended_encoder_hidden_state, (), 0, 1))
                # ============================
                c = output
                hidden = C_gru(hidden_prev, c)
                return C.combine([attention_weights, hidden])

            p1, h1 = H_A_gru_cell(r_Q).outputs
            p2, h2 = H_A_gru_cell(h1).outputs

            return C.combine([p1, p2])

        return C.as_block(
            pointer_network_layer(V_P_ph, r_Q_ph),
            [(V_P_ph, V_P), (r_Q_ph, r_Q)],
            'pointer_network',
            'pointer_network'
        )

    def criterion(self, begin, end, begin_label, end_label):
        b_ph=C.placeholder()
        e_ph=C.placeholder()
        bl_ph=C.placeholder()
        el_ph=C.placeholder()
        # todo: reduce sum
        loss_raw = C.plus(C.binary_cross_entropy(b_ph, bl_ph), C.binary_cross_entropy(e_ph, el_ph))
        # print(loss_raw)
        loss = C.reduce_sum(loss_raw, axis=0)
        loss = C.reshape(loss, (), 0, 1)

        return C.as_block(
            loss,
            [(b_ph,begin),(e_ph,end),(bl_ph,begin_label),(el_ph,end_label)],
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


        # Output('input_layer', [#, questionAxis], [300]) Output('input_layer', [#, passageAxis], [300])
        U_Q, U_P = self.input_layer(question_gw, question_nw, passage_gw, passage_nw).outputs

        # Output('soft_alignment', [#, passageAxis], [300]) Output('soft_alignment', [#], [300])
        V_P, r_Q = self.soft_alignment(U_Q, U_P).outputs
        # print(V_P, r_Q)

        # Output('attention_weights', [#], [* x 1])
        model=self.pointer_network(V_P, r_Q)
        p1, p2 = model.outputs
        # print(p1,p2)

        begin_label = C.sequence.unpack(begin, 0, no_mask_output=True)
        end_label = C.sequence.unpack(end, 0, no_mask_output=True)
        # print(begin_label)
        #
        loss = self.criterion(p1, p2, begin_label, end_label)
        # print(loss)
        #
        return model, loss


# a = EvidenceExtractionModel('config')
#
# a.model()
# print(a.model()[1])
