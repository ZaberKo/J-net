import importlib
import os
import pickle

from cntk.layers import *

from utils import BiGRU


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
                Dropout(self.dropout),
                # ht = BiGRU(ht−1, etq)
                BiGRU(self.hidden_dim, use_cudnn=True)
            ], name=name)
        return model

    def input_layer(self, question_gw, question_nw, question_c, passage_gw, passage_nw, passage_c):
        qgw_ph = C.placeholder()
        qnw_ph = C.placeholder()
        qc_ph = C.placeholder()
        pgw_ph = C.placeholder()
        pnw_ph = C.placeholder()
        pc_ph = C.placeholder()
        question_encoder = self.encoder_factory('question_encoder')
        passage_encoder = self.encoder_factory('passage_encoder')

        input_chars = C.placeholder(shape=(1, self.word_size, self.char_dim))
        input_glove_words = C.placeholder(shape=(self.wg_dim,))
        input_nonglove_words = C.placeholder(shape=(self.wn_dim,))
        embedded = C.splice(
            self.charcnn(input_chars),
            self.embed()(input_glove_words, input_nonglove_words), name='splice_embed')
        qce = C.one_hot(qc_ph, num_classes=self.char_dim, sparse_output=self.use_sparse)
        pce = C.one_hot(pc_ph, num_classes=self.char_dim, sparse_output=self.use_sparse)
        q_emb = embedded.clone(C.CloneMethod.share, {input_chars: qce, input_glove_words: qgw_ph,
                                                     input_nonglove_words: qnw_ph})
        p_emb = embedded.clone(C.CloneMethod.share, {input_chars: pce, input_glove_words: pgw_ph,
                                                     input_nonglove_words: pnw_ph})

        U_Q = question_encoder(q_emb)
        U_P = passage_encoder(p_emb)

        return C.as_block(
            C.combine([U_Q, U_P]),
            [(pgw_ph, passage_gw), (pnw_ph, passage_nw), (pc_ph, passage_c), (qgw_ph, question_gw),
             (qnw_ph, question_nw), (qc_ph, question_c)],
            'input_layer',
            'input_layer'
        )

    def soft_alignment_factory(self):
        # input_layer = self.input_layer_factory()
        C_Q_gru = GRU(self.hidden_dim, enable_self_stabilization=True)
        C_Q_att_layer = AttentionModel(self.attention_dim, name='C_Q_att_layer')
        r_Q_att_layer = AttentionModel(self.attention_dim, name='r_Q_att_layer')

        @C.Function
        def soft_alignment(U_Q: SequenceOver[self.question_seq_axis], U_P: SequenceOver[self.passage_seq_axis]):
            # U_Q, U_P = input_layer(question, passage).outputs

            @C.Function
            def gru_with_attention(hidden_prev, x):
                # todo: structure changed, rewrite new att_layer
                C_Q = C_Q_att_layer(U_Q, hidden_prev)
                hidden = C_Q_gru(hidden_prev, C.splice(C_Q, x))
                return hidden

            rnn = Recurrence(gru_with_attention, name='V_P_GRU')

            V_P = rnn(U_P)
            r_Q = r_Q_att_layer(U_Q, C.sequence.last(V_P))
            return C.combine([V_P, r_Q])

        return soft_alignment

    def pointer_network_factory(self):
        soft_alignment = self.soft_alignment_factory()
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
        def pointer_network(V_P, r_Q):
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

        return pointer_network

    def criterion_factory(self):
        @C.Function
        def criterion(begin, end,
                      begin_label, end_label):
            loss = C.plus(C.binary_cross_entropy(begin, begin_label), C.binary_cross_entropy(end, end_label))
            return loss

        return criterion

    def model(self):
        question_gw = C.sequence.input_variable(self.wg_dim, sequence_axis=self.question_seq_axis, name='question_gw')
        question_nw = C.sequence.input_variable(self.wn_dim, sequence_axis=self.question_seq_axis, name='question_nw')
        question_c = C.sequence.input_variable((1, self.word_size), sequence_axis=self.question_seq_axis,
                                               name='question_c')
        passage_gw = C.sequence.input_variable(self.wg_dim, sequence_axis=self.passage_seq_axis, name='passage_gw')
        passage_nw = C.sequence.input_variable(self.wn_dim, sequence_axis=self.passage_seq_axis, name='passage_nw')
        passage_c = C.sequence.input_variable((1, self.word_size), sequence_axis=self.passage_seq_axis,
                                              name='passage_c')
        begin = C.sequence.input_variable(1, sequence_axis=self.passage_seq_axis, name='begin')
        end = C.sequence.input_variable(1, sequence_axis=self.passage_seq_axis, name='end')

        # input_layer = self.input_layer_factory()
        # processed_q, processed_p = input_layer(question_seq, passage_seq).outputs

        # print(processed_q.dynamic_axes)
        # soft= self.soft_alignment_factory()
        # V_P, r_Q=soft(question_seq,passage_seq).outputs

        U_Q, U_P = self.input_layer(question_gw, question_nw, question_c, passage_gw, passage_nw, passage_c).outputs
        soft = self.soft_alignment_factory()
        # Output('V_P_GRU', [#, passageAxis], [150]) Output('r_Q_att_layer', [#], [300])
        V_P, r_Q = soft(U_Q, U_P).outputs
        # criterion = self.criterion_factory()
        # begin_label = C.sequence.unpack(begin, 0, no_mask_output=True)
        # end_label = C.sequence.unpack(end, 0, no_mask_output=True)
        #
        # loss = criterion(p1, p2, begin_label, end_label)
        #
        # return pointer_net(question_seq, passage_seq), loss


a = EvidenceExtractionModel('config')
a.model()
# print(a.model()[1].arguments)
