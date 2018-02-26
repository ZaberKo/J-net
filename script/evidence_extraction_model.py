import importlib
import os
import pickle

from cntk.layers import *
import cntk as C
from utils import BiRecurrence


class EvidenceExtractionModel(object):
    def __init__(self, config_file):
        config = importlib.import_module(config_file)
        data_config = config.data_config
        model_config = config.evidence_extraction_model

        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_file = os.path.join(self.abs_path, 'data', data_config['pickle_file'])
        with open(pickle_file, 'rb') as vf:
            known, vocab, chars, npglove_matrix = pickle.load(vf)

        self.npglove_matrix = npglove_matrix
        self.vocab_dim = len(vocab)
        self.char_dim = len(chars)
        self.word_emb_dim = model_config['word_emb_dim']
        self.char_emb_dim = model_config['char_emb_dim']
        self.char_convs = model_config['char_convs']
        self.hidden_dim = model_config['hidden_dim']
        self.attention_dim = model_config['attention_dim']
        self.dropout = model_config['dropout']
        self.question_seq_axis = C.Axis.new_unique_dynamic_axis('questionAxis')
        self.passage_seq_axis = C.Axis.new_unique_dynamic_axis('passageAxis')
        self.answer_seq_axis = C.Axis.new_unique_dynamic_axis('answerAxis')
        self.PassageSequence = SequenceOver[self.passage_seq_axis][Tensor[self.vocab_dim]]
        self.QuestionSequence = SequenceOver[self.question_seq_axis][Tensor[self.vocab_dim]]
        self.AnswerSequence = SequenceOver[self.answer_seq_axis][Tensor[self.vocab_dim]]
        self.pointer_seq=SequenceOver
        self.emb_layer = Embedding(weights=npglove_matrix)

    # def charcnn_factory(self):
    #     conv_out = C.layers.Sequential([
    #         C.layers.Embedding(self.char_emb_dim),
    #         C.layers.Dropout(self.dropout),
    #         C.layers.Convolution2D((5, self.char_emb_dim), self.char_convs, activation=C.relu, init=C.glorot_uniform(),
    #                                bias=True, init_bias=0, name='charcnn_conv')])
    #
    #     max = C.reduce_max(conv_out, axis=1)
    #     return C.reshape(max, self.char_convs)

    # def embed_factory(self):
    #     glove_matrix = C.Constant(self.npglove_matrix)
    #     charcnn = self.charcnn_factory()
    #
    #
    #     @C.Function
    #     def embedding(input_word,input_char_raw):
    #         word_emb = C.times(input_word, glove_matrix)
    #         input_char=C.one_hot(input_char_raw,num_classes=self.char_dim)
    #         char_emb = C.reshape(charcnn(input_char),self.char_convs)
    #         emb = C.splice(word_emb, char_emb)
    #         return emb
    #
    #     return embedding



    def question_encoder_factory(self):
        with default_options(enable_self_stabilization=True):
            model = Sequential([
                self.emb_layer,
                Stabilizer(),
                # ht = BiGRU(ht−1, etq)
                BiRecurrence(GRU(shape=self.hidden_dim), GRU(shape=self.hidden_dim))
            ], name='question_encoder')
        return model

    def passage_encoder_factory(self):
        with default_options(enable_self_stabilization=True):
            model = Sequential([
                self.emb_layer,
                Stabilizer(),
                # ht = BiGRU(ht−1, etq)
                BiRecurrence(GRU(shape=self.hidden_dim), GRU(shape=self.hidden_dim))
            ], name='question_encoder')
        return model

    def soft_alignment_factory(self):
        question_encoder = self.question_encoder_factory()
        passage_encoder = self.passage_encoder_factory()

        C_Q_gru = GRU(self.hidden_dim, enable_self_stabilization=True)
        C_Q_att_layer = AttentionModel(self.attention_dim, name='C_Q_att_layer')
        r_Q_att_layer = AttentionModel(self.attention_dim, name='r_Q_att_layer')


        def soft_alignment(question:self.QuestionSequence, passage:self.PassageSequence):
            U_Q = question_encoder(question)
            U_P = passage_encoder(passage)


            def V_P_gru_cell(hidden_prev, x):
                C_Q = C_Q_att_layer(U_Q.output, C.splice(x, hidden_prev))
                hidden = C_Q_gru(hidden_prev, C_Q)
                return hidden

            V_P = Recurrence(V_P_gru_cell)(U_P)
            r_Q = r_Q_att_layer(U_Q.output, C.sequence.last(V_P))

            return (V_P, r_Q)

        return soft_alignment

    def pointer_network_factory(self):
        soft_alignment = self.soft_alignment_factory()
        init = glorot_uniform()
        with default_options(bias=False, enable_self_stabilization=True):  # all the projections have no bias
            attn_proj_enc = Stabilizer() >> Dense(self.attention_dim, init=init,
                                                  input_rank=1)  # projects input hidden state, keeping span axes intact
            attn_proj_dec = Stabilizer() >> Dense(self.attention_dim, init=init,
                                                  input_rank=1)  # projects decoder hidden state, but keeping span and beam-search axes intact
            attn_proj_tanh = Stabilizer() >> Dense(1, init=init,
                                                   input_rank=1)  # projects tanh output, keeping span and beam-search axes intact
            attn_final_stab = Stabilizer()
        C_gru = GRU(self.hidden_dim, enable_self_stabilization=True)

        @C.Function
        def pointer_network(question:self.QuestionSequence, passage:self.PassageSequence):

            V_P, r_Q=soft_alignment(question, passage)
            encoder_hidden_state = V_P

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
                attention_weights = Label('attention_weights')(attention_weights)  # attention_weights = [#, d] [*=e]
                attended_encoder_hidden_state = C.reduce_sum(
                    attention_weights * C.sequence.broadcast_as(unpacked_encoder_hidden_state, attention_weights),
                    axis=0)
                output = attn_final_stab(C.reshape(attended_encoder_hidden_state, (), 0, 1))
                # ============================


                c = output
                hidden = C_gru(hidden_prev, c)
                return (attention_weights,hidden)

            # unfold = UnfoldFrom(H_A_gru_cell)
            # pointer_prob = unfold(initial_state=r_Q,dynamic_axes_like=passage)

            p1,h1=H_A_gru_cell(r_Q)
            p2,h2=H_A_gru_cell(h1)
            return C.combine([p1,p2])

        return pointer_network


    def criterion_factory(self):

        def criterion(begin,end,begin_label,end_label):
            def cross_entropy_loss(x,label):
                result=C.plus(
                    C.element_times(label,C.log(x)),
                    C.element_times(1-label,C.log(1-x))
                )
                return result
            loss=C.negate(C.plus(cross_entropy_loss(begin,begin_label),cross_entropy_loss(end,end_label)))
            return loss

        return criterion()


    def model(self):
        question_seq = C.sequence.input_variable(self.vocab_dim, sequence_axis=self.question_seq_axis, name='question')
        passage_seq = C.sequence.input_variable(self.vocab_dim, sequence_axis=self.passage_seq_axis, name='passage')
        begin=C.sequence.input_variable(1,sequence_axis=self.passage_seq_axis,name='begin')
        end=C.sequence.input_variable(1,sequence_axis=self.passage_seq_axis,name='end')

        pointer_network=self.pointer_network_factory()
        pointer_network(question_seq,passage_seq)


a = EvidenceExtractionModel('config')
print(a.model())
