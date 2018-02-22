import importlib
import os
import pickle

from cntk.layers import *

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
        self.word_emb_dim = model_config['word_emb_dim']
        self.char_emb_dim = model_config['char_emb_dim']
        self.char_convs = model_config['char_convs']
        self.hidden_dim = model_config['hidden_dim']
        self.attention_dim = model_config['attention_dim']
        self.r = model_config['r']
        self.dropout = model_config['dropout']
        self.question_seq_axis = C.Axis.new_unique_dynamic_axis('questionAxis')
        self.passage_seq_axis = C.Axis.new_unique_dynamic_axis('passageAxis')
        self.answer_seq_axis = C.Axis.new_unique_dynamic_axis('answerAxis')
        self.PassageSequence = SequenceOver[self.passage_seq_axis][Tensor[self.vocab_dim]]
        self.QuestionSequence = SequenceOver[self.question_seq_axis][Tensor[self.vocab_dim]]
        self.AnswerSequence = SequenceOver[self.answer_seq_axis][Tensor[self.vocab_dim]]
        self.emb_layer = self.embed_factory()

    def charcnn_factory(self):
        conv_out = C.layers.Sequential([
            C.layers.Embedding(self.char_emb_dim),
            C.layers.Dropout(self.dropout),
            C.layers.Convolution2D((5, self.char_emb_dim), self.char_convs, activation=C.relu, init=C.glorot_uniform(),
                                   bias=True, init_bias=0, name='charcnn_conv')])

        max = C.reduce_max(conv_out, axis=1)
        return C.reshape(max, self.char_convs)

    def embed_factory(self):
        glove_matrix = C.Constant(self.npglove_matrix)
        charcnn = self.charcnn_factory()


        #todo: char issues!!!!
        @C.Function
        def embedding(input_word):
            word_emb = C.times(input_word, glove_matrix)
            char_emb = charcnn(input_word)
            emb = C.splice(word_emb, char_emb)
            return emb

        return embedding

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
        r_Q_att_layer=AttentionModel(self.attention_dim, name='r_Q_att_layer')
        @C.Function
        def soft_alignment(question, passage):
            U_Q = question_encoder(question)
            U_P = passage_encoder(passage)

            @C.Function
            def V_P_gru_cell(hidden_prev, U_P_t):
                C_Q = C_Q_att_layer(U_Q.output, C.splice(U_P_t, hidden_prev))
                hidden = C_Q_gru(hidden_prev, C_Q)
                return hidden

            V_P = Recurrence(V_P_gru_cell)(U_P)
            r_Q = r_Q_att_layer(U_Q.output,C.sequence.last(V_P))



            return C.combine([V_P,r_Q])



        return soft_alignment

    def pointer_network_factory(self):
        soft_alignment = self.soft_alignment_factory()
        C_att_layer = AttentionModel(self.attention_dim, name='C_att_layer')
        C_gru = GRU(self.hidden_dim, enable_self_stabilization=True)

        @C.Function
        def pointer_network(question, passage):
            V_P,r_Q = soft_alignment(question, passage)

            @C.Function
            def H_A_gru_cell(hidden_prev):
                # p_prev: dummy for output
                c = C_att_layer(V_P.output, hidden_prev)
                hidden = C_gru(hidden_prev, c)
                return hidden

            unfold = UnfoldFrom(H_A_gru_cell)
            H_A=unfold(initial_state=r_Q,dynamic_axes_like=passage)


a = EvidenceExtractionModel('config')
print(a.soft_alignment_factory())
