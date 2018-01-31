import cntk as C
from cntk.layers import *

from script.utils import BiRecurrence


class AnswerSynthesisModel(object):
    # Create the containers for input feature (x) and the label (y)
    # axis_qry = C.Axis.new_unique_dynamic_axis('axis_qry')
    # qry = C.sequence.input_variable(QRY_SIZE, sequence_axis=axis_qry)
    #
    # axis_ans = C.Axis.new_unique_dynamic_axis('axis_ans')
    # ans = C.sequence.input_variable(ANS_SIZE, sequence_axis=axis_ans)


    def __init__(self, reader, word_emb_dim, feature_emb_dim, hidden_dim,attention_dim):
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.feature_emb_dim = feature_emb_dim
        self.word_emb_dim = word_emb_dim
        self.reader = reader

    def question_encoder_factory(self):
        with default_options(initial_state=0.1):
            model = Sequential([
                Embedding(self.word_emb_dim, name='embed'),
                # ht = BiGRU(ht−1, etq)
                BiRecurrence(GRU(shape=self.hidden_dim / 2), GRU(shape=self.hidden_dim / 2)),
            ])
        return model

    def passage_encoder_factory(self):
        with default_options(initial_state=0.1):
            model = Sequential([
                Embedding(self.word_emb_dim + self.feature_emb_dim * 2, name='embed'),
                # ht = BiGRU(ht−1, [etp, fts, fte])
                BiRecurrence(GRU(shape=self.hidden_dim / 2), GRU(shape=self.hidden_dim / 2)),
            ])
        return model

    def decoder_initialization_factory(self):
        return splice>>Dense(self.hidden_dim, activation=C.tanh)

    def decoder_factory(self):
        question_encoder = self.question_encoder_factory()
        passage_encoder = self.passage_encoder_factory()
        h_b1_q = question_encoder[self.hidden_dim / 2 - 1:]
        h_b1_p = passage_encoder[self.hidden_dim / 2 - 1:]
        decoder_initialization = self.decoder_initialization_factory()
        d_0=decoder_initialization(h_b1_p,h_b1_q)
        h=splice(question_encoder,passage_encoder)
        attention=AttentionModel(self.attention_dim)
        v_t=C.parameter()
        attention()

        @C.Function
        def attention(hidden_state):
            Dense()








