import cntk as C
from cntk.layers import *


class AnswerSynthesisModel(object):
    # Create the containers for input feature (x) and the label (y)
    # axis_qry = C.Axis.new_unique_dynamic_axis('axis_qry')
    # qry = C.sequence.input_variable(QRY_SIZE, sequence_axis=axis_qry)
    #
    # axis_ans = C.Axis.new_unique_dynamic_axis('axis_ans')
    # ans = C.sequence.input_variable(ANS_SIZE, sequence_axis=axis_ans)


    def __init__(self, reader, word_emb_dim, feature_emb_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.feature_emb_dim = feature_emb_dim
        self.word_emb_dim = word_emb_dim
        self.reader = reader

    def question_encoder_factory(self):
        with default_options(initial_state=0.1):
            model = Sequential([
                Embedding(self.word_emb_dim, name='embed'),
                # ht = BiGRU(ht−1, etq)
                Recurrence(GRU(shape=150), go_backwards=False),
                Recurrence(GRU(shape=150), go_backwards=True)

            ])
        return model

    def passage_encoder_factory(self):
        with default_options(initial_state=0.1):
            model = Sequential([
                Embedding(self.word_emb_dim + self.feature_emb_dim * 2, name='embed'),
                # ht = BiGRU(ht−1, [etp, fts, fte])
                Recurrence(GRU(shape=150), go_backwards=False),
                Recurrence(GRU(shape=150), go_backwards=True),
                splice()
            ])
        return model

    def decoder_initialization(self):
        question_encoder = self.question_encoder_factory()
        passage_encoder = self.passage_encoder_factory()



    def decoder_factory(self):
        pass
