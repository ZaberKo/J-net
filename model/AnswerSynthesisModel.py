from cntk.layers import *

from script.utils import BiRecurrence


class AnswerSynthesisModel(object):
    # Create the containers for input feature (x) and the label (y)
    # axis_qry = C.Axis.new_unique_dynamic_axis('axis_qry')
    # qry = C.sequence.input_variable(QRY_SIZE, sequence_axis=axis_qry)
    #
    # axis_ans = C.Axis.new_unique_dynamic_axis('axis_ans')
    # ans = C.sequence.input_variable(ANS_SIZE, sequence_axis=axis_ans)


    def __init__(self, reader, word_emb_dim, feature_emb_dim, hidden_dim, attention_dim):
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.feature_emb_dim = feature_emb_dim
        self.word_emb_dim = word_emb_dim
        self.reader = reader

    def question_encoder_factory(self):
        with default_options(initial_state=0.1):
            model = Sequential([
                Embedding(self.word_emb_dim, name='embed'),
                Stabilizer(),
                # ht = BiGRU(ht−1, etq)
                BiRecurrence(GRU(shape=self.hidden_dim / 2), GRU(shape=self.hidden_dim / 2)),
            ])
        return model

    def passage_encoder_factory(self):
        with default_options(initial_state=0.1):
            model = Sequential([
                Embedding(self.word_emb_dim + self.feature_emb_dim * 2, name='embed'),
                Stabilizer(),
                # ht = BiGRU(ht−1, [etp, fts, fte])
                BiRecurrence(GRU(shape=self.hidden_dim / 2), GRU(shape=self.hidden_dim / 2)),
            ])
        return model

    def decoder_initialization_factory(self):
        return splice >> Dense(self.hidden_dim, activation=C.tanh, bias=True)

    # def decoder_factory(self):
    #     @Function
    #     def decode(history, input):
    #         encoded_input = encode(input)
    #         r = history
    #         r = embed(r)
    #         r = stab_in(r)
    #
    #         rec_block  # LSTM(hidden_dim)  # :: (dh, dc, x) -> (h, c)
    #
    #
    #             @Function
    #             def lstm_with_attention(dh, dc, x):
    #                 h_att = attention_model(encoded_input.outputs[0], dh)
    #                 x = splice(x, h_att)  # TODO: should this be added instead? (cf. BS example)
    #                 return rec_block(dh, dc, x)
    #
    #             r = Recurrence(lstm_with_attention)(r)
    #
    #
    #     r = stab_out(r)
    #     r = proj_out(r)
    #     r = Label('out_proj_out')(r)
    #     return r


def train(self):
    pass
