from cntk.layers import *

from script.utils import BiRecurrence

word_emb_dim = 300
feature_emb_dim = 50
hidden_dim = 150
attention_dim = 150
vocab_dim = 10000  # issue

question_seq_axis = C.Axis('questionAxis')
passage_seq_axis = C.Axis('passageAxis')
answer_seq_axis = C.Axis('answerAxis')
question_seq  = C.sequence.input_variable((vocab_dim), sequence_axis=question_seq_axis, name='raw_input')
passage_seq = C.sequence.input_variable((vocab_dim), sequence_axis=passage_seq_axis, name='raw_input')
answer_seq = C.sequence.input_variable((vocab_dim), sequence_axis=answer_seq_axis, name='raw_input')

# question_seq = raw_question
# passage_seq = raw_question
# answer_seq = C.sequence.slice(raw_answer, 1)  # <s> A B C </s> --> A B C </s>
# answer_seq_start = C.sequence.first(raw_answer)  # <s>
# is_first_word = C.sequence.is_first(answer_seq)  # 1 0 0 0 ...
# answer_seq_start_scattered = C.sequence.scatter(answer_seq_start,
#                                                 is_first_word)  # <s> 0 0 0 ... (up to the length of label_sequence)
#

def question_encoder_factory():
    with default_options(initial_state=0.1):
        model = Sequential([
            Embedding(word_emb_dim, name='embed'),
            Stabilizer(),
            # ht = BiGRU(ht−1, etq)
            BiRecurrence(GRU(shape=hidden_dim // 2), GRU(shape=hidden_dim // 2)),
        ], name='question_encoder')
    return model


def passage_encoder_factory():
    with default_options(initial_state=0.1):
        model = Sequential([
            Embedding(word_emb_dim, name='embed'),
            Stabilizer(),
            # ht = BiGRU(ht−1, [etp, fts, fte])
            BiRecurrence(GRU(shape=hidden_dim // 2), GRU(shape=hidden_dim // 2))
        ], name='passage_encoder')
    return model


def decoder_initialization_factory():
    return splice >> Dense(hidden_dim, activation=C.tanh, bias=True)


def decoder_factory(question_encoder, passage_encoder, decoder_initialization):
    gru = GRU(hidden_dim)
    q_attention = AttentionModel(attention_dim)
    p_attention = AttentionModel(attention_dim)

    h_q = question_encoder(question)
    h_p = passage_encoder(passage)
    d_0 = decoder_initialization([h_p[0][hidden_dim // 2:], h_q[0][hidden_dim // 2:]])
    @C.Function
    def decoder(passage, question, answer):




        q_att_prev=np.zeros(attention_dim)
        p_att_prev = np.zeros(attention_dim)


        # (h_t-1,x_t-1)->h_t
        @C.Function
        def GRU__with_attention(hidden, x):
            x = splice(x, p_att_prev, q_att_prev)
            p_att = p_attention(h_p, hidden)
            q_att = q_attention(h_q, hidden)
            hidd=gru(hidden,x)

            # emb_layer = Sequential([
            #     Stabilizer(),
            #     Embedding(word_emb_dim)
            # ])
            #
            # # output_layer = Sequential([
            # #     Stabilizer(),
            # #     C.plus(Dense(vocab_dim)(embs_layer),
            # #            Dense(vocab_dim)(q_atts_ph),
            # #            Dense(vocab_dim)(p_atts_ph),
            # #            Dense(vocab_dim)(hidden_state_ph)
            # #            ),
            # #     C.softmax
            # # ])
            #
            # emb_dense = Dense(vocab_dim, activation=identity, bias=0)
            # q_att_dense = Dense(vocab_dim, activation=identity, bias=0)
            # p_att_dense = Dense(vocab_dim, activation=identity, bias=0)
            # hid_dense = Dense(vocab_dim, activation=identity, bias=0)
            # C.plus(emb_dense(x), q_att_dense(q_att), p_att_dense(p_att), hidd)
        # rnn_layer = Sequential([
        #     Stabilizer(),
        #     Recurrence(GRU__with_attention, initial_state=d_0)
        # ])
        #
        # model = Sequential([
        #     Stabilizer(),
        #     C.plus(Dense(vocab_dim)(embs_layer),
        #            Dense(vocab_dim)(q_atts_ph),
        #            Dense(vocab_dim)(p_atts_ph),
        #            Dense(vocab_dim)(hidden_state_ph)
        #            ),
        #     C.softmax
        # ])



    # q_atts = C.placeholder(shape=(attention_dim), dynamic_axes=question_seq_axis, name='q_attentions')
    # p_atts = C.placeholder(shape=(attention_dim), dynamic_axes=passage_seq_axis, name='p_attentions')






    return decoder


def output_factory(decoder):
    embs, q_atts, p_atts, hidden_state = decoder(answer_seq)  # should have the same length
    embs_ph = C.placeholder(shape=(word_emb_dim), dynamic_axes=answer_seq_axis)
    q_atts_ph = C.placeholder(shape=(attention_dim), dynamic_axes=answer_seq_axis)
    p_atts_ph = C.placeholder(shape=(attention_dim), dynamic_axes=answer_seq_axis)
    hidden_state_ph = C.placeholder(shape=hidden_dim, dynamic_axes=answer_seq_axis)

    model = Sequential([
        Stabilizer(),
        C.plus(Dense(vocab_dim)(embs_ph),
               Dense(vocab_dim)(q_atts_ph),
               Dense(vocab_dim)(p_atts_ph),
               Dense(vocab_dim)(hidden_state_ph)
               ),
        C.softmax
    ])


def create_model():
    question_encoder = question_encoder_factory()
    passage_encoder = passage_encoder_factory()
    decoder_initialization = decoder_initialization_factory()
    decoder = decoder_factory(passage_encoder, question_encoder, decoder_initialization)
    probability=decoder(passage_seq,question_seq,answer_seq)
    return probability
