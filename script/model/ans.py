import pickle

from cntk.layers import *

from script.utils import BiRecurrence

word_emb_dim = 300
feature_emb_dim = 50
hidden_dim = 150
attention_dim = 150
vocab_dim = 10000  # todo: issue

pickle_file = 'vocabs.pkl'
question_seq_axis = C.Axis('questionAxis')
passage_seq_axis = C.Axis('passageAxis')
answer_seq_axis = C.Axis('answerAxis')
question_seq = C.sequence.input_variable(vocab_dim, sequence_axis=question_seq_axis, name='raw_input')
passage_seq = C.sequence.input_variable(vocab_dim, sequence_axis=passage_seq_axis, name='raw_input')
answer_seq = C.sequence.input_variable(vocab_dim, sequence_axis=answer_seq_axis, name='raw_input')
test = C.sequence.input_variable(attention_dim, sequence_axis=answer_seq_axis, name='raw_input')
bos='<BOS>'
eos='<EOS>'

with open(pickle_file, 'rb') as vf:
    known, vocab, chars = pickle.load(vf)

start_word=C.constant(np.zeros(vocab_dim,dtype=np.float32))
end_word = C.constant(np.zeros(vocab_dim,dtype=np.float32))
end_word_idx=vocab[eos]


def question_encoder_factory():
    with default_options(initial_state=0.1):
        model = Sequential([
            Embedding(word_emb_dim, name='embed'),
            Stabilizer(),
            # ht = BiGRU(ht−1, etq)
            BiRecurrence(GRU(shape=hidden_dim), GRU(shape=hidden_dim)),
        ], name='question_encoder')
    return model


def passage_encoder_factory():
    with default_options(initial_state=0.1):
        model = Sequential([
            Embedding(word_emb_dim, name='embed'),
            Stabilizer(),
            # ht = BiGRU(ht−1, [etp, fts, fte])
            BiRecurrence(GRU(shape=hidden_dim), GRU(shape=hidden_dim))
        ], name='passage_encoder')
    return model





# def output_layer(emb_word, att_p, att_q, hidden):
#     emb_word_ph = C.placeholder()
#     att_p_ph = C.placeholder()
#     att_q_ph = C.placeholder()
#     hidden_ph = C.placeholder()
#     readout = C.plus(
#         Dense(vocab_dim)(emb_word_ph),
#         Dense(vocab_dim)(att_q_ph),
#         Dense(vocab_dim)(att_p_ph),
#         Dense(vocab_dim)(hidden_ph)
#     )
#     word = C.argmax(C.softmax(readout))  # todo: find the map
#     return C.as_block(
#         word,
#         [(emb_word_ph, emb_word), (att_p_ph, att_p), (att_q_ph, att_q), (hidden_ph, hidden)],
#         'output_layer',
#         'output_layer'
#     )


def model_factory():
    question_encoder = question_encoder_factory()
    passage_encoder = passage_encoder_factory()

    q_attention_layer = AttentionModel(attention_dim, name='query_attention')
    p_attention_layer = AttentionModel(attention_dim, name='passage_attention')
    emb_layer = Embedding(word_emb_dim)
    decoder_gru = GRU(hidden_dim)
    decoder_init_dense = Dense(hidden_dim, activation=C.tanh, bias=True)
    # for readout_layer
    emb_dense = Dense(vocab_dim)
    att_p_dense = Dense(vocab_dim)
    att_q_dense = Dense(vocab_dim)
    hidden_dense = Dense(vocab_dim)

    @C.Function
    def decoder(question, passage,word_prev):
        # question encoder hidden state
        h_q = question_encoder(question)
        # passage encoder hidden state
        h_p = passage_encoder(passage)

        h_q1 = C.sequence.last(h_q)
        h_p1 = C.sequence.last(h_p)

        emb_prev = emb_layer(word_prev)

        @C.Function
        def gru_with_attention(hidden_prev, att_p_prev, att_q_prev, emb_prev):
            x = C.splice(emb_prev, att_p_prev, att_q_prev)
            hidden = decoder_gru(hidden_prev, x)
            att_p = p_attention_layer(h_p.output, hidden)
            att_q = q_attention_layer(h_q.output, hidden)
            return (hidden, att_p, att_q)

        # decoder_initialization
        d_0 =(splice(C.slice(h_p1, 0, hidden_dim, 0), C.slice(h_q1, 0, hidden_dim, 0))>> decoder_init_dense).output
        # todo: is random better?
        att_p_0 = np.zeros(attention_dim)
        att_q_0 = np.zeros(attention_dim)
        init_state = (d_0, att_p_0, att_q_0)

        rnn = Recurrence(gru_with_attention, return_full_state=True)(emb_prev)
        hidden, att_p, att_q = rnn[0], rnn[1], rnn[2]
        readout = C.plus(
            emb_dense(emb_prev),
            att_q_dense(att_q),
            att_p_dense(att_p),
            hidden_dense(hidden)
        )
        word = C.hardmax(C.softmax(readout))  # todo: find the map
        return word

    return decoder


def model_train_factory():
    s2smodel=model_factory()
    @C.Function
    def model_train(question, passage,answer):
        past_answer=Delay(initial_state=start_word)(answer)
        return s2smodel(question,passage,answer)
    return model_train

def model_greedy_factory():
    s2smodel = model_factory()
    #todo: use beam search
    @C.Function
    def model_greedy(question, passage):
        unfold=C.layers.UnfoldFrom(lambda answer:s2smodel(question,passage,answer),
                                   until_predicate=lambda w: w[...,end_word_idx]
                                   )
        return unfold(initial_state=start_word, dynamic_axes_like=passage)# todo: use a new infinity axis
    return model_greedy





