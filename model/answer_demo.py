from cntk.layers import *

from script.utils import BiRecurrence

word_emb_dim = 300
feature_emb_dim = 50
hidden_dim = 150
attention_dim=150
vocab_dim = 10000  # issue

question_seq_axis = C.Axis('questionAxis')
passage_seq_axis = C.Axis('passageAxis')
answer_seq_axis = C.Axis('answerAxis')
raw_question = C.sequence.input_variable((vocab_dim), sequence_axis=question_seq_axis, name='raw_input')
raw_passage = C.sequence.input_variable((vocab_dim), sequence_axis=passage_seq_axis, name='raw_input')
raw_answer = C.sequence.input_variable((vocab_dim), sequence_axis=answer_seq_axis, name='raw_input')

question_seq = raw_question
passage_seq = raw_question
answer_seq = C.sequence.slice(raw_answer, 1)  # <s> A B C </s> --> A B C </s>
answer_seq_start = C.sequence.first(raw_answer)  # <s>
is_first_word = C.sequence.is_first(answer_seq)  # 1 0 0 0 ...
answer_seq_start_scattered = C.sequence.scatter(answer_seq_start,
                                                is_first_word)  # <s> 0 0 0 ... (up to the length of label_sequence)


def question_encoder_factory():
    with default_options(initial_state=0.1):
        model = Sequential([
            Embedding(word_emb_dim, name='embed'),
            Stabilizer(),
            # ht = BiGRU(ht−1, etq)
            BiRecurrence(GRU(shape=hidden_dim / 2), GRU(shape=hidden_dim / 2)),
        ],name='question_encoder')
    return model


def passage_encoder_factory():
    with default_options(initial_state=0.1):
        model = Sequential([
            Embedding(word_emb_dim, name='embed'),
            Stabilizer(),
            # ht = BiGRU(ht−1, [etp, fts, fte])
            BiRecurrence(GRU(shape=hidden_dim / 2), GRU(shape=hidden_dim / 2))
        ],name='passage_encoder')
    return model


def decoder_initialization_factory():
    return splice >> Dense(hidden_dim, activation=C.tanh, bias=True)


def decoder_factory():
    @C.Function
    def decoder(history, input):
        question_encoder = question_encoder_factory()
        passage_encoder = passage_encoder_factory()
        h_b1_q = question_encoder()
        h_b1_p = passage_encoder()
        decoder_initialization = decoder_initialization_factory()
        h = splice(question_encoder, passage_encoder)
        d_0 = decoder_initialization(h_b1_p, h_b1_q)


        gru=GRU(hidden_dim)
        attention = AttentionModel(attention_dim)

        @C.Function
        def GRU__with_attention(source,hidden, x):
            h_att=attention(source,hidden)
            x=splice(x,h_att)
            return gru(hidden,x)

        model=Sequential([
            Embedding(word_emb_dim),
            Stabilizer(),
            Recurrence(GRU__with_attention)
        ])
