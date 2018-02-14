import importlib
import os
import pickle

from cntk.layers import *

from utils import BiRecurrence


class AnswerSynthesisModel(object):
    def __init__(self, config_file):
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).answer_synthesis_model

        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_file = os.path.join(self.abs_path, 'data', data_config['pickle_file'])
        with open(pickle_file, 'rb') as vf:
            known, vocab, chars, known_npglove_matrix = pickle.load(vf)

        self.emb_dim = model_config['emb_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.attention_dim = model_config['attention_dim']
        self.vocab_dim = len(vocab)
        self.known = known
        self.glove = C.constant(known_npglove_matrix)
        self.nonglove = C.parameter(shape=(self.vocab_dim - self.known, self.emb_dim), init=C.glorot_uniform(),
                                    name='TrainableEmbed')
        self.question_seq_axis = C.Axis.new_unique_dynamic_axis('questionAxis')
        self.passage_seq_axis = C.Axis.new_unique_dynamic_axis('passageAxis')
        self.answer_seq_axis = C.Axis.new_unique_dynamic_axis('answerAxis')
        self.PassageKnownSequence = SequenceOver[self.passage_seq_axis][Tensor[self.known]]
        self.PassageUnknownSequence = SequenceOver[self.passage_seq_axis][Tensor[self.vocab_dim - self.known]]
        self.QuestionKnownSequence = SequenceOver[self.question_seq_axis][Tensor[self.known]]
        self.QuestionUnknownSequence = SequenceOver[self.question_seq_axis][Tensor[self.vocab_dim - self.known]]
        self.AnswerKnownSequence = SequenceOver[self.answer_seq_axis][Tensor[self.known]]
        self.AnswerUnknownSequence = SequenceOver[self.answer_seq_axis][Tensor[self.vocab_dim - self.known]]

        self.bos = '<BOS>'
        self.eos = '<EOS>'
        # todo: what about parameter?
        self.start_word_g = C.constant(np.zeros(self.known, dtype=np.float32))
        self.start_word_u = C.constant(np.zeros(self.vocab_dim - self.known, dtype=np.float32))
        self.end_word_idx = vocab[self.eos]

    def embed_factory(self):
        glove = self.glove
        unglove = self.nonglove

        @C.Function
        def embedding_layer(known_seq, unknown_seq):
            return C.times(known_seq, glove) + C.times(unknown_seq, unglove)

        return embedding_layer

    def encoder_factory(self, name=None):
        with default_options(initial_state=0.1):
            model = BiRecurrence(GRU(shape=self.hidden_dim), GRU(shape=self.hidden_dim), name=name)
        return model

    def model_factory(self):
        emb_layer = self.embed_factory() >> Stabilizer()

        question_encoder = emb_layer >> self.encoder_factory(name='question_encoder')
        passage_encoder = emb_layer >> self.encoder_factory(name='question_encoder')

        q_attention_layer = AttentionModel(self.attention_dim, name='query_attention')
        p_attention_layer = AttentionModel(self.attention_dim, name='passage_attention')

        decoder_gru = GRU(self.hidden_dim)
        decoder_init_dense = Dense(self.hidden_dim, activation=C.tanh, bias=True)
        # for readout_layer
        emb_dense = Dense(self.vocab_dim)
        att_p_dense = Dense(self.vocab_dim)
        att_q_dense = Dense(self.vocab_dim)
        hidden_dense = Dense(self.vocab_dim)

        @C.Function
        def decoder(question_g, question_u, passage_g, passage_u, word_prev):
            # question encoder hidden state
            h_q = question_encoder(question_g, question_u)
            # passage encoder hidden state
            h_p = passage_encoder(passage_g, passage_u)
            # print(h_p)
            # print(h_p.output)
            # print(type(h_p))
            # print(type(h_p.output))
            # print('=================')

            h_q1 = C.sequence.last(h_q)
            h_p1 = C.sequence.last(h_p)

            # word_prev:Tuple(gw,uw)

            emb_prev = emb_layer(word_prev[0], word_prev[1])

            @C.Function
            def gru_with_attention(hidden_prev, att_p_prev, att_q_prev, emb_prev):
                x = C.splice(emb_prev, att_p_prev, att_q_prev, axis=0)
                hidden = decoder_gru(hidden_prev, x)
                att_p = p_attention_layer(h_p.output, hidden)
                att_q = q_attention_layer(h_q.output, hidden)
                return (hidden, att_p, att_q)

            # decoder_initialization
            d_0 = (
                splice(C.slice(h_p1, 0, self.hidden_dim, 0),
                       C.slice(h_q1, 0, self.hidden_dim, 0)) >> decoder_init_dense).output
            # todo: is random better? is parameter better?
            att_p_0 = C.constant(np.zeros(self.attention_dim, dtype=np.float32))
            att_q_0 = C.constant(np.zeros(self.attention_dim, dtype=np.float32))
            init_state = (d_0, att_p_0, att_q_0)

            rnn = Recurrence(gru_with_attention, initial_state=init_state, return_full_state=True)(emb_prev)
            hidden, att_p, att_q = rnn[0], rnn[1], rnn[2]
            readout = C.plus(
                emb_dense(emb_prev),
                att_q_dense(att_q),
                att_p_dense(att_p),
                hidden_dense(hidden)
            )
            word = C.hardmax(C.softmax(readout))
            word_g = C.slice(word, 0, 0, self.known)
            word_u = C.slice(word, 0, self.known, 0)
            return C.combine(word_g, word_u)

        return decoder

    def model_train_factory(self):
        s2smodel = self.model_factory()
        init_state_g = self.start_word_g
        init_state_u = self.start_word_u

        @C.Function
        def model_train(question_g, question_u, passage_g, passage_u, answer_g, answer_u):
            past_answer_g = Delay(initial_state=init_state_g)(answer_g)
            past_answer_u = Delay(initial_state=init_state_u)(answer_u)
            past_answer = C.combine(past_answer_g, passage_u)
            return s2smodel(question_g, question_u, passage_g, passage_u, past_answer)

        return model_train

    # def model_greedy_factory(self):
    #     s2smodel = self.model_factory()
    #
    #     # todo: use beam search
    #     @C.Function
    #     def model_greedy(question: self.QuestionKnownSequence, passage: self.PassageKnownSequence):
    #         unfold = C.layers.UnfoldFrom(lambda answer: s2smodel(question, passage, answer),
    #                                      until_predicate=lambda w: w[..., self.end_word_idx]
    #                                      )
    #         return unfold(initial_state=self.start_word, dynamic_axes_like=passage)  # todo: use a new infinity axis
    #
    #     return model_greedy

    def criterion_factory(self):
        @C.Function
        def criterion(answer_g, answer_u, answer_label_g, answer_label_u):
            loss = C.plus(
                C.cross_entropy_with_softmax(answer_g, answer_label_g),
                C.cross_entropy_with_softmax(answer_u, answer_label_u)
            )
            errs = C.plus(
                C.classification_error(answer_g, answer_label_g),
                C.classification_error(answer_u, answer_label_u)
            )  # todo: runtime rouge-L
            return (loss, errs)

        return criterion

    def model(self):
        question_known = C.sequence.input_variable(self.vocab_dim, sequence_axis=self.question_seq_axis, is_sparse=True,
                                                   name='question_known')
        question_unknown = C.sequence.input_variable(self.vocab_dim, sequence_axis=self.question_seq_axis,
                                                     is_sparse=True,
                                                     name='question_unknown')
        passage_known = C.sequence.input_variable(self.vocab_dim, sequence_axis=self.passage_seq_axis, is_sparse=True,
                                                  name='passage_known')
        passage_unknown = C.sequence.input_variable(self.vocab_dim, sequence_axis=self.passage_seq_axis, is_sparse=True,
                                                    name='passage_unknown')
        answer_known = C.sequence.input_variable(self.vocab_dim, sequence_axis=self.answer_seq_axis, is_sparse=True,
                                                 name='answer_known')
        answer_unknown = C.sequence.input_variable(self.vocab_dim, sequence_axis=self.answer_seq_axis, is_sparse=True,
                                                   name='answer_unknown')

        model = self.model_train_factory()
        criterion_model = self.criterion_factory()

        synthesis_answer = model(question_known, question_unknown, passage_known, passage_unknown, answer_known,
                                 answer_unknown)
        criterion = criterion_model(synthesis_answer[0], synthesis_answer[1], answer_known, answer_unknown)

        return synthesis_answer, criterion


# a=C.input_variable(10,name='a')
# b=C.input_variable(20,name='b')
# model=Sequence(splice,Dense(100))
# print(model)
a = AnswerSynthesisModel('config')
a.model()
