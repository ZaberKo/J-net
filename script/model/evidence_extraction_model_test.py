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
        self.use_cuDNN=model_config['use_cuDNN']
        self.question_seq_axis = C.Axis.new_unique_dynamic_axis('questionAxis')
        self.passage_seq_axis = C.Axis.new_unique_dynamic_axis('passageAxis')
        self.PassageSequence = SequenceOver[self.passage_seq_axis][Tensor[self.vocab_dim]]
        self.QuestionSequence = SequenceOver[self.question_seq_axis][Tensor[self.vocab_dim]]
        self.emb_layer = Embedding(weights=npglove_matrix)

    def question_encoder_factory(self):
        with default_options(enable_self_stabilization=True):
            model = Sequential([
                self.emb_layer,
                Stabilizer(),
                # ht = BiGRU(ht−1, etq)
                BiGRU(self.hidden_dim,use_cudnn=self.use_cuDNN)
            ], name='question_encoder')
        return model

    def passage_encoder_factory(self):
        with default_options(enable_self_stabilization=True):
            model = Sequential([
                self.emb_layer,
                Stabilizer(),
                # ht = BiGRU(ht−1, etq)
                BiGRU(self.hidden_dim, use_cudnn=self.use_cuDNN)
            ], name='question_encoder')
        return model

    def network_fac(self):
        question_encoder = self.question_encoder_factory()
        passage_encoder = self.passage_encoder_factory()

        C_Q_gru = GRU(self.hidden_dim)
        r_Q_att_layer = AttentionModel(self.attention_dim, name='r_Q_att_layer')
        C_Q_att_layer = AttentionModel(self.attention_dim, name='C_Q_att_layer')

        @C.Function
        def netowrk(question:self.QuestionSequence,passage:self.PassageSequence):
            U_Q = question_encoder(question)  # ('question_encoder', [#, questionAxis], [300])
            U_P = passage_encoder(passage)

            # U_Q = question
            # U_P = passage

            @C.Function
            def gru_with_attention(hidden_prev, x):
                #todo: structure changed, rewrite new att_layer
                C_Q = C_Q_att_layer(U_Q.output,hidden_prev)
                hidden = C_Q_gru(hidden_prev,C.splice(C_Q,x))
                return hidden

            V_P = Recurrence(gru_with_attention, name='V_P_GRU')

            return V_P(U_P)

        return netowrk

    def criterion_factory(self):
        @C.Function
        def criterion(begin: SequenceOver[self.passage_seq_axis], end: SequenceOver[self.passage_seq_axis],
                      begin_label: SequenceOver[self.passage_seq_axis], end_label: SequenceOver[self.passage_seq_axis]):
            loss = C.plus(C.binary_cross_entropy(begin, begin_label), C.binary_cross_entropy(end, end_label))
            return loss

        return criterion

    def model(self):
        question_seq = C.sequence.input_variable(self.vocab_dim, sequence_axis=self.question_seq_axis, name='question')
        passage_seq = C.sequence.input_variable(self.vocab_dim, sequence_axis=self.passage_seq_axis, name='passage')
        begin = C.sequence.input_variable(1, sequence_axis=self.passage_seq_axis, name='begin')
        end = C.sequence.input_variable(1, sequence_axis=self.passage_seq_axis, name='end')

        # print(C.sequence.last(begin))
        model = self.network_fac()(question_seq, passage_seq)
        print(model)

        #
        # loss = criterion(p1,p2,begin,end)
        # return C.combine([p1,p2]),loss


a = EvidenceExtractionModel('config')
a.model()
