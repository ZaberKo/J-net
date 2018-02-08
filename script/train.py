import importlib

import cntk as C

from script.answer_synthesis_model import AnswerSynthesisModel

HIDDEN_DIM=150
DROPOUT_RATE=0.1
R=0.8
VACABULARY_SIZE=30000
WORD_EMB_DIM=300
FEATURE_EMB_SIZE=50
LR=1


def argument_by_name(func, name):
    found = [arg for arg in func.arguments if arg.name == name]
    if len(found) == 0:
        raise ValueError('no matching names in arguments')
    elif len(found) > 1:
        raise ValueError('multiple matching names in arguments')
    else:
        return found[0]

def create_mb_and_map(func, data_file, vocab_dim, randomize=True, repeat=True):
    mb_source = C.io.MinibatchSource(
        C.io.CTFDeserializer(
            data_file,
            C.io.StreamDefs(
                select_context_words=C.io.StreamDef('mw', shape=vocab_dim, is_sparse=True),
                query_words=C.io.StreamDef('qw', shape=vocab_dim, is_sparse=True),
                answer_words=C.io.StreamDef('aw',shape=vocab_dim,is_sparse=True)
                )),
        randomize=randomize,
        max_sweeps=C.io.INFINITELY_REPEAT if repeat else 1)
    input_map = {
        argument_by_name(func, 'passage'): mb_source.streams.select_context_words,
        argument_by_name(func, 'question'): mb_source.streams.query_words,
        argument_by_name(func, 'answer'): mb_source.streams.answer_words
    }
    return mb_source, input_map


def train(data_path, model_path, log_file, config_file, restore=False, profiling=False, gen_heartbeat=False):
    answer_synthesis_model=AnswerSynthesisModel(config_file)
    synthesis_answer, criterion=answer_synthesis_model.model()
    training_config = importlib.import_module(config_file).training_config
    max_epochs = training_config['max_epochs']
    log_freq = training_config['log_freq']


