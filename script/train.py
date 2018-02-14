import importlib
import os
import pickle

import cntk as C
from cntk.device import try_set_default_device, gpu

from answer_synthesis_model import AnswerSynthesisModel


def argument_by_name(func, name):
    found = [arg for arg in func.arguments if arg.name == name]
    if len(found) == 0:
        raise ValueError('no matching names in arguments')
    elif len(found) > 1:
        raise ValueError('multiple matching names in arguments')
    else:
        return found[0]


def create_mb_and_map(func, data_file, vocab_dim, randomize=True, repeat=False):
    mb_source = C.io.MinibatchSource(
        C.io.CTFDeserializer(
            data_file,
            C.io.StreamDefs(
                select_context_words=C.io.StreamDef('mw', shape=vocab_dim, is_sparse=True),
                query_words=C.io.StreamDef('qw', shape=vocab_dim, is_sparse=True),
                answer_words=C.io.StreamDef('aw', shape=vocab_dim, is_sparse=True)
            )),
        randomize=randomize,
        max_sweeps=C.io.INFINITELY_REPEAT if repeat else 1)
    input_map = {
        argument_by_name(func, 'passage'): mb_source.streams.select_context_words,
        argument_by_name(func, 'question'): mb_source.streams.query_words,
        argument_by_name(func, 'answer'): mb_source.streams.answer_words
    }
    return mb_source, input_map


def train(data_path, model_path, log_file, config_file, isrestore=False, profiling=True, gen_heartbeat=False):
    answer_synthesis_model = AnswerSynthesisModel(config_file)
    model, criterion = answer_synthesis_model.model()
    training_config = importlib.import_module(config_file).training_config
    data_config = importlib.import_module(config_file).data_config
    max_epochs = training_config['max_epochs']
    log_freq = training_config['log_freq']
    mb_size = training_config['minibatch_size']
    epoch_size = training_config['epoch_size']

    pickle_file = os.path.join(data_path, data_config['pickle_file'])
    with open(pickle_file, 'rb') as vf:
        known, vocab, chars, npglove_matrix = pickle.load(vf)

    vocab_dim = len(vocab)

    cntk_writer = C.logging.ProgressPrinter(num_epochs=max_epochs, freq=log_freq, tag='Training', log_to_file=log_file,
                                            rank=C.Communicator.rank(), gen_heartbeat=gen_heartbeat)
    tensorboard_writer = C.logging.TensorBoardProgressWriter(10, './tensorboard', None, model)

    # todo: can be improved
    lr = C.learning_parameter_schedule(training_config['lr'], minibatch_size=mb_size)
    momentum = C.momentum_schedule(training_config['momentum'], minibatch_size=mb_size)

    learner = C.adam(model.parameters, lr, momentum,minibatch_size=mb_size,epoch_size=epoch_size)
    # if C.Communicator.num_workers() > 1:
    #     learner = C.data_parallel_distributed_learner(learner)

    if profiling:
        C.debugging.start_profiler(sync_gpu=True)

    train_data_file = os.path.join(data_path, training_config['train_data'])

    mb_source, input_map = create_mb_and_map(model, train_data_file, vocab_dim)

    trainer = C.Trainer(model, criterion, learner, [cntk_writer, tensorboard_writer])

    session = C.training_session(
        trainer=trainer,
        mb_source=mb_source,
        mb_size=mb_size,
        model_inputs_to_streams=input_map,
        progress_frequency=(log_freq, C.DataUnit.minibatch),
        checkpoint_config=C.CheckpointConfig(
            filename=model_path,
            frequency=(epoch_size, C.DataUnit.sample),
            restore=isrestore,
            preserve_all=True
        )

    )

    session.train()

    if profiling:
        C.debugging.stop_profiler()


if __name__ == '__main__':
    # try_set_default_device(gpu(0))
    abs_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(abs_path, 'Models')
    data_path = os.path.join(abs_path, 'data')
    log_path = os.path.join(abs_path, 'log')
    config_file = 'config'
    train(data_path, model_path, log_path, config_file)
