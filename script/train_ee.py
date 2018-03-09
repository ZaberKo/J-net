import argparse
import importlib
import os

from evidence_extraction_model import EvidenceExtractionModel
from utils import *

model_name = 'ee.model'





def train(data_path, model_path, log_path, config_file):
    evidence_extraction_model = EvidenceExtractionModel(config_file)
    model, loss = evidence_extraction_model.model()
    training_config = importlib.import_module(config_file).training_config
    # data_config = importlib.import_module(config_file).data_config
    max_epochs = training_config['max_epochs']
    log_freq = training_config['log_freq']
    isrestore = training_config['isrestore']
    profiling = training_config['profiling']
    gen_heartbeat = training_config['gen_heartbeat']
    mb_size = training_config['minibatch_size']
    epoch_size = training_config['epoch_size']
    distributed_after = training_config['distributed_after']
    save_all = training_config['save_all']
    save_freq = training_config['save_freq']



    cntk_writer1 = C.logging.ProgressPrinter(
        freq=log_freq,
        distributed_freq=log_freq,
        num_epochs=max_epochs,
        tag='Training',
        log_to_file=os.path.join(log_path, 'log_'),
        rank=C.Communicator.rank(),
        gen_heartbeat=gen_heartbeat
    )
    cntk_writer2 = C.logging.ProgressPrinter(
        freq=log_freq,
        distributed_freq=log_freq,
        num_epochs=max_epochs,
        tag='Training2std',
        rank=0,
        gen_heartbeat=gen_heartbeat
    )
    tensorboard_writer = C.logging.TensorBoardProgressWriter(10, './tensorboard', 0, model)

    # todo: can be improved
    lr = C.learning_parameter_schedule(training_config['lr'], minibatch_size=mb_size)
    momentum = C.momentum_schedule(training_config['momentum'], minibatch_size=mb_size)

    learner = C.adam(model.parameters, lr, momentum, minibatch_size=mb_size, epoch_size=epoch_size)
    # learner = C.adadelta(model.parameters, lr)
    if C.Communicator.num_workers() > 1:
        learner = C.data_parallel_distributed_learner(learner, distributed_after)

    if profiling:
        C.debugging.start_profiler(sync_gpu=True)
        C.debugging.enable_profiler()

    train_data_file = os.path.join(data_path, training_config['train_data'])
    mb_source, input_map = create_mb_and_map(loss, train_data_file, evidence_extraction_model)

    trainer = C.Trainer(model, (loss, None), learner, [cntk_writer1, cntk_writer2, tensorboard_writer])
    # print('model',trainer.model)
    session = C.training_session(
        trainer=trainer,
        mb_source=mb_source,
        mb_size=mb_size,
        model_inputs_to_streams=input_map,
        progress_frequency=(epoch_size, C.DataUnit.sample),
        max_samples=max_epochs * epoch_size,
        checkpoint_config=C.CheckpointConfig(
            filename=os.path.join(model_path, model_name),
            frequency=(epoch_size * save_freq, C.DataUnit.sample),
            restore=isrestore,
            preserve_all=save_all
        )

    )
    session.train()
    tensorboard_writer.flush()
    tensorboard_writer.close()

    if profiling:
        C.debugging.disable_profiler()
        C.debugging.stop_profiler()


if __name__ == '__main__':
    # C.try_set_default_device(C.cpu())
    abs_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(abs_path, 'Model')
    data_path = os.path.join(abs_path, 'data')
    log_path = os.path.join(abs_path, 'log')

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config',
                        help='Config file', required=False, default='config')
    parser.add_argument(
        '-r', '--restart',
        help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)',
        action='store_true')
    parser.add_argument('-test', '--test',
                        help='Test data file', required=False, default=False)

    args = vars(parser.parse_args())

    config_file = args['config']

    is_test = args['test']

    test_data = os.path.join(data_path, 'dev.tsv')

    print('===============Training Start=============')
    train(data_path, model_path, log_path, config_file)

    print('===============Training Finish=============')
    C.Communicator.finalize()
