import argparse
import importlib
import os
import pickle
import ujson

from evidence_extraction_model import EvidenceExtractionModel
from utils import *

model_name = 'ee.model'


def create_mb_and_map(func, data_file, ee_model, randomize=True, repeat=True):
    mb_source = C.io.MinibatchSource(
        C.io.CTFDeserializer(
            data_file,
            C.io.StreamDefs(
                context_g_words=C.io.StreamDef(
                    'cgw', shape=ee_model.wg_dim, is_sparse=True),
                query_g_words=C.io.StreamDef(
                    'qgw', shape=ee_model.wg_dim, is_sparse=True),
                context_ng_words=C.io.StreamDef(
                    'cnw', shape=ee_model.wn_dim, is_sparse=True),
                query_ng_words=C.io.StreamDef(
                    'qnw', shape=ee_model.wn_dim, is_sparse=True),
                answer_begin=C.io.StreamDef(
                    'ab', shape=ee_model.a_dim, is_sparse=False),
                answer_end=C.io.StreamDef(
                    'ae', shape=ee_model.a_dim, is_sparse=False),
                context_chars=C.io.StreamDef(
                    'cc', shape=ee_model.word_size, is_sparse=False),
                query_chars=C.io.StreamDef('qc', shape=ee_model.word_size, is_sparse=False))),
        randomize=randomize,
        max_sweeps=C.io.INFINITELY_REPEAT if repeat else 1)

    input_map = {
        argument_by_name(func, 'passage_gw'): mb_source.streams.context_g_words,
        argument_by_name(func, 'question_gw'): mb_source.streams.query_g_words,
        argument_by_name(func, 'passage_nw'): mb_source.streams.context_ng_words,
        argument_by_name(func, 'question_nw'): mb_source.streams.query_ng_words,
        # argument_by_name(func, 'passage_c'): mb_source.streams.context_chars,
        # argument_by_name(func, 'question_c'): mb_source.streams.query_chars,
        argument_by_name(func, 'begin'): mb_source.streams.answer_begin,
        argument_by_name(func, 'end'): mb_source.streams.answer_end
    }
    return mb_source, input_map


def train(data_path, model_path, log_path, config_file):
    evidence_extraction_model = EvidenceExtractionModel(config_file)
    model, loss = evidence_extraction_model.model()
    training_config = importlib.import_module(config_file).training_config
    data_config = importlib.import_module(config_file).data_config
    max_epochs = training_config['max_epochs']
    log_freq = training_config['log_freq']
    isrestore = training_config['isrestore']
    profiling = training_config['profiling']
    gen_heartbeat = training_config['gen_heartbeat']
    mb_size = training_config['minibatch_size']
    epoch_size = training_config['epoch_size']

    # pickle_file = os.path.join(data_path, data_config['pickle_file'])
    # with open(pickle_file, 'rb') as vf:
    #     known, vocab, chars, npglove_matrix = pickle.load(vf)



    cntk_writer1 = C.logging.ProgressPrinter(
        # freq=log_freq,
        # distributed_freq=100,
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
        rank=C.Communicator.rank(),
        gen_heartbeat=gen_heartbeat
    )
    tensorboard_writer = C.logging.TensorBoardProgressWriter(10, './tensorboard', 0, model)

    # todo: can be improved
    lr = C.learning_parameter_schedule(training_config['lr'], minibatch_size=mb_size)
    momentum = C.momentum_schedule(training_config['momentum'], minibatch_size=mb_size)

    learner = C.adam(model.parameters, lr, momentum, minibatch_size=mb_size, epoch_size=epoch_size)
    # learner = C.adadelta(model.parameters, lr)
    if C.Communicator.num_workers() > 1:
        learner = C.data_parallel_distributed_learner(learner)

    if profiling:
        C.debugging.start_profiler(sync_gpu=True)
        C.debugging.enable_profiler()

    train_data_file = os.path.join(data_path, training_config['train_data'])

    mb_source, input_map = create_mb_and_map(loss, train_data_file, evidence_extraction_model)

    trainer = C.Trainer(model, (loss, None), learner, [cntk_writer1, cntk_writer2, tensorboard_writer])

    session = C.training_session(
        trainer=trainer,
        mb_source=mb_source,
        mb_size=mb_size,
        model_inputs_to_streams=input_map,
        progress_frequency=(epoch_size, C.DataUnit.sample),
        max_samples=max_epochs * epoch_size,
        checkpoint_config=C.CheckpointConfig(
            filename=os.path.join(model_path, model_name),
            frequency=(epoch_size * 10, C.DataUnit.sample),
            restore=isrestore,
            # preserve_all=True
        )

    )

    session.train()
    tensorboard_writer.flush()
    tensorboard_writer.close()

    if profiling:
        C.debugging.disable_profiler()
        C.debugging.stop_profiler()


def validate_model(test_data, model, polymath):
    begin_logits = model.outputs[0]
    end_logits = model.outputs[1]
    loss = model.outputs[2]
    root = C.as_composite(loss.owner)
    mb_source, input_map = create_mb_and_map(
        root, test_data, polymath, randomize=False, repeat=False)
    begin_label = argument_by_name(root, 'ab')
    end_label = argument_by_name(root, 'ae')

    begin_prediction = C.sequence.input_variable(
        1, sequence_axis=begin_label.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input_variable(
        1, sequence_axis=end_label.dynamic_axes[1], needs_gradient=True)

    best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    predicted_span = C.layers.Recurrence(C.plus)(
        begin_prediction - C.sequence.past_value(end_prediction))
    true_span = C.layers.Recurrence(C.plus)(
        begin_label - C.sequence.past_value(end_label))
    common_span = C.element_min(predicted_span, true_span)
    begin_match = C.sequence.reduce_sum(
        C.element_min(begin_prediction, begin_label))
    end_match = C.sequence.reduce_sum(C.element_min(end_prediction, end_label))

    predicted_len = C.sequence.reduce_sum(predicted_span)
    true_len = C.sequence.reduce_sum(true_span)
    common_len = C.sequence.reduce_sum(common_span)
    f1 = 2 * common_len / (predicted_len + true_len)
    exact_match = C.element_min(begin_match, end_match)
    precision = common_len / predicted_len
    recall = common_len / true_len
    overlap = C.greater(common_len, 0)

    def s(x):
        return C.reduce_sum(x, axis=C.Axis.all_axes())

    stats = C.splice(s(f1), s(exact_match), s(precision), s(
        recall), s(overlap), s(begin_match), s(end_match))

    # Evaluation parameters
    minibatch_size = 8192
    num_sequences = 0

    stat_sum = 0
    loss_sum = 0

    while True:
        data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
        if not data or not (begin_label in data) or data[begin_label].num_sequences == 0:
            break
        out = model.eval(
            data, outputs=[begin_logits, end_logits, loss], as_numpy=False)
        testloss = out[loss]
        g = best_span_score.grad({begin_prediction: out[begin_logits], end_prediction: out[end_logits]}, wrt=[
            begin_prediction, end_prediction], as_numpy=False)
        other_input_map = {begin_prediction: g[begin_prediction], end_prediction: g[end_prediction],
                           begin_label: data[begin_label], end_label: data[end_label]}
        stat_sum += stats.eval((other_input_map))
        loss_sum += np.sum(testloss.asarray())
        num_sequences += data[begin_label].num_sequences

    stat_avg = stat_sum / num_sequences
    loss_avg = loss_sum / num_sequences

    print(
        "Validated {} sequences, loss {:.4f}, F1 {:.4f}, EM {:.4f}, precision {:4f}, recall {:4f} hasOverlap {:4f}, start_match {:4f}, end_match {:4f}".format(
            num_sequences,
            loss_avg,
            stat_avg[0],
            stat_avg[1],
            stat_avg[2],
            stat_avg[3],
            stat_avg[4],
            stat_avg[5],
            stat_avg[6]))

    return loss_avg


def test(test_data, model_path, model_file, config_file):
    evidence_extraction_model = EvidenceExtractionModel(config_file)
    model = C.load_model(os.path.join(
        model_path, model_file))
    begin_logits = model.outputs[0]
    end_logits = model.outputs[1]
    loss = C.as_composite(model.outputs[2].owner)
    begin_prediction = C.sequence.input_variable(
        1, sequence_axis=begin_logits.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input_variable(
        1, sequence_axis=end_logits.dynamic_axes[1], needs_gradient=True)
    best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    predicted_span = C.layers.Recurrence(C.plus)(
        begin_prediction - C.sequence.past_value(end_prediction))

    batch_size = 32  # in sequences
    misc = {'rawctx': [], 'ctoken': [], 'answer': [], 'uid': []}
    tsv_reader = create_tsv_reader(
        loss, test_data, evidence_extraction_model, batch_size, 1, is_test=True, misc=misc)
    results = {}
    with open('{}_out.json'.format(model_file), 'w', encoding='utf-8') as json_output:
        for data in tsv_reader:
            out = model.eval(
                data, outputs=[begin_logits, end_logits, loss], as_numpy=False)
            g = best_span_score.grad({begin_prediction: out[begin_logits], end_prediction: out[end_logits]}, wrt=[
                begin_prediction, end_prediction], as_numpy=False)
            other_input_map = {
                begin_prediction: g[begin_prediction], end_prediction: g[end_prediction]}
            span = predicted_span.eval((other_input_map))
            for seq, (raw_text, ctokens, answer, uid) in enumerate(
                    zip(misc['rawctx'], misc['ctoken'], misc['answer'], misc['uid'])):
                seq_where = np.argwhere(span[seq])[:, 0]
                span_begin = np.min(seq_where)
                span_end = np.max(seq_where)
                predict_answer = get_answer(
                    raw_text, ctokens, span_begin, span_end)
                results['query_id'] = int(uid)
                results['answers'] = [predict_answer]
                ujson.dump(results, json_output)
                json_output.write("\n")
            misc['rawctx'] = []
            misc['ctoken'] = []
            misc['answer'] = []
            misc['uid'] = []


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
    if is_test:
        test(None, model_path, None, config_file)
    try:
        print('===============Training Start=============')
        train(data_path, model_path, log_path, config_file)
        print('===============Training Finish=============')
    finally:
        C.Communicator.finalize()
