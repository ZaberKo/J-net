import argparse
import importlib
import json
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
    distributed_after = training_config['distributed_after']
    val_interval = training_config['val_interval']
    save_all = training_config['save_all']
    save_freq = training_config['save_freq']
    # pickle_file = os.path.join(data_path, data_config['pickle_file'])
    # with open(pickle_file, 'rb') as vf:
    #     known, vocab, chars, npglove_matrix = pickle.load(vf)


    # ============config trainer===============
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

    progress_writers = [cntk_writer1, cntk_writer2, tensorboard_writer]

    # todo: can be improved
    lr = C.learning_parameter_schedule(training_config['lr'], minibatch_size=mb_size)
    momentum = C.momentum_schedule(training_config['momentum'], minibatch_size=mb_size)

    learner = C.adam(model.parameters, lr, momentum, minibatch_size=mb_size, epoch_size=epoch_size)
    # learner = C.adadelta(model.parameters, lr)
    if C.Communicator.num_workers() > 1:
        learner = C.data_parallel_distributed_learner(learner, distributed_after)

    train_data_file = os.path.join(data_path, training_config['train_data'])
    val_data_file = os.path.join(data_path, training_config['val_data'])
    mb_source, input_map = create_mb_and_map(loss, train_data_file, evidence_extraction_model)

    trainer = C.Trainer(model, (loss, None), learner, progress_writers)

    # ==================================
    if profiling:
        C.debugging.start_profiler(sync_gpu=True)

    model_file = os.path.join(model_path, model_name)
    model_all = C.combine(list(model.outputs) + [loss.output])



    #===========misc=================================
    ema = {}
    dummies = []
    for p in model_all.parameters:
        ema_p = C.constant(0, shape=p.shape, dtype=p.dtype,
                           name='ema_%s' % p.uid)
        ema[p.uid] = ema_p
        dummies.append(C.reduce_sum(
            C.assign(ema_p, 0.999 * ema_p + 0.001 * p)))
    dummy = C.combine(dummies)

    epoch_stat = {
        'best_val_err': 100,
        'best_since': 0,
        'val_since': 0}

    if isrestore and os.path.isfile(model_file):
        trainer.restore_from_checkpoint(model_file)
        # after restore always re-evaluate
        epoch_stat['best_val_err'] = validate(val_data_file, model_all, evidence_extraction_model)

    def post_epoch_work(epoch_stat):
        trainer.summarize_training_progress()
        epoch_stat['val_since'] += 1

        if epoch_stat['val_since'] == val_interval:
            epoch_stat['val_since'] = 0
            temp = dict((p.uid, p.value) for p in model.parameters)
            for p in trainer.model.parameters:
                p.value = ema[p.uid].value
            val_err = validate(val_data_file, model_all, evidence_extraction_model)
            if epoch_stat['best_val_err'] > val_err:
                epoch_stat['best_val_err'] = val_err
                epoch_stat['best_since'] = 0
                trainer.save_checkpoint(model_file)
                for p in trainer.model.parameters:
                    p.value = temp[p.uid]
            else:
                epoch_stat['best_since'] += 1
                if epoch_stat['best_since'] > training_config['stop_after']:
                    return False

        if profiling:
            C.debugging.enable_profiler()

        return True


    # ===============Training part===================
    for epoch in range(max_epochs):
        num_seq = 0
        while True:
            if trainer.total_number_of_samples_seen >= distributed_after:
                data = mb_source.next_minibatch(mb_size * C.Communicator.num_workers(
                ), input_map=input_map, num_data_partitions=C.Communicator.num_workers(),
                                                partition_index=C.Communicator.rank())
            else:
                data = mb_source.next_minibatch(
                    mb_size, input_map=input_map)

            trainer.train_minibatch(data)
            num_seq += trainer.previous_minibatch_sample_count
            dummy.eval()
            if num_seq >= epoch_size:
                break
        if not post_epoch_work(epoch_stat):
            break

    if profiling:
        C.debugging.disable_profiler()
        C.debugging.stop_profiler()


def validate(test_data, model, ee_model):
    begin_prob = model.outputs[0]
    end_prob = model.outputs[1]
    loss = model.outputs[2]
    root = C.as_composite(loss.owner)
    mb_source, input_map = create_mb_and_map(
        root, test_data, ee_model, randomize=False, repeat=False)
    begin_label = argument_by_name(root, 'begin')
    end_label = argument_by_name(root, 'end')


    # rouge-L
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
    minibatch_size = 64
    num_sequences = 0

    stat_sum = 0
    loss_sum = 0
    while True:
        data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
        if not data or not (begin_label in data) or data[begin_label].num_sequences == 0:
            break
        out = model.eval(
            data, outputs=[begin_prob, end_prob, loss], as_numpy=False)
        testloss = out[loss]
        other_input_map = {begin_prediction: out[begin_prob], end_prediction: out[end_prob],
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

    return loss_sum


def test(test_data, model_path, config_file):
    evidence_extraction_model = EvidenceExtractionModel(config_file)
    model = C.load_model(os.path.join(model_path, model_name))
    begin_prob = model.outputs[0]
    end_prob = model.outputs[1]
    loss = C.as_composite(model.outputs[2].owner)

    # print(model)
    # begin_prediction = C.sequence.input_variable(
    #     1, sequence_axis=begin_prob.dynamic_axes[1], needs_gradient=True)
    # end_prediction = C.sequence.input_variable(
    #     1, sequence_axis=end_prob.dynamic_axes[1], needs_gradient=True)
    # best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    # predicted_span = C.layers.Recurrence(C.plus)(
    #     begin_prediction - C.sequence.past_value(end_prediction))
    #
    batch_size = 1  # in sequences
    misc = {'rawctx': [], 'ctoken': [], 'answer': [], 'uid': []}
    tsv_reader = create_tsv_reader(loss, test_data, evidence_extraction_model, batch_size, 1, misc=misc)
    # predicted_span = C.layers.Recurrence(C.plus)(begin_prediction - C.sequence.past_value(end_prediction))
    results = {}
    # mb_source,input_map=create_mb_and_map(loss,test_data,evidence_extraction_model,False,False)
    with open('{}_out.json'.format('FinalModel666'), 'w', encoding='utf-8') as json_output:
        for data in tsv_reader:
            out = model.eval(data, outputs=[begin_prob, end_prob], as_numpy=True)
            # print(out)
            for seq, (raw_text, ctokens, answer, uid) in enumerate(
                    zip(misc['rawctx'], misc['ctoken'], misc['answer'], misc['uid'])):
                begin = np.asarray(out[begin_prob])
                end = np.asarray(out[end_prob])
                # with open('record','w') as f:
                #     f.write(str(begin.tolist()))
                #     f.write('\n==================================\n')
                #     f.write(str(end.tolist()))
                # print(np.argmax(begin))
                span_begin = int(np.argmax(begin))
                span_end = int(np.argmax(end))
                predict_answer = get_answer(raw_text, ctokens, span_begin, span_end)
                results['query_id'] = int(uid)
                print('{} begin:{} end:{}'.format(results['query_id'], span_begin, span_end))
                results['answers'] = [predict_answer]
                json.dump(results, json_output)
                json_output.write("\n")
            # print('ojbk')
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

    test_data = os.path.join(data_path, 'dev.tsv')

    if is_test:
        test(test_data, model_path, config_file)

    else:
        print('===============Training Start=============')
        train(data_path, model_path, log_path, config_file)

        print('===============Training Finish=============')
        C.Communicator.finalize()
