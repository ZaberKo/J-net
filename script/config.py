data_config = {
    'word_size': 25,
    'word_count_threshold': 10,
    'char_count_threshold': 50,
    'pickle_file': 'vocabs.pkl',
    'glove_file': 'glove.6B.300d.txt',
    'emb_dim': 300
}

evidence_extraction_model = {
    'hidden_dim': 100,
    'char_convs': 100,
    'char_emb_dim': 8,
    'dropout': 0.2,
    'highway_layers': 2,
    'two_step': True,
    'use_cudnn': True,
}

answer_synthesis_model = {
    'emb_dim': 300,
    'hidden_dim': 150,
    'attention_dim': 300  # todo: need to be diff form emb_dim
}

training_config = {
    'minibatch_size': 8192,  # in samples when using ctf reader, per worker8192
    'epoch_size': 82326,  # in sequences, when using ctf reader 82326
    'log_freq': 500,  # in minibatchs(print for minibatch number: n*freq)
    'max_epochs': 1,
    'lr': 0.2,
    'momentum': 0.9,
    'train_data': 'train.ctf',  # or 'train.tsv'
    'val_data': 'dev.ctf',
    'val_interval': 1,  # interval in epochs to run validation
    'stop_after': 2,  # num epochs to stop if no CV improvement
    'distributed_after': 0,  # num sequences after which to start distributed training
}
