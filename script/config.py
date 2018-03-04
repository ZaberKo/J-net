data_config = {
    'word_size': 20,
    'word_count_threshold': 10,
    'char_count_threshold': 50,
    'pickle_file': 'vocabs.pkl',
    'glove_file': 'glove.6B.300d.txt',
    'emb_dim': 300,  # mast correspond to evidence_extraction_model['word_emb_dim']
    'is_limited_type': True,
    'limited_types': ['description']
}

evidence_extraction_model = {
    'word_emb_dim': 300,
    # 'char_convs': 100,
    # 'char_emb_dim': 8,
    'hidden_dim': 150,
    'attention_dim': 200,
    'dropout': 0.2,
    'use_cuDNN': True,  # GPU required
    'use_sparse': True
}

answer_synthesis_model = {
    'emb_dim': 100,
    'hidden_dim': 150,
    'attention_dim': 300,
    'dropout': 0.2,
}

training_config = {
    'minibatch_size': 128,  # in samples when using ctf reader, per worker8192
    'epoch_size': 44179,  # in sequences, when using ctf reader 44179
    'log_freq': 100,  # in minibatchs(print for minibatch number: n*freq)
    'max_epochs': 5,
    'lr': 0.02,
    'momentum': 0.5,
    'isrestore': False,
    'profiling': False,
    'gen_heartbeat': False,
    'train_data': 'train.ctf',  # or 'train.tsv'
    'val_data': 'dev.ctf',
    'restore_all': True,
    'restore_freq': 1,
    'val_interval': 1,  # interval in epochs to run validation
    'stop_after': 2,  # num epochs to stop if no CV improvement
    'distributed_after': 0,  # num sequences after which to start distributed training
}
