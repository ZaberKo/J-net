import importlib
import pickle

import cntk as C
import os
from cntk.layers import *

class EvidenceExtractionModel(object):
    def __init__(self,config_file):
        config = importlib.import_module(config_file)
        data_config = config.data_config
        model_config=config.evidence_extraction_model

        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_file = os.path.join(self.abs_path, 'data', data_config['pickle_file'])
        with open(pickle_file, 'rb') as vf:
            known, vocab, chars, npglove_matrix = pickle.load(vf)

        self.emb_dim