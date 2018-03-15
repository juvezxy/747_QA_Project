# coding:utf-8
from __future__ import unicode_literals, print_function, division
from Evaluate import *
from COREQA import *
from DataUtils import *
from config import *


if __name__ == '__main__':
    # Process Data
    data_loader = DataLoader(syn_data_path)

    # Init Model
    model_params = {}
    model_params["word_indexer"] = data_loader.wordIndexer
    model_params["embedding_size"] = 300
    model_params["state_size"] = 1024
    model_params["ques_attention_size"] = 200
    model_params["kb_attention_size"] = 200
    model_params["learning_rate"] = 0.01
    model_params["MAX_LENGTH"] = 10

    model = COREQA(model_params)

    # Train Model
    model.fit(data_loader.training_data)

    '''
    # Evaluate
    evaluate(model, testPairs, wordIndexer)
    '''