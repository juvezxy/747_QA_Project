# coding:utf-8
from __future__ import unicode_literals, print_function, division
from Evaluate import *
from COREQA import *
from DataUtils import *
from config import *


if __name__ == '__main__':

    # Process Data
    data_loader = DataLoader(cqa_data_path, True)

    # Init Model
    model_params = {}
    model_params["word_indexer"] = data_loader.wordIndexer
    model_params["embedding_size"] = 200
    model_params["state_size"] = 256
    model_params["mode_size"] = 200
    model_params["ques_attention_size"] = 200
    model_params["kb_attention_size"] = 200
    model_params["learning_rate"] = 0.001
    model_params["mode_loss_rate"] = 0.5
    model_params["batch_size"] = 1
    model_params["epoch_size"] = 1
    model_params["L2_factor"] = 0.0001
    model_params["max_fact_num"] = data_loader.max_fact_num
    model_params["max_ques_len"] = data_loader.max_ques_len
    model_params["MAX_LENGTH"] = 20

    #model = COREQA(model_params)

    # Train Model
    #model.fit(data_loader.training_data)

    # Evaluate
    #evaluate(model, data_loader.testing_data)

