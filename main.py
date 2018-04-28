# coding:utf-8
from __future__ import unicode_literals, print_function, division
from Evaluate import *
from QAGAN import *
from DataUtilsNew import *
from config import *


if __name__ == '__main__':

    # Process Data
    #data_loader = DataLoader(cqa_data_path, True)
    #data_loader = DataLoader(syn_data_path, False)
    if load_from_preprocessed:
        print ("Loading from preprocessed file ...")
        with open(preprocessed_data_path, 'rb') as preprocessed:
            data_loader = pickle.load(preprocessed)
        with open(preprocessed_data_path + "training1", 'rb') as preprocessed:
            data_loader.training_data = pickle.load(preprocessed)
        with open(preprocessed_data_path + "testing1", 'rb') as preprocessed:
            data_loader.testing_data = pickle.load(preprocessed)
        for i in range(2, 7):
            with open(preprocessed_data_path + "training"+str(i), 'rb') as preprocessed:
                data_loader.training_data += pickle.load(preprocessed)
            with open(preprocessed_data_path + "testing"+str(i), 'rb') as preprocessed:
                data_loader.testing_data += pickle.load(preprocessed)
    else:
        data_loader = DataLoader(msmarco_path, True)
        print ("Saving to preprocessed file ...")
        with open(preprocessed_data_path, 'wb') as preprocessed:
            pickle.dump(data_loader, preprocessed)

    # Init Model
    model_params = {}
    model_params["word_indexer"] = data_loader.wordIndexer
    model_params["embedding_size"] = 1024
    model_params["state_size"] = 512
    model_params["mode_size"] = 200
    model_params["ques_attention_size"] = 200
    model_params["kb_attention_size"] = 200
    model_params["learning_rate"] = 0.0001
    model_params["mode_loss_rate"] = 1.0
    model_params["batch_size"] = 1
    model_params["epoch_size"] = 1
    model_params["L2_factor"] = 0.000001
    model_params["max_fact_num"] = data_loader.max_fact_num
    model_params["max_ques_len"] = data_loader.max_ques_len
    model_params["MAX_LENGTH"] = 20

    model = QAGAN(model_params)

    # Train Model
    model.fit(data_loader.training_data)

    # Evaluate
    evaluate(model, data_loader.testing_data, data_loader.gold_answer, True)
