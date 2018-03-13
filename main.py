# coding:utf-8
from __future__ import unicode_literals, print_function, division
from Evaluate import *
from COREQA import *
from DataUtils import *
import config


if __name__ == '__main__':
    # Process Data
    qaPairs = loadData(config.synDataPath)
    wordIndexer, trainingPairs, testPairs = processData(qaPairs)

    # Init Model
    model_params = {}
    model_params["wordIndexer"] = wordIndexer
    model_params["embeddingSize"] = 256
    model_params["learningRate"] = 0.1
    model_params["MAX_LENGTH"] = 20

    model = COREQA(model_params)

    # Train Model
    model.fit(trainingPairs)

    # Evaluate
    evaluate(model, testPairs, wordIndexer)