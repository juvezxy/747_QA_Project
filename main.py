# coding:utf-8
from __future__ import unicode_literals, print_function, division
from model import *
from data_utils import *
import config

qaPairs = loadData(config.synDataPath)
wordIndexer, trainingPairs, testPairs = processData(qaPairs)
encoder = Encoder(wordIndexer.wordCount, config.hiddenSize)
decoder = Decoder(config.hiddenSize, wordIndexer.wordCount)
if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# use a smaller subset for now
# trainingPairs = trainingPairs[:20000] + trainingPairs[-20000:]
# testPairs = testPairs[:2000] + testPairs[-2000:]
trainIters(trainingPairs, encoder, decoder)
evaluate(testPairs, encoder, decoder, wordIndexer)