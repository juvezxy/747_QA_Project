from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import config
import jieba

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


entPattern = re.compile('<.*>')

SOS = 0
EOS = 1
class WordIndexer:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.wordCount = 2

    # index each word in the sentence and return a list of indices
    def addSentence(self, sentence):
        indexList = []
        # first split out <ent_###>
        entMatch = entPattern.search(sentence)
        if (entMatch):
            preEnt = sentence[:entMatch.start()]
            ent = entMatch.group()
            postEnt = sentence[entMatch.end():]
            for word in jieba.cut(preEnt, cut_all=False):
                indexList.append(self.addWord(word))
            indexList.append(self.addWord(ent))
            for word in jieba.cut(postEnt, cut_all=False):
                indexList.append(self.addWord(word))
        else:
            for word in jieba.cut(sentence, cut_all=False):
                indexList.append(self.addWord(word))
        return indexList

    # index the word and return its correponding index
    def addWord(self, word):
        if word not in self.word2index:
            index = self.wordCount
            self.word2index[word] = self.wordCount
            self.word2count[word] = 1
            self.index2word[self.wordCount] = word
            self.wordCount += 1
        else:
            self.word2count[word] += 1
            index = self.word2index[word]
        return index


#jieba.load_userdict(config.userDictPath)
def loadData(dataPath):
    qaCount = 0
    qaPairs = []
    print ('Reading from file', dataPath, '...')
    with open(dataPath, 'r', encoding='utf-8') as inputFile:
        for line in inputFile:
            question, answer = line.split()
            qaPairs.append((question, answer))
            qaCount += 1
    print (qaCount, 'pairs read.')
    return qaPairs

def processData(qaPairs):
    wordIndexer = WordIndexer()
    qaIndexPairs = []
    print ('Processing qa pairs ...')
    for (question, answer) in qaPairs:
        questionIndexlist = wordIndexer.addSentence(question)
        answerIndexlist = wordIndexer.addSentence(answer)
        qaIndexPairs.append((questionIndexlist, answerIndexlist))
    '''print (questionIndexer.word2index)
    print (answerIndexer.word2index)
    print (qaIndexPairs)'''
    print ('Processing done.')
    #print (wordIndexer.wordCount)
    return (wordIndexer, qaIndexPairs)

qaPairs = loadData(config.synDataPath)
wordIndexer, qaIndexPairs = processData(qaPairs)


class EncoderRNN(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(EncoderRNN, self).__init__()
        self.hiddenSize = hiddenSize

        self.embedding = nn.Embedding(inputSize, hiddenSize)
        self.gru = nn.GRU(hiddenSize, hiddenSize)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hiddenSize))
        if use_cuda:
            return result.cuda()
        else:
            return result

'''x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x)

x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()

print(z, out)
out.backward()
print(x.grad)'''