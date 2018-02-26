from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import math
import time
import re
import random
import config
import jieba

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import recall_score, precision_score

use_cuda = torch.cuda.is_available()


entPattern = re.compile('<.*>')
yearPattern = re.compile('\d+年')
monthPattern = re.compile('\d+月')
dayPattern = re.compile('\d+[日|号]')

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
        # append EOS at end of each sentence
        indexList.append(EOS)
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
    trainingPairs = []
    testPairs = []
    print ('Processing qa pairs ...')
    pairCount = 0
    for (question, answer) in qaPairs:
        questionIndexlist = wordIndexer.addSentence(question)
        answerIndexlist = wordIndexer.addSentence(answer)
        pairCount += 1
        if (pairCount % 10 == 0):
            testPairs.append((questionIndexlist, answer))
        else:
            trainingPairs.append((questionIndexlist, answerIndexlist))
    '''print (questionIndexer.word2index)
    print (answerIndexer.word2index)
    print (qaIndexPairs)'''
    print ('Processing done.', len(trainingPairs), 'training pairs,', len(testPairs), 'test pairs.')
    #print (wordIndexer.wordCount)
    return (wordIndexer, trainingPairs, testPairs)

class Encoder(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(Encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding(inputSize, hiddenSize)
        self.gru = nn.GRU(hiddenSize, hiddenSize)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hiddenSize))
        if use_cuda:
            return result.cuda()
        else:
            return result

class Decoder(nn.Module):
    def __init__(self, hiddenSize, outputSize):
        super(Decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding(outputSize, hiddenSize)
        self.gru = nn.GRU(hiddenSize, hiddenSize)
        self.out = nn.Linear(hiddenSize, outputSize)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = F.relu(embedded)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hiddenSize))
        if use_cuda:
            return result.cuda()
        else:
            return result

def train(inputVar, targetVar, encoder, decoder, encoderOptimizer, decoderOptimizer, criterion):
    encoderOptimizer.zero_grad()
    decoderOptimizer.zero_grad()
    inputLength = inputVar.size()[0]
    targetLength = targetVar.size()[0]
    encoderHidden = encoder.initHidden()
    for i in range(inputLength):
        encoderOutput, encoderHidden = encoder(inputVar[i], encoderHidden)

    decoderInput = Variable(torch.LongTensor([[SOS]]))
    if use_cuda:
        decoderInput = decoderInput.cuda()
    decoderHidden = encoderHidden
    loss = 0
    # Teacher forcing: Feed the target as the next input
    for i in range(targetLength):
        decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
        loss += criterion(decoderOutput, targetVar[i])
        decoderInput = targetVar[i]
    loss.backward()

    encoderOptimizer.step()
    decoderOptimizer.step()
    return loss.data[0] / targetLength

MAX_LENGTH = 20
def predict(inputVar, encoder, decoder):
    inputLength = inputVar.size()[0]
    encoderHidden = encoder.initHidden()
    for i in range(inputLength):
        encoderOutput, encoderHidden = encoder(inputVar[i], encoderHidden)

    decoderInput = Variable(torch.LongTensor([[SOS]]))
    if use_cuda:
        decoderInput = decoderInput.cuda()
    decoderHidden = encoderHidden
    decoded = []
    for i in range(MAX_LENGTH):
        decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
        topv, topi = decoderOutput.data.topk(1)
        token = topi[0][0]
        if token == EOS:
            decoded.append(EOS)
            break
        else:
            decoded.append(token)
        decoderInput = Variable(torch.LongTensor([[token]]))
        if use_cuda:
            decoderInput = decoderInput.cuda()
    return decoded

def varsFromPair(pair):
    inputVar = Variable(torch.LongTensor(pair[0]).view(-1, 1))
    targetVar = Variable(torch.LongTensor(pair[1]).view(-1, 1))
    if use_cuda:
        return (inputVar.cuda(), targetVar.cuda())
    else:
        return (inputVar, targetVar)

def trainIters(pairs, encoder, decoder, learningRate=0.01):
    print ('Training ...')
    startTime = time.time()
    lossTotal = 0

    encoderOptimizer = optim.Adam(encoder.parameters(), lr=learningRate)
    decoderOptimizer = optim.Adam(decoder.parameters(), lr=learningRate)
    #trainingPairs = [varsFromPair(random.choice(pairs)) for i in range(iters)]
    criterion = nn.NLLLoss()

    for iter in range(len(pairs)):
        inputVar, targetVar = varsFromPair(pairs[iter])
        loss = train(inputVar, targetVar, encoder, decoder, encoderOptimizer, decoderOptimizer, criterion)
        lossTotal += loss
        if (iter+1) % 1000 == 0:
            lossAvg = lossTotal / 1000
            lossTotal = 0
            secs = time.time() - startTime
            mins = math.floor(secs / 60)
            secs -= mins * 60
            print ('%dm %ds' % (mins, secs), 'after iteration:', iter+1, 'with avg loss:', lossAvg)

def evaluate(testPairs, encoder, decoder, wordIndexer):
    print ('Evaluating ...')
    precisionTotal = 0
    recallTotal = 0
    F1Total = 0
    testLength = len(testPairs)
    genderCorrect = 0
    genderPredicted = 0
    yearCorrect = 0
    monthCorrect = 0
    dayCorrect = 0
    yearPredicted = 0
    monthPredicted = 0
    dayPredicted = 0
    yearAppear = 0
    monthAppear = 0
    dayAppear = 0
    for i in range(testLength):
        inputSeq, target = testPairs[i]
        inputVar = Variable(torch.LongTensor(inputSeq).view(-1, 1))
        if use_cuda:
            inputVar = inputVar.cuda()
        predictedSeq = predict(inputVar, encoder, decoder)
        '''precision = precision_score(targetSeq, predictedSeq, average='micro')
        recall = recall_score(targetSeq, predictedSeq, average='micro')
        F1 = 2 * (precision * recall) / (precision + recall)
        precisionTotal += precision
        recallTotal += recall
        F1Total += F1
        if (i+1) % 1000 == 0:
            print ('Test size so far:', i, 'precision:', precisionTotal / (i+1), 'recall:', recallTotal / (i+1),
                'F1:', F1Total / (i+1))
        '''
        if (i+1) % 1000 == 0:
            predicted = [wordIndexer.index2word[index] for index in predictedSeq]
            print (predicted)
            print (target)
        
        entIndex = inputSeq[0] # index of the entity
        entity = wordIndexer.index2word[entIndex]
        entityNumber = int(re.findall('\d+', entity)[0])
        predictedEntity = entIndex in predictedSeq
        predictedMale = wordIndexer.word2index['他'] in predictedSeq
        predictedFemale = wordIndexer.word2index['她'] in predictedSeq
        if predictedEntity or predictedMale or predictedFemale:
            genderPredicted += 1
        if predictedEntity or (predictedMale and entityNumber <= 40000) or (predictedFemale and entityNumber > 40000):
            genderCorrect += 1

        yearMatch = yearPattern.search(target)
        monthMatch = monthPattern.search(target)
        dayMatch = dayPattern.search(target)
        yearPredicted += wordIndexer.word2index['年'] in predictedSeq
        monthPredicted += wordIndexer.word2index['月'] in predictedSeq
        dayPredicted += wordIndexer.word2index['日'] in predictedSeq or wordIndexer.word2index['号'] in predictedSeq

        if (yearMatch):
            yearAppear += 1
            year = yearMatch.group()[:-1]
            if wordIndexer.word2index[year] in predictedSeq:
                yearCorrect += 1
        if (monthMatch):
            monthAppear += 1
            month = monthMatch.group()[:-1]
            if wordIndexer.word2index[month] in predictedSeq:
                monthCorrect += 1
        if (dayMatch):
            dayAppear += 1
            day = dayMatch.group()[:-1]
            if wordIndexer.word2index[day] in predictedSeq:
                dayCorrect += 1

    #print ('Average precision:', precisionTotal / testLength, 'recall:', recallTotal / testLength, 'F1:', F1Total / testLength)
    print ('Precision of gender:', genderCorrect * 1.0 / genderPredicted)
    print ('Precision of year:', yearCorrect * 1.0 / yearPredicted)
    print ('Precision of month:', monthCorrect * 1.0 / monthPredicted)
    print ('Precision of day:', dayCorrect * 1.0 / dayPredicted)
    print ('Precision:', (genderCorrect+yearCorrect+monthCorrect+dayCorrect) * 1.0 / (genderPredicted+yearPredicted+monthPredicted+dayPredicted))
    print ('Recall:', (genderCorrect+yearCorrect+monthCorrect+dayCorrect) * 1.0 / (testLength+yearAppear+monthAppear+dayAppear)) 

qaPairs = loadData(config.toyDataPath)
wordIndexer, trainingPairs, testPairs = processData(qaPairs)
encoder = Encoder(wordIndexer.wordCount, config.hiddenSize)
decoder = Decoder(config.hiddenSize, wordIndexer.wordCount)
if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# use a smaller subset for now
trainingPairs = trainingPairs[:20000] + trainingPairs[-20000:]
testPairs = testPairs[:2000] + testPairs[-2000:]
trainIters(trainingPairs, encoder, decoder)
evaluate(testPairs, encoder, decoder, wordIndexer)

