# coding:utf-8

from config import *

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


# jieba.load_userdict(config.userDictPath)
def loadData(dataPath):
    qaCount = 0
    qaPairs = []
    print('Reading from file', dataPath, '...')
    with open(dataPath, 'r', encoding='utf-8') as inputFile:
        for line in inputFile:
            question, answer = line.split()
            qaPairs.append((question, answer))
            qaCount += 1
    print(qaCount, 'pairs read.')
    return qaPairs


def processData(qaPairs):
    wordIndexer = WordIndexer()
    trainingPairs = []
    testPairs = []
    print('Processing qa pairs ...')
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
    print('Processing done.', len(trainingPairs), 'training pairs,', len(testPairs), 'test pairs.')
    # print (wordIndexer.wordCount)
    return (wordIndexer, trainingPairs, testPairs)

def varsFromPair(pair):
    inputVar = Variable(torch.LongTensor(pair[0]).view(-1, 1))
    targetVar = Variable(torch.LongTensor(pair[1]).view(-1, 1))
    if use_cuda:
        return (inputVar.cuda(), targetVar.cuda())
    else:
        return (inputVar, targetVar)