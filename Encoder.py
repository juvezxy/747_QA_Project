# coding:utf-8
from __future__ import unicode_literals, print_function, division

from config import *

class Encoder(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(Encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding(inputSize, hiddenSize)
        #self.gru = nn.GRU(hiddenSize, hiddenSize)
        self.lstm = nn.LSTM(hiddenSize, hiddenSize, bidirectional=True)
        self.hidden = self.initHidden()

    def forward(self, input_sent):
        embedded = self.embedding(input_sent).view(len(input_sent), 1, -1)
        outputs, self.hidden = self.lstm(embedded, self.hidden)
        return outputs

    def initHidden(self):
        h = Variable(torch.zeros(2, 1, self.hiddenSize))
        c = Variable(torch.zeros(2, 1, self.hiddenSize))
        if use_cuda:
            return (h.cuda(), c.cuda())
        else:
            return (h, c)
