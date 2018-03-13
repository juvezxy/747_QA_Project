# coding:utf-8
from __future__ import unicode_literals, print_function, division

import math
import time
import re

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

entPattern = re.compile('<.*>')
yearPattern = re.compile('\d+年')
monthPattern = re.compile('\d+月')
dayPattern = re.compile('\d+[日|号]')

SOS = 0
EOS = 1

use_cuda = torch.cuda.is_available()

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