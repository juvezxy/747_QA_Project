# coding:utf-8
from __future__ import unicode_literals, print_function, division

from config import *

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.lstm = nn.LSTM(embedding.weight.size()[1], hidden_size, bidirectional=True)
        self.hidden = self.initHidden()

    def forward(self, input_sent):
        embedded = self.embedding(input_sent).view(len(input_sent), 1, -1)
        outputs, self.hidden = self.lstm(embedded, self.hidden)
        return outputs

    def initHidden(self):
        h = Variable(torch.zeros(2, 1, self.hidden_size))
        c = Variable(torch.zeros(2, 1, self.hidden_size))
        if use_cuda:
            return (h.cuda(), c.cuda())
        else:
            return (h, c)
