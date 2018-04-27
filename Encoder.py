# coding:utf-8
from __future__ import unicode_literals, print_function, division

from config import *

class Encoder(nn.Module):
    def __init__(self, input_size, state_size, embedding_size):
        super(Encoder, self).__init__()
        self.state_size = state_size
        self.lstm = nn.LSTM(embedding_size*3, self.state_size, bidirectional=True)
        self.hidden = self.init_hidden()

    def forward(self, input_sent):
        outputs, self.hidden = self.lstm(input_sent, self.hidden)
        return outputs

    def init_hidden(self):
        h = Variable(torch.zeros(2, 1, self.state_size))
        c = Variable(torch.zeros(2, 1, self.state_size))
        if use_cuda:
            return (h.cuda(), c.cuda())
        else:
            return (h, c)
