# coding:utf-8
from __future__ import unicode_literals, print_function, division

from config import *

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, embedding):
        super(Decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.embedding = embedding
        self.lstm = nn.LSTM(embedding.weight.size()[1], hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # KB

        embedded = self.embedding(input).view(1, 1, -1)
        output = F.relu(embedded)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        h = Variable(torch.zeros(2, 1, self.hiddenSize))
        c = Variable(torch.zeros(2, 1, self.hiddenSize))
        if use_cuda:
            return (h.cuda(), c.cuda())
        else:
            return (h, c)
