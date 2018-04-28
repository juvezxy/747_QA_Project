# coding:utf-8
from __future__ import unicode_literals, print_function, division

from config import *

class Positioner(nn.Module):
    def __init__(self, state_size, position_size, output_size):
        super(Positioner, self).__init__()
        self.state_size = state_size
        self.output_size = output_size
        self.position_size = position_size

        # Positioner Network
        self.position_state_size = self.state_size * 2
        self.position_mlp_w1 = nn.Linear(self.position_state_size, self.position_size)
        self.position_mlp_w2 = nn.Linear(self.position_size, self.output_size)

    def forward(self, question_embedded):
        position_state = F.tanh(self.position_mlp_w1(question_embedded))
        position_predict = self.position_mlp_w2(position_state)
        return position_predict
