# coding:utf-8
from __future__ import unicode_literals, print_function, division

from config import *

class Decoder(nn.Module):
    def __init__(self, output_size, state_size, embedding, kb_attention_size, max_fact_num):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.state_size = 2 * state_size
        self.embedding = embedding
        self.embedding_size = self.embedding.weight.size()[1]
        self.kb_attention_size = kb_attention_size
        self.max_fact_num = max_fact_num
        self.lstm = nn.LSTM(self.embedding_size, self.state_size) #TODO: change embedding size to the new embedding mode
        self.out = nn.Linear(self.state_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        # KB attention network
        self.kb_atten_state_size = 2 * self.state_size + 2 * self.embedding_size + self.max_fact_num
        self.kb_atten_mlp_w1 = nn.Linear(self.kb_atten_state_size, self.kb_attention_size)
        self.kb_atten_mlp_w2 = nn.Linear(self.kb_attention_size, 1)

    def forward(self, input_embedded, hidden, question_embedded, kb_facts_embedded, hist_kb):
        output = F.relu(input_embedded)
        output, hidden = self.lstm(output, hidden)

        state = hidden[0]
        ###################### KB attention #################################
        kb_atten_states = torch.cat((state, hist_kb), 2)
        kb_atten_states = [torch.cat((kb_atten_states, kb_fact_embedded, question_embedded), 2) for kb_fact_embedded in
                           kb_facts_embedded]
        kb_atten_states = [F.tanh(self.kb_atten_mlp_w1(kb_atten_state)) for kb_atten_state in kb_atten_states]
        kb_atten_states = [self.kb_atten_mlp_w2(kb_atten_state) for kb_atten_state in kb_atten_states]
        kb_atten_output = torch.cat(kb_atten_states, 2) # Add softmax?
        hist_kb += kb_atten_output

        #####################################################################

        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        h = Variable(torch.zeros(1, 1, self.state_size))
        c = Variable(torch.zeros(1, 1, self.state_size))
        if use_cuda:
            return (h.cuda(), c.cuda())
        else:
            return (h, c)
