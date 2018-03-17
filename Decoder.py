# coding:utf-8
from __future__ import unicode_literals, print_function, division

from config import *

class Decoder(nn.Module):
    def __init__(self, output_size, state_size, embedding, mode_size, kb_attention_size, max_fact_num, ques_attention_size, max_ques_len):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.state_size = 2 * state_size
        self.embedding = embedding
        self.embedding_size = self.embedding.weight.size()[1]
        self.mode_size = mode_size
        self.kb_attention_size = kb_attention_size
        self.max_fact_num = max_fact_num
        self.ques_attention_size = ques_attention_size
        self.max_ques_len = max_ques_len
        self.lstm_input_size = 5 * self.embedding_size + self.state_size
        self.lstm = nn.LSTM(self.lstm_input_size, self.state_size)
        self.out = nn.Linear(self.state_size, self.output_size)

        # Question attention network
        self.ques_atten_state_size = 2 * self.state_size + self.max_ques_len
        self.ques_atten_mlp_w1 = nn.Linear(self.ques_atten_state_size, self.ques_attention_size)
        self.ques_atten_mlp_w2 = nn.Linear(self.ques_attention_size, 1)

        # KB attention network
        self.kb_atten_state_size = 2 * self.state_size + 2 * self.embedding_size + self.max_fact_num
        self.kb_atten_mlp_w1 = nn.Linear(self.kb_atten_state_size, self.kb_attention_size)
        self.kb_atten_mlp_w2 = nn.Linear(self.kb_attention_size, 1)

        # mode prediction network
        self.mode_state_size = self.embedding_size + self.state_size
        self.mode_mlp_w1 = nn.Linear(self.mode_state_size, self.mode_size)
        self.mode_mlp_w2 = nn.Linear(self.mode_size, 3)

    def forward(self, input_embedded, input_cat, hidden, question_embedded, kb_facts_embedded, hist_kb, encoder_outputs, hist_ques):
        #output = F.relu(input_cat)
        output, hidden = self.lstm(input_cat, hidden)

        state = hidden[0]
        ##################### Mode prediction ###############################
        mode_state = torch.cat((state, input_embedded), 2)
        mode_state = F.tanh(self.mode_mlp_w1(mode_state))
        mode_predict = self.mode_mlp_w2(mode_state)
        #####################################################################

        ##################### Ques attention ################################
        ques_atten_states = torch.cat((state, hist_ques), 2)
        ques_atten_states = [torch.cat((ques_atten_states, encoder_output[0].view(1, 1, -1)), 2) for encoder_output in
                           encoder_outputs]
        ques_atten_states = [F.tanh(self.ques_atten_mlp_w1(ques_atten_state)) for ques_atten_state in ques_atten_states]
        ques_atten_states = [self.ques_atten_mlp_w2(ques_atten_state) for ques_atten_state in ques_atten_states]
        ques_atten_predict = torch.cat(ques_atten_states, 2)
        hist_ques += ques_atten_predict
        #####################################################################

        ###################### KB attention #################################
        kb_atten_states = torch.cat((state, hist_kb), 2)
        kb_atten_states = [torch.cat((kb_atten_states, kb_fact_embedded, question_embedded), 2) for kb_fact_embedded in
                           kb_facts_embedded]
        kb_atten_states = [F.tanh(self.kb_atten_mlp_w1(kb_atten_state)) for kb_atten_state in kb_atten_states]
        kb_atten_states = [self.kb_atten_mlp_w2(kb_atten_state) for kb_atten_state in kb_atten_states]
        kb_atten_predict = torch.cat(kb_atten_states, 2)
        hist_kb += kb_atten_predict
        #####################################################################

        common_predict = self.out(output)
        return common_predict, hidden, mode_predict, kb_atten_predict, hist_kb, ques_atten_predict, hist_ques

    def init_hidden(self):
        h = Variable(torch.zeros(1, 1, self.state_size))
        c = Variable(torch.zeros(1, 1, self.state_size))
        if use_cuda:
            return (h.cuda(), c.cuda())
        else:
            return (h, c)
