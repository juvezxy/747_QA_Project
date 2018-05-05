import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, input, hidden):
        emb = input
        emb = emb.permute(1, 0, 2)
        _, hidden = self.gru(emb, hidden)
        hidden = hidden.permute(1, 0, 2).contiguous()
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))
        out = F.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)
        out = F.sigmoid(out)
        return out

    def batchClassify(self, inp):
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)

