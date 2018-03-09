import torch.nn as nn
import torch
from torch.autograd import Variable
import utils
class LSTMModel(nn.Module):

    def __init__(self, word_count, args, dropout_prob=0.5):
        
        super(LSTMModel, self).__init__()
        
        self.word_count = word_count
        self.dropout   = nn.Dropout(dropout_prob)
        self.embedding = nn.Embedding(num_embeddings = word_count, embedding_dim=args.embedding_dim)
        
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_dim, batch_first=True),
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.hidden_dim, batch_first=True),
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.embedding_dim, batch_first=True)])
        
        self.projection = nn.Linear(in_features=args.embedding_dim, out_features=word_count)
        #weight tieing
        self.embedding.weight = self.projection.weight
        
    def forward(self, x, forward=0, stochastic=False):
        
        h = x  # (n, t)
        h = self.embedding(h)  # (n, t, c)
        states = []
        for rnn in self.rnns:
            h, state = rnn(h)
            states.append(state)
        h = self.projection(h)
        if stochastic:
            gumbel = Variable(utils.sample_gumbel(shape=h.size(), out=h.data.new()))
            h += gumbel
        logits = h
        if forward > 0:
            outputs = []
            h = torch.max(logits[:, -1:, :], dim=2)[1] + 1
            for i in range(forward):
                h = self.embedding(h)
                for j, rnn in enumerate(self.rnns):
                    h, state = rnn(h, states[j])
                    states[j] = state
                h = self.projection(h)
                if stochastic:
                    gumbel = Variable(utils.sample_gumbel(shape=h.size(), out=h.data.new()))
                    h += gumbel
                outputs.append(h)
                h = torch.max(h, dim=2)[1] + 1
            logits = torch.cat([logits] + outputs, dim=1)
        return logits
    
class CrossEntropyLoss3D(nn.CrossEntropyLoss):
    def forward(self, input, target):
        return super(CrossEntropyLoss3D, self).forward(input.view(-1, input.size()[2]), target.view(-1))
