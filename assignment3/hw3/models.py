import torch.nn as nn
import torch
from torch.autograd import Variable
import utils
from collections import Counter
import numpy as np



class LSTMModel(nn.Module):

    def __init__(self, word_count, args, dropout_prob=0.2):

        super(LSTMModel, self).__init__()

        self.word_count = word_count
        self.dropout = nn.Dropout(dropout_prob)
        self.embedding = nn.Embedding(
            num_embeddings=word_count,
            embedding_dim=args.embedding_dim)

        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_dim,
                    batch_first=True),
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.hidden_dim,
                    batch_first=True),
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.embedding_dim,
                    batch_first=True)])

        self.projection = nn.Linear(
            in_features=args.embedding_dim,
            out_features=word_count)
        # weight tieing
        self.embedding.weight = self.projection.weight
        self.init_embedding()

    def forward(self, x, forward=0, stochastic=False):

        h = x  # (n, t)
        h = self.dropout(self.embedding(h))  # (n, t, c)

        states = []
        for rnn in self.rnns:
            h, state = rnn(h)
            states.append(state)
        h = self.projection(self.dropout(h))
        if stochastic:
            gumbel = utils.to_variable(
                utils.sample_gumbel(
                    shape=h.size(),
                    out=h.data.new()))
            h += gumbel
        logits = h
        if forward > 0:
            outputs = []
            h = torch.max(logits[:, -1:, :], dim=2)[1] + 1
            for i in range(forward):
                h = self.embedding(self.dropout(h))
                for j, rnn in enumerate(self.rnns):
                    h, state = rnn(h, states[j])
                    states[j] = state
                h = self.projection(self.dropout(h))
                if stochastic:
                    gumbel = utils.to_variable(
                        utils.sample_gumbel(
                            shape=h.size(), out=h.data.new()))
                    h += gumbel
                outputs.append(h)
                h = torch.max(h, dim=2)[1] + 1
            logits = torch.cat([logits] + outputs, dim=1)
        return logits

    def init_embedding(self):
        # Load words and vocabulary
        words = np.concatenate(np.load('dataset/wiki.train.npy'))
        vocab = np.load('dataset/vocab.npy')

        # Count each word
        vocab_size = vocab.shape[0]
        counter = Counter(words)
        word_counts = np.array([counter[i] for i in range(vocab_size)], dtype=np.float32)
        word_count = np.sum(word_counts)

        # P(word)
        word_probabilities = word_counts / word_count
        # log(P(word))
        epsilon = 1e-12
        word_logits = np.log(word_probabilities + epsilon)

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.projection.weight.data.uniform_(-0.1, 0.1)
        self.projection.bias.data = torch.from_numpy(word_logits)

class LSTMModelSingle(nn.Module):

    def __init__(self, word_count,embedding_dim, hidden_dim, dropout_prob=0.2):

        super(LSTMModelSingle, self).__init__()

        self.word_count = word_count
        self.dropout = nn.Dropout(dropout_prob)
        self.embedding = nn.Embedding(
            num_embeddings=word_count,
            embedding_dim=embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, 3, dropout=dropout_prob)

        self.projection = nn.Linear(
            in_features=embedding_dim,
            out_features=word_count)
        # weight tieing
        self.embedding.weight = self.projection.weight
        self.init_embedding()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden, forward=0, stochastic=False):
        emb = self.dropout(self.embedding(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout(output)
        decoded = self.projection(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
    
    def init_embedding(self):
        # Load words and vocabulary
        words = np.concatenate(np.load('dataset/wiki.train.npy'))
        vocab = np.load('dataset/vocab.npy')

        # Count each word
        vocab_size = vocab.shape[0]
        counter = Counter(words)
        word_counts = np.array([counter[i] for i in range(vocab_size)], dtype=np.float32)
        word_count = np.sum(word_counts)

        # P(word)
        word_probabilities = word_counts / word_count
        # log(P(word))
        epsilon = 1e-12
        word_logits = np.log(word_probabilities + epsilon)

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.projection.weight.data.uniform_(-0.1, 0.1)
        self.projection.bias.data = torch.from_numpy(word_logits)
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(3, bsz, self.hidden_dim).zero_()),
                    Variable(weight.new(3, bsz, self.hidden_dim).zero_()))


class LSTMModelV2(nn.Module):

    def __init__(
            self,
            word_count,
            embedding_dim,
            hidden_dim,
            dropout_prob=0.5):

        super(LSTMModelV2, self).__init__()

        self.word_count = word_count
        self.dropout = nn.Dropout(dropout_prob)
        self.embedding = nn.Embedding(
            num_embeddings=word_count,
            embedding_dim=embedding_dim)

        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                    batch_first=True),
            nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                    batch_first=True),
            nn.LSTM(input_size=hidden_dim, hidden_size=embedding_dim,
                    batch_first=True)])

        self.projection = nn.Linear(
            in_features=embedding_dim,
            out_features=word_count)
        # weight tieing
        self.embedding.weight = self.projection.weight
        self.init_embedding()

    def forward(self, x, forward=0, stochastic=False):

        h = x  # (n, t)
        h = self.embedding(h)  # (n, t, c)
        states = []
        for rnn in self.rnns:
            h, state = rnn(h)
            states.append(state)
        h = self.projection(h)
        if stochastic:
            gumbel = utils.to_variable(
                utils.sample_gumbel(
                    shape=h.size(),
                    out=h.data.new()))
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
                    gumbel = utils.to_variable(
                        utils.sample_gumbel(
                            shape=h.size(), out=h.data.new()))
                    h += gumbel
                outputs.append(h)
                h = torch.max(h, dim=2)[1] + 1
            logits = torch.cat([logits] + outputs, dim=1)
        return logits

    def init_embedding(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.projection.bias.data.fill_(0)


class CrossEntropyLoss3D(nn.CrossEntropyLoss):
    def forward(self, input, target):
        return super(CrossEntropyLoss3D, self).forward(
            input.view(-1, input.size()[2]), target.view(-1))
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
