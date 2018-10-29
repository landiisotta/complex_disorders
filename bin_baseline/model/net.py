import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class LSTMehrEncoding(nn.Module):
    
    def __init__(self, vocab_size, max_seq_len, emb_dim, batch_size):
        super(LSTMehrEncoding, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.emb_dim = emb_dim

        self.n_layers = 2
        self.n_lstm_units = int(self.emb_dim / 2)
        self.ch_l2 = int(self.n_lstm_units / 2)

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.n_lstm_units, self.n_layers, batch_first=True, dropout=0.5)
        #self.linear1 = nn.Linear(self.ch_l2*self.features, 64)
        self.linear1 = nn.Linear(self.n_lstm_units*max_seq_len, self.ch_l2)
        #self.linear1 = nn.Linear(self.ch_l2, 1)            
        self.linear2 = nn.Linear(self.ch_l2, vocab_size*max_seq_len)
        #self.linear1 = nn.Linear(self.n_lstm_units, 1)            
        #self.linear2 = nn.Linear(64, vocab_size*max_seq_len)
        #self.linear2 = nn.Linear(1, vocab_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden_a = torch.randn(self.n_layers, self.batch_size, self.n_lstm_units)
        hidden_b = torch.randn(self.n_layers, self.batch_size, self.n_lstm_units)
        
        hidden_a = hidden_a.cuda()
        hidden_b = hidden_b.cuda()
        
        return (hidden_a, hidden_b)

    def forward(self, x, x_lengths):
        input_vect = x

        embeds = self.embedding(x)
        #print(embeds.shape)
        #out = nn.utils.rnn.pack_padded_sequence(embeds, x_lengths, batch_first=True)
        out, self.hidden = self.lstm(embeds, self.hidden)
        #out, h = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=self.max_seq_len)
        #print(out.shape)
        out = out.contiguous().view(self.batch_size, out.shape[2] * out.shape[1])

        #out = out.view(-1, out.shape[2], out.shape[1])
        #print(out.shape)
        out = self.linear1(out)
        #encoded_vect = out.view(-1,out.shape[1])
        #print(encoded_vect.shape)
        #out = F.relu(out)
        #out = F.relu(out)
        encoded_vect = out
        out = F.relu(out)
        out = self.linear2(out)
        out = out.view(-1, self.vocab_size, input_vect.shape[1])

        #out = self.linear1(out)
        #print(out.shape)
        #encoded_vect = out.view(-1,out.shape[1])
        #print(encoded_vect.shape)
        #out = F.relu(out)
        #print(out.shape)
        #out = self.linear2(out)
        #print(out.shape)
        #out = out.permute(0,2,1)
        #print(out.shape)
        return(out, encoded_vect)


def accuracy(out, target):
    logsoft = F.log_softmax(out, dim=1)
    pred = torch.argmax(logsoft, dim=1)
    #print(pred)
    mask = (target >= 0).float()
    #print(mask)
    nb_tokens = int(torch.sum(mask).item())
    #print(target)
    return torch.sum((pred==target).float() * mask)/float(nb_tokens)

criterion = nn.CrossEntropyLoss()

metrics = {'accuracy': accuracy}
