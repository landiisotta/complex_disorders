import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ehrEncoding(nn.Module):
    
    def __init__(self, vocab_size, max_seq_len, emb_dim, kernel_size):
        super(ehrEncoding, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.kernel_size = kernel_size

        self.ch_l1 = int(emb_dim / 2)
        self.ch_l2 = int(self.ch_l1 / 2)
        self.padding = int((kernel_size - 1) / 2)
        self.features = math.floor(max_seq_len + 2*self.padding - kernel_size + 1) + 2*self.padding - kernel_size + 1
        
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cnn_l1 = nn.Conv1d(emb_dim, self.ch_l1, kernel_size=kernel_size, padding=self.padding)
        self.cnn_l2 = nn.Conv1d(self.ch_l1, self.ch_l2, kernel_size=kernel_size, padding=self.padding)
        self.linear1 = nn.Linear(self.ch_l2*self.features, 64)
        self.linear2 = nn.Linear(64, vocab_size*max_seq_len)

    def forward(self, x):
        input_vect = x
        
        embeds = self.embedding(x)
        embeds = embeds.permute(0,2,1)
        out = F.relu(self.cnn_l1(embeds))
        out = F.max_pool1d(out, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        out = self.cnn_l2(out)
        out = F.max_pool1d(out, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        out = F.relu(out)
        out = out.view(-1, out.shape[2] * out.shape[1])
        out = self.linear1(out)
        encoded_vect = out
        out = F.relu(out)
        out = self.linear2(out)
        out = out.view(-1, self.vocab_size, input_vect.shape[1])

        return(out, encoded_vect)


def accuracy(out, target):
    logsoft = F.log_softmax(out, dim=1)
    pred = np.argmax(logsoft, dim=1)
    mask = (target > 0).float()
    nb_tokens = int(torch.sum(mask).data[0])
    return np.sum((pred==target).numpy() * mask.numpy())/float(nb_tokens)

criterion = nn.CrossEntropyLoss(ignore_index=0, size_average=True)

metrics = {'accuracy': accuracy}