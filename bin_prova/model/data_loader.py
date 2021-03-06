import torch
from torch.utils.data import Dataset
import os
import csv

import utils
from utils import L

##define myData class
class myData(Dataset):
    def __init__(self, data_folder, ehr_file):
        self.ehr = {}
        with open(os.path.join(data_folder, ehr_file)) as f:
            rd = csv.reader(f)
            for r in rd:
                seq = list(map(int, r[1::]))
                if len(seq) < L:
                    self.ehr[r[0]] = seq + [0]*(L-len(seq))
                elif len(seq) % L != 0:
                    self.ehr[r[0]] = seq + [0]*(L - len(seq)%L)
                else:
                    self.ehr[r[0]] = seq
    
    def __getitem__(self, index):
        ehr_list = []
        for mrn, term in self.ehr.items():
            ehr_list.append([mrn, term])
        seq = ehr_list[index][1]
        pat = ehr_list[index][0]
        return seq, pat

    ##len(dataset) returns the number of patients     
    def __len__(self):
        return len(self.ehr)

def my_collate(batch):
    data = []
    mrn = []
    for seq, pat in batch:
        mrn.append(pat)
        if len(seq) == L:
            data.append([seq])
        elif len(seq) % L == 0:
            l = []
            for idx in range(0, len(seq)-L+1, L):
                l.append(seq[idx:idx+L])
            data.append(l)
        else:
            raise Warning("Not all sequences have length multiple than {0}".format(L))

    data = torch.tensor(data, dtype=torch.long)
    data = data.view(-1, L)
    return [data, mrn[0]]
        
