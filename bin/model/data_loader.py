import torch
from torch.utils.data import Dataset
import os
import csv

import utils

##define myData class
class myData(Dataset):
    def __init__(self, data_folder, list_mrn_file, padded_ehr_file, mt_to_ix_file):
        with open(os.path.join(data_folder, list_mrn_file)) as f:
            rd = csv.reader(f)
            self.list_mrn = [r for r in rd]
        with open(os.path.join(data_folder, padded_ehr_file)) as f:
            rd = csv.reader(f)
            self.ehr = [list(map(int, r)) for r in rd]    
        with open(os.path.join(data_folder, mt_to_ix_file)) as f:
            rd = csv.reader(f)
            self.vocab_size = 0
            for r in rd:
                self.vocab_size+=1
                    

    def __getitem__(self, index):
        seq = self.ehr[index]
        pat = self.list_mrn[index]
        return seq, pat[0]

    ##len(dataset) returns the number of patients     
    def __len__(self):
        return self.vocab_size

def my_collate(batch):
    data = []
    mrn = []
    for seq, pat in batch:
        data.append(seq)
        mrn.append(pat)
    data = torch.tensor(data, dtype=torch.long)
    return [data, mrn]
        
