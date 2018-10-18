import csv
import os
import numpy as np
from collections import OrderedDict

disorder = 'multiple_myeloma'
date_time_folder = '2018-10-10-16-44-0'
data_folder = os.path.expanduser('~/data1/complex_disorders/data/%s/cohorts/%s/' % (disorder, date_time_folder))

with open(os.path.join(data_folder, 'ehr-shuffle.csv')) as f:
    rd = csv.reader(f)
    ehr_shuffle = {}
    for r in rd:
        ehr_shuffle.setdefault(r[0], list()).extend(r[1::])

with open(os.path.join(data_folder, 'list_mrnToDrop.csv')) as f:
    rd = csv.reader(f)
    mrnToDrop = next(rd)

ehr_shuffleRid = {}
for mrn in ehr_shuffle:
    if mrn not in mrnToDrop:
        ehr_shuffleRid[mrn] = ehr_shuffle[mrn]

sorted_mrns = sorted(ehr_shuffleRid, 
                     key=lambda x: len(ehr_shuffleRid[x]), 
                     reverse=True)

X_tmp = [ehr_shuffleRid[mrn] for mrn in sorted_mrns]
X_tmp.sort(key=lambda x: len(x), reverse=True)
X_lengths = [len(x) for x in X_tmp]

##create dictionary
mt_to_ix = OrderedDict()
count = 1
for x in X_tmp:
    for t in x:
        if t not in mt_to_ix:
            mt_to_ix[t] = count
            count += 1

X = [[mt_to_ix[i] for i in x] for x in X_tmp]

padded_X = np.ones((len(X), max(X_lengths)), dtype=int) * int(0)

for i, x_len in enumerate(X_lengths):
    sequence = X[i]
    padded_X[i, 0:x_len] = sequence[:x_len]
 
with open(os.path.join(data_folder, 'padded_ehrs.csv'), 'w') as f:
    wr = csv.writer(f, delimiter=',')
    for w in padded_X:
        wr.writerow(w)

with open(os.path.join(data_folder, 'ordered_mrns.csv'), 'w') as f:
    wr = csv.writer(f, delimiter=',')
    for w in sorted_mrns:
        wr.writerow([w])

with open(os.path.join(data_folder, 'mt_to_ix.csv'), 'w') as f:
    wr = csv.writer(f, delimiter=',')
    for w in mt_to_ix:
        wr.writerow([w, mt_to_ix[w]])
