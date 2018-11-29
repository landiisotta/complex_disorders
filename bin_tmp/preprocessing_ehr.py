import csv 
import os
import numpy as np
from create_ehr_cohorts import outdir
from utils import data_preprocessing_pars

data_file_name = 'cohort-ehr.csv'
stop_list_file = 'stop-words.csv'

with open(os.path.join(outdir, data_file_name), 'r') as f:
    rd = csv.reader(f)
    next(rd)
    ehrs = {}
    for r in rd:
        ehrs.setdefault(r[0], list()).append([r[1], int(r[2])])

with open(os.path.join(outdir, stop_list_file), 'r') as f:
    rd = csv.reader(f, quotechar="'")
    stop_words = next(rd)

##Sort ehrs with respect to age_in_days and eliminate stop words
ehrs_rid = {}
    
for mrn in ehrs:
    ehrs[mrn] = sorted(ehrs[mrn], key=lambda x: x[1])
    ehrs_rid.setdefault(mrn, list()).extend(list(filter(lambda x: x[0] not in stop_words, ehrs[mrn])))


ehrs_subseq = {}
for mrn in ehrs_rid:
    step = ehrs_rid[mrn][0][1] + data_preprocessing_pars['age_step']
    tmp = set()
    for el in ehrs_rid[mrn]:
        if el[1] <= step:
            tmp.add(int(el[0]))
        else:
            ehrs_subseq.setdefault(mrn, list()).append(tmp)
            tmp = set()
            step = el[1] + data_preprocessing_pars['age_step']
            tmp.add(int(el[0]))
ehrs_subseq.setdefault(mrn, list()).append(tmp)

ehrs_subseq_shuff = {}
for mrn in ehrs_subseq:
    for el in ehrs_subseq[mrn]:
        tmp = list(el)
        np.random.shuffle(tmp)
        ehrs_subseq_shuff.setdefault(mrn, list()).append(tmp)


with open(os.path.join(outdir, 'ehr-shuffle.csv'), 'w') as f:
    wr = csv.writer(f)
    for mrn in ehrs_subseq_shuff:
        for el in ehrs_subseq_shuff[mrn]:
            wr.writerow([mrn] + el)


