
# coding: utf-8

# In[1]:


'''
###
Term frequency and Document frequency for EHR medical terms.

Drop the least informative terms (among filter_out) with high relative TF and DF.
Plot the distribution of (collection_frequency/collection) * (patient_frequency/patient).
Return the list of stop words.
###
'''
import os
import csv
import numpy as np
from utils import dtype
import re
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def filter_collection_frequencies(outdir):
##Terms to filter, distribution threshold. Do not consider diagnosis
    filter_out = [d for d in dtype if not(bool(re.match('^icd', d)))]

    with open(os.path.join(outdir, 'cohort-vocab.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        lab_vocab = []
        for r in rd:
            lab_vocab.append([r[0], int(r[1])])

    with open(os.path.join(outdir, 'cohort-ehr.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        ehr_seq = {}
        for r in rd:
            ehr_seq.setdefault(r[0], list()).extend([int(r[1])])


# In[2]:


##Compute the collection frequency for each clinical term
    coll_freq = []
    for lv in lab_vocab:
        counts = [ehr_seq[m].count(lv[1]) for m in ehr_seq]
        doc_freq = len(list(filter(lambda x: x!=0, counts)))
        coll_freq.append([lv[1], sum(counts), doc_freq])


# In[3]:


##P(t)*P(t in p), where t is the clinical term and p is a patient
    T = sum([len(ehr_seq[m]) for m in ehr_seq])
    N = len(ehr_seq)

    fr = []
    for l in coll_freq:
        tmp = l[1]/T * l[2]/N
        fr.append(tmp)
        l.append(tmp)


# In[4]:


    coll_freq_sorted = sorted(coll_freq, key=lambda x: x[3]) ##sort by the probability product


# In[5]:




    plt.figure(figsize=[20,20])
    plt.yticks(np.arange(0, max(fr), 0.01))
    plt.plot(fr)
    plt.savefig(os.path.join(outdir, 'hist-term_frequencies.png'))

# In[7]:


    thresh = 0.002
    stop_words = []
    for cfs in reversed(coll_freq_sorted):
        if cfs[3] >= thresh or cfs[3] <= (1/N*1/T):
            #print(lab_vocab[cfs[0]-1], cfs[1::])
            for fo in filter_out:
                if lab_vocab[cfs[0]-1][0].find(fo)!=-1:
                    stop_words.append(cfs[0])


# In[8]:


    print("Discard {0} terms".format(len(stop_words)))


# In[9]:


##Write files stop_words.csv and collection_frequencies.csv

    with open(os.path.join(outdir, 'stop-words.csv'), 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_NONE)
        wr.writerows([stop_words])
    
    with open(os.path.join(outdir, 'collection-frequencies.csv'), 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_NONE)
        wr.writerow(["EVENT", "COLLECTION_FREQUENCY", "PATIENT_FREQUENCY", "CF*PF"])
        for el in coll_freq:
            wr.writerow(el)

