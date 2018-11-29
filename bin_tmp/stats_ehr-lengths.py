
# coding: utf-8

# In[2]:


import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import data_preprocessing_pars
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


file_name = 'ehr-shuffle.csv'

with open(os.path.join(outdir, file_name)) as f:
    rd = csv.reader(f)
    ehr_shuffle = {}
    sub_len = []
    for r in rd:
        sub_len.append(len(r[1::]))
        ehr_shuffle.setdefault(r[0], list()).extend(r[1::])


# In[5]:


print("The average number of tokens for each time slot of N days is {0:.2f}".format(np.mean(sub_len)))


# In[6]:


plt.hist(sub_len)
plt.savefig(os.path.join(outdir, 'hist-encounter_avglengths.jpg'))

# In[7]:


l = []
for mrn in ehr_shuffle:
    l.append(len(ehr_shuffle[mrn]))
print("The average length of ehr sequences is: {0:.2f}".format(np.mean(l)))


# In[8]:


count = 0
for ll in l:
    if ll<3:
        count += 1
print("{0} of {1} patients have less than 3 records".format(count, len(l)))


# In[9]:


print("The sequence length ranges from {0} to {1}".format(min(l), max(l)))


# In[10]:


plt.figure(figsize=[20,10])
plt.xticks(np.arange(0, max(l), 100))
plt.hist(l, bins = 36)
plt.savefig(os.path.join(outdir, 'hist-seq_lengths.jpg'))

# In[11]:


with open(os.path.join(outdir, 'list_mrnToDrop.csv'), 'w') as f:
    wr = csv.writer(f, delimiter=',')
    discard_list = []
    for mrn in ehr_shuffle:
        if len(ehr_shuffle[mrn]) < data_preprocessing_pars['len_min']:
            discard_list.append(mrn)
    wr.writerow(discard_list)


# In[12]:


print("We are dropping {0} out of {1} patients".format(len(discard_list), len(ehr_shuffle)))

