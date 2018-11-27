import torch
import os
from datetime import datetime

##PATHS
disease_name = 'autism'
date_time_folder = '2018-11-20-17-20-33'
data_folder = os.path.expanduser('~/data1/complex_disorders/data/%s/cohorts/%s/' % (disease_name, date_time_folder))
ehr_file = 'cohort-new_ehr.csv'
mt_to_ix_file = 'cohort-new_vocab.csv'

experiment_folder = os.path.expanduser('~/data1/complex_disorders/experiments/') + disease_name +\
                    '-'.join(map(str, list(datetime.now().timetuple()[:6])))
os.makedirs(experiment_folder)

##MODEL PARAMETERS
model_pars = {'num_epochs' : 200,
              'batch_size' : 1, ##batch size equal 1 required
              'embedding_dim' : 128,
              'kernel_size' : 3,
              'learning_rate' : 0.001}

L = 128

def save_best_model(state, folder):
    print("-- Found new best")
    torch.save(state, os.path.join(folder, "best_model.pt"))
