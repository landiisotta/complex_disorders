import torch
import os
from datetime import datetime

##PATHS
disease_name = 'autism'
date_time_folder = '2018-10-29-9-55-11'
data_folder = os.path.expanduser('~/data1/complex_disorders/data/%s/cohorts/%s/' % (disease_name, date_time_folder))
mrn_file = 'TRIMMEDordered_mrns.csv'
trimmed_ehr_file = 'trimmed_ehrs.csv'
mt_to_ix_file = 'TRIMMEDcohort-vocab.csv'
labels = 'TRIMMEDordered_labels.csv'

experiment_folder = os.path.expanduser('~/data1/complex_disorders/experiments/') + disease_name +\
                    '-'.join(map(str, list(datetime.now().timetuple()[:6])))
os.makedirs(experiment_folder)

##MODEL PARAMETERS
model_pars = {'num_epochs' : 500,
              'batch_size' : 32,
              'embedding_dim' : 128,
              'kernel_size' : 5,
              'learning_rate' : 0.001}


def save_best_model(state, folder):
    print("-- Found new best")
    torch.save(state, os.path.join(folder, "best_model.pt"))
