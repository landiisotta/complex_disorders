import torch
import os
from datetime import datetime

##PATHS
disease_name = 'autism'
date_time_folder = '2018-10-23-17-35-48'
data_folder = os.path.expanduser('~/data1/complex_disorders/data/%s/cohorts/%s/' % (disease_name, date_time_folder))
mrn_file = 'ordered_mrns.csv'
padded_ehr_file = 'padded_ehrs.csv'
mt_to_ix_file = 'mt_to_ix.csv'

experiment_folder = os.path.expanduser('~/data1/complex_disorders/experiments/') + disease_name +\
                    '-'.join(map(str, list(datetime.now().timetuple()[:6])))
os.makedirs(experiment_folder)

##MODEL PARAMETERS
model_pars = {'num_epochs' : 200,
              'batch_size' : 14,
              'embedding_dim' : 128,
              'kernel_size' : 3,
              'learning_rate' : 0.001}


def save_best_model(state, folder):
    print("-- Found new best")
    torch.save(state, os.path.join(folder, "best_model.pt"))
