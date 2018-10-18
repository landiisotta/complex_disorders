import torch
import os
from datetime import datetime

##PATHS
disease_name = 'multiple_myeloma'
date_time_folder = '2018-10-10-16-44-0'
data_folder = os.path.expanduser('~/data1/complex_disorders/data/%s/cohorts/%s/' % (disease_name, date_time_folder))
mrn_file = 'ordered_mrns.csv'
padded_ehr_file = 'padded_ehrs.csv'
mt_to_ix_file = 'mt_to_ix.csv'

experiment_folder = os.path.expanduser('~/data1/complex_disorders/experiments') + disease_name +\
                    '-'.join(map(str, list(datetime.now().timetuple()[:6])))
#os.makedirs(experiment_folder)

##MODEL PARAMETERS
model_pars = {'num_epochs' : 1,
              'batch_size' : 2,
              'embedding_dim' : 128,
              'kernel_size' : 3,
              'learning_rate' : 0.001}


def save_best_model(model, folder, is_best):
    if(is_best):
        print("-- Found new best")
        torch.save(model.state_dict(), os.path.join(folder, "best_model.pt"))
