import torch
import os

##PATHS
disease_folder = 'autism'
data_folder = os.path.expanduser('~/data1/complex_disorders/data/%s/' % disease_folder) 

ehr_file = 'cohort-new_ehr.csv'
mt_to_ix_file = 'cohort-new_vocab.csv'

##DISEASES AND TERM TYPES
#diseases = ['autism', 'active disintegrative psychoses', 
#            'psychosis with origin in childhood', 
#            'pervasive developmental disorder']
diseases = ['autism']
dtype = ['icd9']

##PREPROCESSING STEP PARAMETERS
data_preprocessing_pars = {'min_diagn' : 10,
                           'n_rndm' : 0,
                           'age_step' : 15,
                           'len_min' : 3}
##MODEL PARAMETERS
model_pars = {'num_epochs' : 1,
              'batch_size' : 1, ##batch size equal 1 required
              'embedding_dim' : 128,
              'kernel_size' : 3,
              'learning_rate' : 0.001}

L = 128

def save_best_model(state, folder):
    print("-- Found new best")
    torch.save(state, os.path.join(folder, "best_model.pt"))
