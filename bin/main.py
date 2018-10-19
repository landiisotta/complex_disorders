import utils
from utils import disease_name, date_time_folder, data_folder, mrn_file, padded_ehr_file, mt_to_ix_file, experiment_folder, model_pars

from model.data_loader import myData, my_collate
import model.net as net
from model.net import metrics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train
from train import train_and_evaluate 

def main():

    #set random seed for reproducible experiments
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    ##Import data
    data = myData(data_folder, mrn_file, padded_ehr_file, mt_to_ix_file)
    data_generator = DataLoader(data, model_pars['batch_size'], shuffle=True, collate_fn=my_collate)

    #define model and optimizer
    print("vocab_size:{0} -- max_seq_length:{1}".format(len(data), len(list(filter(lambda x: x > 0, data[2][0])))))
    model = net.ehrEncoding(len(data), len(data[0][0]), model_pars['embedding_dim'], model_pars['kernel_size'])
    #model = nn.DataParallel(model, device_ids=[1,2,3])
    optimizer = torch.optim.Adam(model.parameters(), lr=model_pars['learning_rate'], weight_decay=1e-5)

    #start the unsupervised trauning and evaluation
    model.cuda()
    loss_fn = net.criterion
    print("Starting training for {} epochs...".format(model_pars['num_epochs']))
    train_and_evaluate(model, data_generator, loss_fn, metrics, optimizer, experiment_folder)

    ##load and evaluate best model
    print("Evaluating best model...")
    model.load_state_dict(torch.load(data_folder + 'best_model.pt'))
    mrn, encoded, metrics_avg = evaluate(model, loss_fn, data_generator, metrics, best_eval=True)
    
if __name__ == "__main__":
    main()
    
