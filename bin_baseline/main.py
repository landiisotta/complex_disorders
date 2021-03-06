import csv
import os

import utils
from utils import disease_name, date_time_folder, data_folder, mrn_file, trimmed_ehr_file, mt_to_ix_file, experiment_folder, model_pars, labels

from model.data_loader import myData, my_collate
import model.net as net
from model.net import metrics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train
from train import train_and_evaluate 

import evaluate
from evaluate import evaluate

def main():
    
    ##pass the size of the vocabulary to the model
    with open(os.path.join(data_folder, mt_to_ix_file)) as f:
            rd = csv.reader(f)
            vocab_size = 0
            for r in rd:
                vocab_size+=1

    #set random seed for reproducible experiments
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)

    ##Import data
    data = myData(data_folder, mrn_file, trimmed_ehr_file, labels)
    data_generator = DataLoader(data, model_pars['batch_size'], shuffle=True, collate_fn=my_collate, drop_last=True)
    #define model and optimizer
    print("cohort numerosity:{0} -- max_seq_length:{1}".format(len(data), len(list(filter(lambda x: x >= 0, data[0][0])))))
    model = net.LSTMehrEncoding(vocab_size, len(data[0][0]), model_pars['embedding_dim'], model_pars['batch_size'])
    #model = nn.DataParallel(model, device_ids=[1,2,3])
    optimizer = torch.optim.Adam(model.parameters(), lr=model_pars['learning_rate'], weight_decay=1e-5)

    #start the unsupervised trauning and evaluation
    model.cuda()
    loss_fn = net.criterion
    print("Starting training for {} epochs...".format(model_pars['num_epochs']))
    mrn, encoded, metrics_avg, lab = train_and_evaluate(model, data_generator, loss_fn, optimizer, experiment_folder, metrics)
    
    #with open(experiment_folder + '/TRencoded_vect.csv', 'w') as f:
    #    wr = csv.writer(f, delimiter=',')
    #    for e in encoded_tr:
    #        wr.writerow(e)

    #with open(experiment_folder + '/TRmrns.csv', 'w') as f:
    #    wr = csv.writer(f, delimiter=',')
    #    for m in mrn_tr:
    #        wr.writerow([m])

    #with open(experiment_folder + '/TRmetrics.txt', 'w') as f:
    #    wr = csv.writer(f, delimiter = '\t')
        #for m, v in metrics_average.items():
        #    wr.writerow([m, v])
    #    wr.writerow(["Mean loss:", loss_tr])

    
    ##load and evaluate best model
    #print("Evaluating best model...")
    #best_saved = torch.load(experiment_folder + '/best_model.pt')
    #model.load_state_dict(best_saved['state_dict'])
    #mrn, encoded, metrics_avg = evaluate(model, loss_fn, data_generator, metrics, best_eval=True)
     

    with open(experiment_folder + '/encoded_vect.csv', 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for e in encoded:
            wr.writerow(e)

    with open(experiment_folder + '/mrns.csv', 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for m in mrn:
            wr.writerow([m])
    
    with open(experiment_folder + '/labels.csv', 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for l in lab:
           wr.writerow([l])

    with open(experiment_folder + '/metrics.txt', 'w') as f:
        wr = csv.writer(f, delimiter='\t')
        #for m, v in metrics_average.items():
        #    wr.writerow([m, v])
        wr.writerow(["Mean loss:", metrics_avg['loss']])
        wr.writerow(["Accuracy:", metrics_avg['accuracy']])
 
if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" %(time.time() - start_time))
