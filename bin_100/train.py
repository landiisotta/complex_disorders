import csv
import torch
import utils
from utils import model_pars, experiment_folder
import numpy as np
from evaluate import evaluate

def train(model, optimizer, loss_fn, data_iterator):
    model.train()
    encoded_list = []
    mrn_list = []   
    lab_list = []
    loss_batch = []
    for idx, (batch, mrn, lab) in enumerate(data_iterator):
        batch = batch.cuda()
            
        optimizer.zero_grad()
        out, encoded_vect = model(batch)
        loss = loss_fn(out, batch)
        loss.backward()
        optimizer.step()
        loss_batch.append(loss.item())
        
        encoded_list += encoded_vect.tolist()
        mrn_list.extend([m for m in mrn])
        lab_list.extend([l for l in lab])
        #print("Batch: {}".format(idx))

    loss_mean = np.mean(loss_batch)

    return lab_list, mrn_list, encoded_list, loss_mean

def train_and_evaluate(model, data_iterator, loss_fn, optimizer, model_dir, metrics):
    #best_eval_acc = 0.0
    num_epochs = model_pars['num_epochs']
    for epoch in range(num_epochs):
        print("Epoch {0} of {1}".format(epoch, num_epochs))
        lab, mrn, encoded, loss_mean = train(model, optimizer, loss_fn, data_iterator)
        print("Mean loss: {0}, epoch {1}".format(loss_mean, epoch))
        #with torch.no_grad():
        #    _, _, test_metrics = evaluate(model, loss_fn, data_iterator, metrics)
        
        #acc_epoch = test_metrics['accuracy']
        #is_best = acc_epoch < best_eval_acc
        is_best = loss_mean < 0.001

        if(is_best or epoch == (num_epochs - 1)):

            with open(experiment_folder + '/TRencoded_vect.csv', 'w') as f:
                wr = csv.writer(f, delimiter=',')
                for e in encoded:
                    wr.writerow(e)

            with open(experiment_folder + '/TRmrns.csv', 'w') as f:
                wr = csv.writer(f, delimiter=',')
                for m in mrn:
                    wr.writerow([m])
            
            with open(experiment_folder + '/TRlabels.csv', 'w') as f:
                 wr = csv.writer(f, delimiter=',')
                 for l in lab:
                     wr.writerow([l])

            with open(experiment_folder + '/TRmetrics.txt', 'w') as f:
                wr = csv.writer(f, delimiter = '\t')
                #for m, v in metrics_average.items():
                #    wr.writerow([m, v])
                wr.writerow(["Mean loss:", loss_mean])  
            
            #utils.save_best_model({'epoch':epoch,
            #                       'state_dict':model.state_dict(),
            #                       'optim_dict':optimizer.state_dict(),
            #                       #'best_acc':acc_epoch
            #                      },
            #                      folder=model_dir)  
            print("-- Found new best  at epoch {0}".format(epoch))
            print("Evaluating the model...")
            lab, mrn, enoded, test_metrics = evaluate(model, loss_fn, data_iterator, metrics, best_eval=True)

                #acc_epoch = test_metrics['accuracy']
                #is_best = acc_epoch < best_eval_acc
            return lab, mrn, encoded, test_metrics
                #best_eval_acc = acc_epoch

