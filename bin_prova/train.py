import csv
import torch
import utils
from utils import model_pars
import numpy as np
from evaluate import evaluate

def train(model, optimizer, loss_fn, data_iterator):
    model.train()
    encoded_list = []
    loss_batch = []
    mrn_list = []
    for idx, (batch, mrn) in enumerate(data_iterator):
        batch = batch.cuda()
            
        optimizer.zero_grad()
        out, encoded_vect = model(batch)
        loss = loss_fn(out, batch)
        loss.backward()
        optimizer.step()
        loss_batch.append(loss.item())
        
        encoded_list.append(np.mean(encoded_vect.tolist(), axis=0).tolist())
        mrn_list.append(mrn)
        #print("Batch: {}".format(idx))

    loss_mean = np.mean(loss_batch)

    return mrn_list, encoded_list, loss_mean

def train_and_evaluate(model, data_iterator, loss_fn, optimizer, model_dir, metrics, experiment_folder):
    #best_eval_acc = 0.0
    num_epochs = model_pars['num_epochs']
    for epoch in range(num_epochs):
        print("Epoch {0} of {1}".format(epoch, num_epochs))
        mrn, encoded, loss_mean = train(model, optimizer, loss_fn, data_iterator)
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
            utils.save_best_model(model, experiment_folder)
            print("Evaluating the model...")
            mrn, encoded, test_metrics = evaluate(model, loss_fn, data_iterator, metrics, best_eval=True)

                #acc_epoch = test_metrics['accuracy']
                #is_best = acc_epoch < best_eval_acc
            return mrn, encoded, test_metrics
                #best_eval_acc = acc_epoch

