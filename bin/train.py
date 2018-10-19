import torch
import utils
from utils import model_pars
import numpy as np

def train(model, optimizer, loss_fn, data_iterator):
    model.train()
   
    loss_batch = []
    for idx, (batch, mrn) in enumerate(data_iterator):
        #batch = batch.cuda()
            
        optimizer.zero_grad()
        out, encoded_vect = model(batch)
        loss = loss_fn(out, batch)
        loss.backward()
        optimizer.step()
        loss_batch.append(loss.item())
        
        print("Batch: {}".format(idx))

    loss_mean = np.mean(loss_batch)

def train_and_evaluate(model, data_iterator, loss_fn, metrics, optimizer, model_dir):
    best_eval_acc = 0.0
    num_epochs = model_pars['num_epochs']
    for epoch in range(num_epochs):
        train(model, optimizer, loss_fn, data_iterator)
        print("Epoch {0} of {1}".format(epoch, num_epochs))
        _, _, test_metrics = evaluate(model, loss_fn, data_iterator, metrics)
        
        acc_epoch = test_metrics['accuracy']
        is_best = acc_epoch > beast_val_acc

        utils.save_best_model({'epoch':epoch,
                               'state_dict':model.state_dict(),
                               'optim_dict':optimizer.state_dict(),
                               'best_acc':acc_epoch},
                              is_best=is_best,
                              folder=model_dir)  
        if(is_best):
            print("-- Found new best accuracy")
            best_eval_acc = val_acc

