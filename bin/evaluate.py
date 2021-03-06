import torch
import numpy as np
#import model.net as net 

def evaluate(model, loss_fn, data_iterator, metrics, best_eval=False):
    model.eval()

    summ = []
    encoded_list = []
    mrn_list = []

    with torch.no_grad():
        for idx, (batch, mrn) in enumerate(data_iterator):
        
            batch = batch.cuda()

            out, encoded = model(batch)
            loss = loss_fn(out, batch)

            out.cpu()
            encoded.cpu()

            summary_batch = {metric:metrics[metric](out, batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
        
            #print("Batch eval: {0}".format(idx))        
            if best_eval:
                encoded_list.append(np.mean(encoded.tolist(), axis=0).tolist())
                mrn_list.append(mrn)

        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = "--".join("{}: {:05.3f}".format(k,v) for k,v in metrics_mean.items())
        print(metrics_string)

        return mrn_list, encoded_list, metrics_mean
