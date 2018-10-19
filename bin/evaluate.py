import torch
import numpy as np
import model.net as net 

def evaluate(model, loss_fn, data_iterator, metrics, best_eval=False)
    model.eval()

    summ = []
    encoded_list = []
    mrn_list = []
    for idx, (epoch, mrn) in enumerate(data_iterator):
        out, encoded = model(batch)
        loss = loss_fn(out, batch)

        out.cpu()
        encoded.cpu()

        summary_batch = {metric:metrics[metric](out, batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)
        
        if best_eval:
            encoded_list += encoded.tolist()
            mrn_list.extend([m in mrn])

    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = "--".join("{}: {:05.3f}".format(k,v) for k,v in metrics_mean.items())
    
    if best_eval:
        return mrn_list, encoded_list, metrics_mean
    else:
        return metrics_mean               
