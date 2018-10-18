import torch
import numpy as np
import model.net as net 

def evaluate(model, loss_fn, data_iterator, metrics):
    model.eval()

    summ = []
    for idx, (epoch, mrn) in enumerate(data_iterator):
        out, encoded = model(batch)
        loss = loss_fn(out, batch)

        out.cpu()
        encoded.cpu()

        summary_batch = {metric:metrics[metric](out, batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)

    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = "--".join("{}: {:05.3f}".format(k,v) for k,v in metrics_mean.items())
    return metrics_mean               
