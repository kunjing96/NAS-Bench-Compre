import math
import torch
import numpy as np


class BPRLoss(torch.nn.Module):

    def __init__(self, exp_weighted=False):
        super(BPRLoss, self).__init__()
        self.exp_weighted = exp_weighted

    def forward(self, input, target):
        N = input.size(0)
        total_loss = 0
        for i in range(N):
            indices = (target>target[i])
            x = torch.log(1 + torch.exp(-(input[indices] - input[i])))
            if self.exp_weighted:
                x = (torch.exp(target[i]) - 1) * (torch.exp(target[indices]) - 1) * x
            else:
                x = x
            total_loss += torch.sum(x)
        if self.exp_weighted:
            return 2 / (N * (math.e - 1))**2 * total_loss
        else:
            return 2 / N**2                  * total_loss


def calculate_fisher(loader, predictor, criterion, reserve_p):
    '''
    Calculate Fisher Information for different parameters
    '''
    gradient_mask = dict()
    predictor.train()

    for params in predictor.encoder.parameters():
        gradient_mask[params] = params.new_zeros(params.size())

    N = len(loader)
    for _, batch in enumerate(loader):
        batch = batch[0]
        x            = batch.x.cuda(non_blocking=True)
        edge_index_x = batch.edge_index.cuda(non_blocking=True)
        ptr_x        = batch.ptr.cuda(non_blocking=True)
        target       = batch.y.cuda(non_blocking=True)[ptr_x[:-1]]

        output = predictor(x, edge_index_x, ptr_x)
        loss = criterion(output.squeeze(), target.squeeze())
        loss.backward()

        for params in predictor.encoder.parameters():
            if params.grad is not None:
                torch.nn.utils.clip_grad_norm_(params, 5)
                gradient_mask[params] += (params.grad ** 2) / N
        predictor.encoder.zero_grad()

    # Numpy
    r = None
    for k, v in gradient_mask.items():
        v = v.view(-1).cpu().numpy()
        if r is None:
            r = v
        else:
            r = np.append(r, v)
    polar = np.percentile(r, (1-reserve_p)*100)
    for k in gradient_mask:
        gradient_mask[k] = gradient_mask[k] >= polar

    return gradient_mask