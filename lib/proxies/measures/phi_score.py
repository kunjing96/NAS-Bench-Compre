import torch

from lib.proxies.measures import measure
from lib.proxies.p_utils import AvgrageMeter


@measure('phi_score', bn=False, mode='param')
@measure('phi_score_bn', bn=True, mode='param')
def compute_phi_score(net, device, inputs, targets, mode, loss_fn, split_data=1, skip_grad=False):
    alpha = 0.01
    p = 1
    phi_score = AvgrageMeter()
    net.zero_grad()
    N = inputs.shape[0]
    with torch.no_grad():
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            epsilon = torch.zeros_like(inputs[st:en]).normal_()
            epsilon_ = torch.zeros_like(inputs[st:en]).normal_()
            outputs1 = net.forward(epsilon + alpha*epsilon_, pre_GAP=True)
            if isinstance(outputs1, tuple):
                outputs1 = outputs1[0]
            outputs2 = net.forward(epsilon, pre_GAP=True)
            if isinstance(outputs2, tuple):
                outputs2 = outputs2[0]
            phi_score.update((outputs1 - outputs2).norm(p=p).detach().cpu().item(), en - st)

    return phi_score.avg
