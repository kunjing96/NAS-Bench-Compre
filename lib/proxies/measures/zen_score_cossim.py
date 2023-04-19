import torch
import torch.nn.functional as F

from lib.proxies.measures import measure
from lib.proxies.p_utils import AvgrageMeter


@measure('zen_score_cossim', bn=True, mode='param')
def compute_zen_score_cossim(net, device, inputs, targets, mode, loss_fn, split_data=1, skip_grad=False):
    alpha = 0.01
    zen_score_cossim = AvgrageMeter()
    net.zero_grad()
    N = inputs.shape[0]
    with torch.no_grad():
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            epsilon = torch.zeros_like(inputs[st:en]).normal_()
            epsilon_ = torch.zeros_like(inputs[st:en]).normal_()
            out1 = net.forward(epsilon, pre_GAP=True)
            if isinstance(out1, tuple):
                out1 = out1[0]
            out1 = out1.reshape(en - st, -1)
            out2 = net.forward(epsilon + alpha*epsilon_, pre_GAP=True)
            if isinstance(out2, tuple):
                out2 = out2[0]
            out2 = out2.reshape(en - st, -1)
            cos_sim = F.cosine_similarity(out1, out2).abs_()
            zen_score_cossim.update(torch.sum(cos_sim).item(), en - st)

    return zen_score_cossim.avg
