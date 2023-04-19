import torch, math

from lib.proxies.measures import measure
from lib.proxies.p_utils import AvgrageMeter


@measure('zen_score_with_data', bn=True, mode='param')
def compute_zen_score_with_data(net, device, inputs, targets, mode, loss_fn, split_data=1, skip_grad=False):
    alpha = 0.01
    p = 1
    eps = 1e-6
    zen_score_with_data = AvgrageMeter()
    log_mean_std_pre_layer = []
    net.zero_grad()
    N = inputs.shape[0]
    with torch.no_grad():
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            epsilon_ = torch.zeros_like(inputs[st:en]).normal_()
            out1 = net.forward(inputs[st:en] + alpha*epsilon_, pre_GAP=True)
            if isinstance(out1, tuple):
                out1 = out1[0]
            out2 = net.forward(inputs[st:en], pre_GAP=True)
            if isinstance(out2, tuple):
                out2 = out2[0]
            zen_score_with_data.update((out1 - out2).norm(p=p).detach().cpu().item(), en - st)

    for layer in net.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            log_mean_std_pre_layer.append(math.log(layer.running_var.mean().sqrt().detach().cpu().item() + eps))

    return math.log(zen_score_with_data.avg + eps) + sum(log_mean_std_pre_layer)
