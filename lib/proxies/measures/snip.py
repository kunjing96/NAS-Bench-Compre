import torch
import torch.nn as nn
import torch.nn.functional as F
import types

from lib.proxies.measures import measure
from lib.proxies.p_utils import get_layer_metric_array, sum_arr


def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

@measure('snip', bn=True, mode='param')
def compute_snip_per_weight(net, device, inputs, targets, mode, loss_fn, split_data=1):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
    
        outputs = net.forward(inputs[st:en])
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def snip(layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad / split_data)
        else:
            return None # torch.zeros_like(layer.weight)
    
    grads_abs = get_layer_metric_array(net, snip, mode)

    return sum_arr(grads_abs)
