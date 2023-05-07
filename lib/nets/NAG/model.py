from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn as nn

from lib.nets.NAG.base_ops import *
from lib.nets.DARTS import ops


class NASNetworkCIFAR(nn.Module):
    def __init__(self, num_labels, layers, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps, spec):
        super(NASNetworkCIFAR, self).__init__()

        self.num_labels = num_labels
        self.layers = layers
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        self.conv_spec = spec[0]
        self.reduc_spec = spec[1]

        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.layers = self.layers * 3 + 2
        
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1]

        stem_multiplier = 3
        channels = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        outs = [[32, 32, channels],[32, 32, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.layers):
            if i not in self.pool_layers:
                cell = Cell(self.conv_spec, outs, channels, False, i, self.layers, self.steps, self.drop_path_keep_prob)
            else:
                channels *= 2
                cell = Cell(self.reduc_spec, outs, channels, True, i, self.layers, self.steps, self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
            
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadCIFAR(outs[-1][-1], num_labels)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], num_labels)

        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, x, step=None):
        aux_logits = None
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p


class NASNetworkImageNet(nn.Module):
    def __init__(self, num_labels, layers, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps, spec):
        super(NASNetworkImageNet, self).__init__()

        self.num_labels = num_labels
        self.layers = layers
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        self.conv_spec = spec[0]
        self.reduc_spec = spec[1]
        
        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.layers = self.layers * 3 + 2
        
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1]

        channels = self.channels
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        outs = [[56, 56, channels], [28, 28, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.layers):
            if i not in self.pool_layers:
                cell = Cell(self.conv_spec, outs, channels, False, i, self.layers, self.steps, self.drop_path_keep_prob)
            else:
                channels *= 2
                cell = Cell(self.reduc_spec, outs, channels, True, i, self.layers, self.steps, self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
            
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadImageNet(outs[-1][-1], num_labels)
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], num_labels)
        
        self.init_parameters()
    
    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)
    
    def forward(self, input, step=None):
        aux_logits = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1)
        
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p


class Cell(nn.Module):
    """
    Builds the model using the adjacency matrix and op labels specified. Channels
    controls the module output channel count but the interior channels are
    determined via equally splitting the channel count whenever there is a
    concatenation of Tensors.
    """
    def __init__(self, spec, prev_layers, channels, reduction, layer_id, layers, steps, drop_path_keep_prob=None):
        super(Cell, self).__init__()
        assert len(prev_layers) == 2
        self.spec = spec
        self.reduction = reduction
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.num_vertices = np.shape(self.spec.matrix)[0]

        # maybe calibrate size
        prev_layers = [list(prev_layers[0]), list(prev_layers[1])]
        self.maybe_calibrate_size = MaybeCalibrateSize(prev_layers, channels)
        prev_layers = self.maybe_calibrate_size.out_shape

        stride = 2 if self.reduction else 1
        # operation for each vertex
        self.vertex_ops = nn.ModuleList([None, None])
        self.vertex_sums = nn.ModuleList([None, None])
        for t in range(2, self.num_vertices-1):
            shape = min([prev_layers[src] for src in range(t) if spec.matrix[src, t]])
            pres = [src for src in range(t) if spec.matrix[src, t]]
            self.vertex_sums.append(VertexInputSum(prev_layers, shape[0], channels, pres))
            stride_t = stride if sum([src not in [0, 1] for src in range(t) if spec.matrix[src, t]])==0 else 1
            op = Node(spec.ops[t], shape, channels, stride_t, drop_path_keep_prob, layer_id, layers, steps)
            self.vertex_ops.append(op)
            prev_layers.append(op.out_shape)

        self.concat = [src for src in range(self.num_vertices-1) if spec.matrix[src, -1]]
        out_hw = min([shape[0]//2 if self.reduction and i in [0, 1] else shape[0] for i, shape in enumerate(prev_layers) if i in self.concat])
        self.final_combine = FinalCombine(prev_layers, out_hw, channels, self.concat)
        self.out_shape = [out_hw, out_hw, channels * len(self.concat)]

    def forward(self, s0, s1, step):
        s0, s1 = self.maybe_calibrate_size(s0, s1)
        states = [s0, s1]
        for t in range(2, self.num_vertices-1):
            vertex_input = self.vertex_sums[t](states)
            vertex_output = self.vertex_ops[t](vertex_input, step)
            states.append(vertex_output)
        return self.final_combine(states)


class Node(nn.Module):
    def __init__(self, op, shape, channels, stride=1, drop_path_keep_prob=None, layer_id=0, layers=0, steps=0):
        super(Node, self).__init__()
        self.channels = channels
        self.stride = stride
        self.drop_path_keep_prob = drop_path_keep_prob
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.op = op
        self.id_fact_reduce = None
        shape = list(shape)

        if op == 'seqconv3x3':
            self.op = OP_MAP[op](channels, channels, 3, stride, 1)
            shape = [shape[0] // stride, shape[1] // stride, channels]
        elif op == 'seqconv5x5':
            self.op = OP_MAP[op](channels, channels, 5, stride, 2)
            shape = [shape[0] // stride, shape[1] // stride, channels]
        elif op == 'avgpool3x3':
            self.op = OP_MAP[op](3, stride=stride, padding=1, count_include_pad=False)
            shape = [shape[0] // stride, shape[1] // stride, shape[-1]]
        elif op == 'maxpool3x3':
            self.op = OP_MAP[op](3, stride=stride, padding=1)
            shape = [shape[0] // stride, shape[1] // stride, shape[-1]]
        elif op == 'identity':
            self.op = OP_MAP[op]()
            if stride > 1:
                assert stride == 2
                self.id_fact_reduce = FactorizedReduce(shape[-1], channels)
                shape = [shape[0] // stride, shape[1] // stride, channels]
        self.droppath = ops.DropPath_()

        self.out_shape = list(shape)

    def forward(self, x, step):
        out = self.op(x)
        if self.id_fact_reduce is not None:
            out = self.id_fact_reduce(out)
        if self.op != 'identity' and self.drop_path_keep_prob is not None and self.training:
            out = self.droppath(out)
        return out
