from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import torch
import torch.nn as nn

from lib.nets.NAG.base_ops import *
from lib.nets.NAG.utils import count_parameters


class NASWSNetworkCIFAR(nn.Module):
    def __init__(self, num_labels, layers, max_num_vertices, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps):
        super(NASWSNetworkCIFAR, self).__init__()

        self.max_num_vertices = max_num_vertices
        self.num_labels = num_labels
        self.layers = layers
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps

        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.total_layers = self.layers * 3 + 2
        
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
        for i in range(self.total_layers):
            if i not in self.pool_layers:
                cell = Cell(outs, self.max_num_vertices, channels, False, i, self.total_layers, self.steps, self.drop_path_keep_prob)
            else:
                channels *= 2
                cell = Cell(outs, self.max_num_vertices, channels, True, i, self.total_layers, self.steps, self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
            
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadCIFAR(outs[-1][-1], self.num_labels)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], self.num_labels)

        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, x, spec, step=None, bn_train=False):
        aux_logits = None
        conv_spec, reduc_spec = spec[0], spec[1]
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                assert i in self.pool_layers
                s0, s1 = s1, cell(s0, s1, reduc_spec, step, bn_train=bn_train)
            else:
                s0, s1 = s1, cell(s0, s1, conv_spec, step, bn_train=bn_train)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1, bn_train=bn_train)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits

    def get_param_size(self, spec):
        param_size = 0
        conv_spec, reduc_spec = spec[0], spec[1]
        param_size += count_parameters(self.stem)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                assert i in self.pool_layers
                param_size += cell.get_param_size(reduc_spec)
            else:
                param_size += cell.get_param_size(conv_spec)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                param_size += count_parameters(self.auxiliary_head)
        param_size += count_parameters(self.global_pooling)
        param_size += count_parameters(self.dropout)
        param_size += count_parameters(self.classifier)
        return param_size


class Cell(nn.Module):
    """
    Builds the model using the adjacency matrix and op labels specified. Channels
    controls the module output channel count but the interior channels are
    determined via equally splitting the channel count whenever there is a
    concatenation of Tensors.
    """
    def __init__(self, prev_layers, max_num_vertices, channels, reduction, layer_id, layers, steps, drop_path_keep_prob=None):
        super(Cell, self).__init__()
        assert len(prev_layers) == 2
        self.reduction = reduction
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.max_num_vertices = max_num_vertices

        # maybe calibrate size
        prev_layers = [list(prev_layers[0]), list(prev_layers[1])]
        self.maybe_calibrate_size = MaybeCalibrateSize(prev_layers, channels)
        prev_layers = self.maybe_calibrate_size.out_shape

        stride = 2 if self.reduction else 1
        # operation for each vertex
        self.input_op0 = nn.ModuleList([None, None])
        self.input_op1 = nn.ModuleList([None, None])
        # self.vertex_sums = nn.ModuleList([None, None])
        self.vertex_ops = nn.ModuleList([None, None])
        for t in range(2, self.max_num_vertices-1):
            if reduction:
                self.input_op0.append(FactorizedReduce(prev_layers[0][-1], channels))
                self.input_op1.append(FactorizedReduce(prev_layers[1][-1], channels))
            # self.vertex_sums.append(WSReLUConvBN(t, channels, channels, 1))
            op = Node(prev_layers, channels, stride, drop_path_keep_prob, t, layer_id, layers, steps)
            self.vertex_ops.append(op)
            prev_layers.append(op.out_shape)
        out_hw = min([shape[0] for i, shape in enumerate(prev_layers)])

        if reduction:
            self.fac_0 = FactorizedReduce(prev_layers[0][-1], channels)
            self.fac_1 = FactorizedReduce(prev_layers[1][-1], channels)
        self.final_combine_conv = WSReLUConvBN(self.max_num_vertices-1, channels, channels, 1)

        self.out_shape = [out_hw, out_hw, channels]

    def forward(self, s0, s1, spec, step, bn_train=False):
        num_vertices = np.shape(spec.matrix)[0]
        s0, s1 = self.maybe_calibrate_size(s0, s1, bn_train=bn_train)
        states = [s0, s1]
        for t in range(2, num_vertices-1):
            s = copy.copy(states)
            pres = [src for src in range(t) if spec.matrix[src, t]]
            if self.reduction:
                if (sum([src in [0, 1] for src in pres]))!=0 and (sum([src not in [0, 1] for src in pres]))!=0:
                    if 0 in pres:
                        s[0] = self.input_op0[t](states[0])
                    if 1 in pres:
                        s[1] = self.input_op1[t](states[1])
            vertex_input = sum([s[i] for i in pres])
            # vertex_input = self.vertex_sums[t](vertex_input, pres, bn_train=bn_train)
            vertex_output = self.vertex_ops[t](vertex_input, spec.ops[t], step, pres, bn_train=bn_train)
            states.append(vertex_output)
        concat = [src for src in range(num_vertices-1) if spec.matrix[src, num_vertices-1]]
        if self.reduction:
            if 0 in concat:
                states[0] = self.fac_0(states[0])
            if 1 in concat:
                states[1] = self.fac_1(states[1])
        out = torch.cat([states[i] for i in concat], dim=1)
        return self.final_combine_conv(out, concat, bn_train=bn_train)

    def get_param_size(self, spec):
        param_size = 0
        num_vertices = np.shape(spec.matrix)[0]
        param_size += count_parameters(self.maybe_calibrate_size)
        for t in range(2, num_vertices-1):
            pres = [src for src in range(t) if spec.matrix[src, t]]
            if self.reduction:
                if (sum([src in [0, 1] for src in pres]))!=0 and (sum([src not in [0, 1] for src in pres]))!=0:
                    if 0 in pres:
                        param_size += count_parameters(self.input_op0)
                    if 1 in pres:
                        param_size += count_parameters(self.input_op1)
            param_size += self.vertex_ops[t].get_param_size(spec.ops[t], pres)
        concat = [src for src in range(num_vertices-1) if spec.matrix[src, num_vertices-1]]
        if self.reduction:
            if 0 in concat:
                param_size += count_parameters(self.fac_0)
            if 1 in concat:
                param_size += count_parameters(self.fac_1)
        param_size += self.final_combine_conv.get_param_size(concat)
        return param_size


class Node(nn.Module):
    def __init__(self, prev_layers, channels, stride, drop_path_keep_prob=None, node_id=0, layer_id=0, layers=0, steps=0):
        super(Node, self).__init__()
        self.channels = channels
        self.stride = stride
        self.drop_path_keep_prob = drop_path_keep_prob
        self.node_id = node_id
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.op = nn.ModuleList()

        # avg_pool
        self.avg_pool = WSAvgPool2d(3, None, padding=1)
        # max_pool
        self.max_pool = WSMaxPool2d(3, None, padding=1)
        # sep_conv
        self.sep_conv_3 = WSSepConv(channels, channels, 3, None, 1)
        self.sep_conv_5 = WSSepConv(channels, channels, 5, None, 2)
        if self.stride > 1:
            assert self.stride == 2
            assert prev_layers[0][-1] == prev_layers[1][-1]
            self.id_reduce = FactorizedReduce(prev_layers[0][-1], channels)

        self.out_shape = [prev_layers[0][0]//stride, prev_layers[0][1]//stride, channels]
        
    def forward(self, x, op, step, pres, bn_train=False):
        stride = self.stride if sum([src not in [0, 1] for src in pres])==0 else 1
        out = None
        if op == 'seqconv3x3':
            out = self.sep_conv_3(x, stride, bn_train=bn_train)
        elif op == 'seqconv5x5':
            out = self.sep_conv_5(x, stride, bn_train=bn_train)
        elif op == 'avgpool3x3':
            out = self.avg_pool(x, stride)
        elif op == 'maxpool3x3':
            out = self.max_pool(x, stride)
        else:
            assert op == 'identity'
            if stride > 1:
                assert stride == 2
                out = self.id_reduce(x, bn_train=bn_train)
            else:
                assert stride == 1
                out = x

        if op != 'identity' and self.drop_path_keep_prob is not None and self.training:
            out = apply_drop_path(out, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)

        return out

    def get_param_size(self, op, pres):
        param_size = 0
        stride = self.stride if sum([src not in [0, 1] for src in pres])==0 else 1
        if op == 'seqconv3x3':
            param_size += count_parameters(self.sep_conv_3)
        elif op == 'seqconv5x5':
            param_size += count_parameters(self.sep_conv_5)
        elif op == 'avgpool3x3':
            param_size += count_parameters(self.avg_pool)
        elif op == 'maxpool3x3':
            param_size += count_parameters(self.max_pool)
        else:
            assert op == 'identity'
            if stride > 1:
                assert stride == 2
                param_size += count_parameters(self.id_reduce)
        return param_size
