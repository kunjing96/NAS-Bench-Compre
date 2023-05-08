import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing

from lib.generators.NAG.utils import edges2index, CODE, get_graphs
from lib.generators import _register


class GNNLayer(MessagePassing):
    def __init__(self, ndim):
        super(GNNLayer, self).__init__(aggr='add')
        self.msg = nn.Linear(ndim*2, ndim*2)
        self.msg_rev = nn.Linear(ndim*2, ndim*2)
        self.upd = nn.GRUCell(2*ndim, ndim)

    def forward(self, edge_index, h):
        return self.propagate(edge_index, h=h)

    def message(self, h_j, h_i):
        m = torch.cat([h_j, h_i], dim=1)
        m, m_reverse = torch.split(m, m.size(0)//2, 0)
        a = torch.cat([self.msg(m), self.msg_rev(m_reverse)], dim=0)
        return a

    def update(self, aggr_out, h):
        h = self.upd(aggr_out, h)
        return h


class NodeEmbUpd(nn.Module):
    def __init__(self, ndim, num_layers, dropout):
        super(NodeEmbUpd, self).__init__()
        self.dropout = dropout
        self.GNNLayers = nn.ModuleList([GNNLayer(ndim) for _ in range(num_layers)])

    def forward(self, h, edge_index):
        edge_index = torch.cat([edge_index, torch.index_select(edge_index, 0, torch.tensor([1, 0]).cuda())], 1)
        for layer in self.GNNLayers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(edge_index, h)
        return F.normalize(h, 2, dim=-1)


class GraphAggr(nn.Module):
    def __init__(self, ndim, sdim, aggr):
        super(GraphAggr, self).__init__()
        self.sdim = sdim
        self.aggr = aggr
        self.f_m = nn.Linear(ndim, sdim)
        if aggr == 'gsum':
            self.g_m = nn.Linear(ndim, 1)
            self.sigm = nn.Sigmoid()

    def forward(self, h, idx):
        if self.aggr == 'mean':
            h = self.f_m(h).view(-1, idx, self.sdim)
            return F.normalize(torch.mean(h, 1), 2, dim=-1)
        elif self.aggr == 'gsum':
            h_vG = self.f_m(h)
            g_vG = self.sigm(self.g_m(h))
            h = torch.mul(h_vG, g_vG).view(-1, idx, self.sdim)
            return F.normalize(torch.sum(h, 1), 2, dim=-1)


class GraphEmbed(nn.Module):
    def __init__(self, ndim, sdim, num_layers, dropout, aggr):
        super(GraphEmbed, self).__init__()
        self.ndim = ndim
        self.NodeEmb = NodeEmbUpd(ndim, num_layers, dropout)
        self.GraphEmb = GraphAggr(ndim, sdim, aggr)
        self.GraphEmb_init = GraphAggr(ndim, sdim, aggr)

    def forward(self, h, edge_index):
        idx = h.size(1)
        h = h.view(-1, self.ndim)
        if edge_index.size(1) == 0:
            return h.view(-1, idx, self.ndim), self.GraphEmb(h, idx), self.GraphEmb_init(h, idx)
        else:
            h = self.NodeEmb(h, edge_index)
            h_G = self.GraphEmb(h, idx)
            h_G_init = self.GraphEmb_init(h, idx)
            return h.view(-1, idx, self.ndim), h_G, h_G_init


class NodeAdd(nn.Module):
    def __init__(self, latent_dim, sdim, num_node_atts):
        super(NodeAdd, self).__init__()
        self.f_an = nn.Linear(latent_dim+sdim, sdim)
        self.f_an_2 = nn.Linear(sdim, num_node_atts)

    def forward(self, h_G, z):
        s = self.f_an(torch.cat([h_G, z], 1))
        return self.f_an_2(F.leaky_relu(s, negative_slope=0.2))


class NodeInit(nn.Module):
    def __init__(self, latent_dim, ndim, sdim, num_node_atts):
        super(NodeInit, self).__init__()
        self.NodeInits = nn.Embedding(num_node_atts, ndim)
        self.f_init = nn.Linear(ndim+sdim+latent_dim, ndim+sdim)
        self.f_init_2 = nn.Linear(ndim+sdim, ndim)
        self.f_start = nn.Linear(ndim+latent_dim, ndim+sdim)
        self.f_start_2 = nn.Linear(ndim+sdim, ndim)

    def forward(self, h_G_init, node_atts, z):
        e = self.NodeInits(node_atts)
        if isinstance(h_G_init, str):
            h_inp = self.f_start(torch.cat([e, z], 1))
            return self.f_start_2(F.leaky_relu(h_inp, negative_slope=0.2))
        h_v = self.f_init(torch.cat([e, h_G_init, z], 1))
        return F.normalize(self.f_init_2(F.leaky_relu(h_v, negative_slope=0.2)), 2, dim=-1)


class EdgeAdd(nn.Module): 
    def __init__(self, latent_dim, ndim, sdim):
        super(EdgeAdd, self).__init__()
        self.ndim = ndim
        self.f_s_1 = nn.Linear(ndim*2+sdim+latent_dim, ndim+sdim)
        self.f_s_2 = nn.Linear(ndim+sdim, 1)

    def forward(self, h, h_v, h_G, z):
        idx = h.size(1)
        s = self.f_s_1(torch.cat([h.view(-1, self.ndim),
                                  h_v.unsqueeze(1).repeat(1, idx, 1).view(-1, self.ndim),
                                  h_G.repeat(idx, 1),
                                  z.repeat(idx, 1)], dim=1))
        return self.f_s_2(F.leaky_relu(s, negative_slope=0.2)).view(-1, idx)


class StepGenerator(nn.Module):
    def __init__(self, num_node_atts, latent_dim, ndim, sdim, num_layers, dropout, aggr, stop):
        super(StepGenerator, self).__init__()
        self.prop = GraphEmbed(ndim, sdim, num_layers, dropout, aggr)
        self.nodeAdd = NodeAdd(latent_dim, sdim, num_node_atts)
        self.nodeInit = NodeInit(latent_dim, ndim, sdim, num_node_atts)
        self.edgeAdd = EdgeAdd(latent_dim, ndim, sdim)
        self.stop = stop

    def forward(self, h, z, edge_index):
        h, h_G, h_G_init = self.prop(h, edge_index)
        node_logit = self.nodeAdd(h_G, z)
        node_logit[:, CODE['input0']] = -np.inf
        node_logit[:, CODE['input1']] = -np.inf
        node_prob = F.softmax(node_logit, dim=-1)
        node_atts = torch.topk(node_prob, 1)[1].squeeze().long()
        # if node_atts.dim == 0:
        #     node_atts = node_atts.unsqueeze(-1)
        nodes = torch.zeros_like(node_logit).scatter_(1, node_atts.unsqueeze(-1), 1) #- node_logit.data + node_logit
        no_stop = (node_atts != CODE['output'])

        if h.size(1) >= self.stop-1:
            node_atts = CODE['output'] * torch.ones_like(node_atts)
            nodes = torch.zeros_like(node_logit).scatter_(1, node_atts.unsqueeze(-1), 1)
            node_prob = torch.zeros_like(node_logit).scatter_(1, node_atts.unsqueeze(-1), 1)
            no_stop = torch.zeros_like(no_stop)

        h_v = self.nodeInit(h_G_init, node_atts, z)
        edge_logit = self.edgeAdd(h, h_v, h_G, z)
        edge_prob = torch.sigmoid(edge_logit)
        edge_pred = (edge_prob > 0.5).float()
        edges = edge_pred #- edge_logit.data + edge_logit
        h = torch.cat([h, h_v.unsqueeze(1)], 1)

        return h, nodes, edges, no_stop, node_prob, edge_prob


class Generator(nn.Module):
    def __init__(self, opt, z_label_dim):
        super(Generator, self).__init__()
        self.stop = opt.MAXSTEP
        self.vocab_size = opt.VOCABSIZE
        self.stepGenerator = StepGenerator(opt.VOCABSIZE, z_label_dim, opt.GEMBSIZE, opt.GHIDDENSIZE, opt.GNUMLAYERS, opt.GDROPOUT, opt.GAGGR, opt.MAXSTEP)

    def forward(self, z_label):
        batch_size = z_label.size(0)
        input_node0 = CODE['input0']*torch.ones(batch_size, dtype=torch.long).cuda()
        input_node1 = CODE['input1']*torch.ones(batch_size, dtype=torch.long).cuda()
        h0 = self.stepGenerator.nodeInit('start', input_node0, z_label).unsqueeze(1) # Tensor(batch_size, 1, ndim)
        h1 = self.stepGenerator.nodeInit('start', input_node1, z_label).unsqueeze(1) # Tensor(batch_size, 1, ndim)
        h = torch.cat([h0, h1], dim=1) # Tensor(batch_size, 2, ndim)
        edges = torch.zeros(batch_size, 1).float().cuda()
        nodes = torch.cat([torch.zeros(batch_size, self.vocab_size).float().cuda().scatter_(1, input_node0.unsqueeze(-1), 1).unsqueeze(1), torch.zeros(batch_size, self.vocab_size).float().cuda().scatter_(1, input_node1.unsqueeze(-1), 1).unsqueeze(1)], dim=1) # Tensor(batch_size, 2, vocab_size)
        edge_probs = torch.zeros(batch_size, 1).float().cuda()
        node_probs = torch.cat([torch.zeros(batch_size, self.vocab_size).float().cuda().scatter_(1, input_node0.unsqueeze(-1), 1).unsqueeze(1), torch.zeros(batch_size, self.vocab_size).float().cuda().scatter_(1, input_node1.unsqueeze(-1), 1).unsqueeze(1)], dim=1)
        no_stops = torch.ones(batch_size).type(torch.uint8).cuda()
        num_stops = (no_stops == 0).sum().item()
        while num_stops < batch_size:
            edge_index = edges2index(edges) # Tensor(batch_size, 2, -1)
            h, nodes_new, edges_new, no_stop, node_prob_new, edge_prob_new = self.stepGenerator(h, z_label, edge_index) # Tensor(batch_size, i, ndim), Tensor(batch_size, vocab_size), Tensor(batch_size, i-1), Tensor(batch_size)
            edges = torch.cat([edges, edges_new], 1) # Tensor(batch_size, sum(range(i))
            nodes = torch.cat([nodes, nodes_new.unsqueeze(1)], 1) # Tensor(batch_size, i, vocab_size)
            edge_probs = torch.cat([edge_probs, edge_prob_new], 1)
            node_probs = torch.cat([node_probs, node_prob_new.unsqueeze(1)], 1)
            no_stops = torch.mul(no_stops, no_stop) # Tensor(batch_size)
            num_stops = (no_stops == 0).sum().item()
        return get_graphs(edge_probs, node_probs, self.stop) # Tensor(batch_size, sum(range(i)), Tensor(batch_size, i, vocab_size)


@_register
class NAG(nn.Module):
    def __init__(self, opt):
        super(NAG, self).__init__()
        self.perf_label_embedding = nn.Embedding(opt.NUMPERFS, opt.PERFEMBSIZE)
        self.conv_generator = Generator(opt, opt.LATENTDIM+opt.PERFEMBSIZE)
        self.reduc_generator = Generator(opt, opt.LATENTDIM+opt.PERFEMBSIZE)

    def forward(self, z, perf_labels, param_labels=None):
        z_label = torch.cat([z, F.normalize(self.perf_label_embedding(perf_labels), dim=-1)], -1)
        gen_conv_edges, gen_conv_nodes, gen_conv_ns = self.conv_generator(z_label)
        gen_reduc_edges, gen_reduc_nodes, gen_reduc_ns = self.reduc_generator(z_label)
        return gen_conv_edges, gen_conv_nodes, gen_conv_ns, gen_reduc_edges, gen_reduc_nodes, gen_reduc_ns


@_register
class NAGMultiObj(nn.Module):
    def __init__(self, opt):
        super(NAGMultiObj, self).__init__()
        self.perf_label_embedding = nn.Embedding(opt.NUMPERFS, opt.PERFEMBSIZE)
        self.param_label_embedding = nn.Embedding(opt.NUMPARAMS, opt.PARAMEMBSIZE)
        self.conv_generator = Generator(opt, opt.LATENTDIM+opt.PERFEMBSIZE+opt.PARAMEMBSIZE)
        self.reduc_generator = Generator(opt, opt.LATENTDIM+opt.PERFEMBSIZE+opt.PARAMEMBSIZE)

    def forward(self, z, perf_labels, param_labels):
        z_label = torch.cat([z, F.normalize(self.perf_label_embedding(perf_labels), dim=-1), F.normalize(self.param_label_embedding(param_labels), dim=-1)], -1)
        gen_conv_edges, gen_conv_nodes, gen_conv_ns = self.conv_generator(z_label)
        gen_reduc_edges, gen_reduc_nodes, gen_reduc_ns = self.reduc_generator(z_label)
        return gen_conv_edges, gen_conv_nodes, gen_conv_ns, gen_reduc_edges, gen_reduc_nodes, gen_reduc_ns

