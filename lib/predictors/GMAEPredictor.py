import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import dropout_adj

from lib.predictors import _register


class AvgPooling(torch.nn.Module):

    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, x, ptr):
        g = []
        for i in range(ptr.size(0)-1):
            g.append(torch.mean(x[ptr[i]:ptr[i+1]], 0, True))
        return torch.cat(g, 0)


class GraphEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, out_channels, activation, base_model=GCNConv, k=2, dropout=0.0, skip=True, use_bn=True):
        super(GraphEncoder, self).__init__()
        self.base_model = base_model
        self.num_hidden = out_channels

        assert k >= 2
        self.k = k
        self.skip = skip
        self.use_bn = use_bn
        self.activation = activation
        self.dropout = nn.Dropout(p = dropout)
        self.readout = AvgPooling()
        if self.skip:
            self.fc_skip = torch.nn.Linear(embedding_dim, out_channels)
            self.conv = [base_model(embedding_dim, out_channels)]
            if self.use_bn:
                self.bn = [torch.nn.LayerNorm(out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
                if self.use_bn:
                    self.bn.append(torch.nn.LayerNorm(out_channels))
            self.conv = torch.nn.ModuleList(self.conv)
            if self.use_bn:
                self.bn = torch.nn.ModuleList(self.bn)
        else:
            self.conv = [base_model(embedding_dim, 2 * out_channels)]
            if self.use_bn:
                self.bn = [torch.nn.LayerNorm(2 * out_channels)]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
                if self.use_bn:
                    self.bn.append(torch.nn.LayerNorm(2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            if self.use_bn:
                self.bn.append(torch.nn.LayerNorm(out_channels))
            self.conv = torch.nn.ModuleList(self.conv)
            if self.use_bn:
                self.bn = torch.nn.ModuleList(self.bn)

    def forward(self, x, edge_index, ptr):
        if self.skip:
            if self.use_bn:
                h = self.dropout(self.activation(self.bn[0](self.conv[0](x, edge_index))))
            else:
                h = self.dropout(self.activation(self.conv[0](x, edge_index)))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                if self.use_bn:
                    hs.append(self.dropout(self.activation(self.bn[i](self.conv[i](u, edge_index)))))
                else:
                    hs.append(self.dropout(self.activation(self.conv[i](u, edge_index))))
            return hs[-1], self.readout(hs[-1], ptr)
        else:
            for i in range(self.k):
                if self.use_bn:
                    x = self.dropout(self.activation(self.bn[i](self.conv[i](x, edge_index))))
                else:
                    x = self.dropout(self.activation(self.conv[i](x, edge_index)))
            return x, self.readout(x, ptr)


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


class Predictor(torch.nn.Module):
    def __init__(self, graph_dim):
        super(Predictor, self).__init__()
        #self.fc = torch.nn.Linear(graph_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.PReLU(),
            nn.Linear(graph_dim, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.PReLU(),
            nn.Linear(graph_dim, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, z):
        return self.fc(z)


class CLSHead(nn.Module):
    def __init__(self, n_vocab, d_model, dropout, init_weights=None):
        super(CLSHead, self).__init__()
        self.layer_1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_2 = nn.Linear(d_model, n_vocab)
        if init_weights is not None:
            self.layer_2.weight = init_weights

    def forward(self, x):
        x = self.dropout(torch.tanh(self.layer_1(x)))
        return F.log_softmax(self.layer_2(x), dim=-1)


class MAELearning(nn.Module):
    def __init__(self, n_vocab, n_layers, d_model, activation, base_model, graph_encoder_dropout, dropout, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2, projector_layers):
        super(MAELearning, self).__init__()
        self.d_model = d_model
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        self.dropout = dropout
        # Embedding Layer
        self.opEmb = nn.Embedding(n_vocab, d_model)
        #self.dropout_op = nn.Dropout(p=dropout)
        # Graph Encoder
        self.graph_encoder = GraphEncoder(d_model, d_model, activation, base_model=base_model, k=n_layers, dropout=graph_encoder_dropout, skip=True, use_bn=True)
        # cls
        self.cls = CLSHead(n_vocab, d_model, graph_encoder_dropout)
        self.ssl_loss = nn.KLDivLoss()

    def forward(self, x, edge_index_x, ptr_x, y=None, edge_index_y=None, ptr_y=None):
        emb_x = self.opEmb(x)
        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        return h_x, g_x

    def loss(self, x, edge_index_x, ptr_x, y, edge_index_y, ptr_y, lambd=0.0051, batch_size=0, pretrain_target=None):
        emb_x = self.opEmb(x)
        mask = (torch.rand(emb_x.size(0))<self.dropout).to(emb_x.device)
        emb_x[mask] = 0
        edge_index_x = dropout_adj(edge_index_x, p=self.drop_edge_rate_1)[0]
        emb_x = drop_feature(emb_x, self.drop_feature_rate_1)
        h_x, g_x = self.graph_encoder(emb_x, edge_index_x, ptr_x)
        # compute loss
        output = self.cls(h_x)
        target = torch.zeros_like(output).scatter_(1, x.unsqueeze(-1), 1)
        if pretrain_target == 'masked':
            loss = self.ssl_loss(output[mask], target[mask])
        else:
            loss = self.ssl_loss(output, target)
        return loss


@_register
class GMAEPredictor(nn.Module):
    def __init__(self, n_vocab, config):
        n_vocab = n_vocab
        n_layers = config.NUMLAYERS
        d_model = config.MODELDIM
        activation = eval(config.ACT)
        base_model = eval(config.BASEMODEL)
        graph_encoder_dropout = config.GRAPHENCDROPOUT
        dropout = config.DROPOUT
        drop_edge_rate_1 = config.EDGEDROP1
        drop_edge_rate_2 = config.EDGEDROP2
        drop_feature_rate_1 = config.FEATDROP1
        drop_feature_rate_2 = config.FEATDROP2
        projector_layers = config.PROJLAYERS
        super(GMAEPredictor, self).__init__()
        self.encoder = MAELearning(n_vocab, n_layers, d_model, activation, base_model, graph_encoder_dropout, dropout, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2, projector_layers)
        self.predictor = Predictor(d_model)

    def forward(self, x, edge_index_x, ptr_x, fixed_encoder=False):
        z, g = self.encoder(x, edge_index_x, ptr_x)
        if fixed_encoder:
            z = z.detach()
            g = g.detach()
        output = self.predictor(g)
        return output
