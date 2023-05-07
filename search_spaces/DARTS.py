import torch.nn as nn
import numpy as np

from search_spaces import _register
from search_spaces.Base import Base
from lib.nets.DARTS.genotypes import PRIMITIVES, Genotype
from lib.nets.DARTS.search_net import SearchCNNController
from lib.nets.DARTS.net import Network


@_register
class DARTS(Base):

    def __init__(self, config):
        super(DARTS, self).__init__(config)
        self.choices = PRIMITIVES
        net_crit = nn.CrossEntropyLoss()
        self.search_model = SearchCNNController(3, 16, 10, 8, net_crit)
        self.model_cls = Network

    def decode(self, arch):
        decoded_arch = []
        for arch_list, concat in [(arch.normal, arch.normal_concat), (arch.reduce, arch.reduce_concat)]:
            num_ops = len(arch_list) * 2 + 3
            adj = np.zeros((num_ops, num_ops), dtype=np.uint8)
            ops = ['input1', 'input2', 'output']
            node_lists = [[0], [1] , [2, 3], [4, 5], [6, 7], [8, 9], [10]]
            for nodex2 in arch_list:
                for node in nodex2:
                    node_idx = len(ops) - 1
                    adj[node_lists[node[1]], node_idx] = 1
                    ops.insert(-1, node[0])
            adj[[x for c in concat for x in node_lists[c]], -1] = 1
            cell = {'adj': adj,
                    'ops': ops,}
            decoded_arch.append(cell)
        adj = np.zeros((num_ops*2, num_ops*2), dtype=np.uint8)
        adj[:num_ops, :num_ops] = decoded_arch[0]['adj']
        adj[num_ops:, num_ops:] = decoded_arch[1]['adj']
        ops = decoded_arch[0]['ops'] + decoded_arch[1]['ops']
        decoded_arch = {'adj': adj, 'ops': ops,}
        return decoded_arch
    
    def sample(self):
        geno = []
        for _ in range(2):
            cell = []
            for i in range(4):
                ops_normal = np.random.choice(self.choices, 2)
                nodes_in_normal = sorted(np.random.choice(range(i+2), 2, replace=False))
                cell.append([(ops_normal[0], nodes_in_normal[0]), (ops_normal[1], nodes_in_normal[1])])
            geno.append(cell)
        genotype = Genotype(normal=geno[0], normal_concat=[2, 3, 4, 5], reduce=geno[1], reduce_concat=[2, 3, 4, 5])
        return genotype
