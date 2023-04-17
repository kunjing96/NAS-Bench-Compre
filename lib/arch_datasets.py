import os
import torch
import numpy as np
from itertools import chain, repeat
from torch_geometric.data import InMemoryDataset, Data
import copy
from collections.abc import Sequence


class Dataset(InMemoryDataset):

    def __init__(self, search_space=None, archs=None):
        self.search_space = search_space
        self._indices = None
        self.archs = archs
        self.process()

    def process(self):
        self.available_ops = ['input1', 'input2',] + self.search_space.choices + ['output']
        self.max_num_vertices = 11
        archs = self.archs

        def arch2Data(arch):
            x = torch.tensor([self.available_ops.index(x) for x in arch['ops']], dtype=torch.long)
            if 'acc' in arch.keys():
                y = torch.ones_like(x) * arch['acc']
            else:
                y = None
            forward_edges = [[(i, j) for j, x in enumerate(xs) if x > 0] for i, xs in enumerate(arch['adj'])]
            forward_edges = np.array(list(chain(*forward_edges)))
            backward_edges = forward_edges[::-1, ::-1]
            edges = np.concatenate([forward_edges, backward_edges])
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            data = Data(x=x, y=y, edge_index=edge_index)
            return data
        
        self.data = tuple()
        self.slices = tuple()
        for i in range(len(archs[0])):
            acc_list  = []
            data_list = []
            for arch in archs:
                data = arch2Data(arch[i])
                acc_list.append(arch[i]['acc'])
                data_list.append(data)
            top_indices = np.argsort(acc_list)[::-1].tolist()
            data, slices = self.collate(data_list)
            self.data = (*self.data, data)
            self.slices = (*self.slices, slices)
        self.top_indices = top_indices

    def len(self):
        for item in self.slices[0].values():
            return len(item) - 1
        return 0

    def __getitem__(self, idx):
        if (isinstance(idx, (int, np.integer)) or (isinstance(idx, torch.Tensor) and idx.dim() == 0) or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
            data = self.get(self.indices()[idx])
            ret_data = tuple()
            for i in range(len(data)):
                _data = data[i]
                ret_data = (*ret_data, _data)
            return ret_data
        else:
            return self.index_select(idx)

    def get(self, idx):
        if hasattr(self, '_data_list'):
            if self._data_list is None:
                self._data_list = self.len() * [None]
            else:
                return copy.copy(self._data_list[idx])
        data = tuple()
        for i in range(len(self.data)):
            _data = self.data[i].__class__()
            if hasattr(self.data[i], '__num_nodes__'):
                _data.num_nodes = self.data[i].__num_nodes__[idx]
            for key in self.data[i].keys:
                item, slices = self.data[i][key], self.slices[i][key]
                start, end = slices[idx].item(), slices[idx + 1].item()
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    cat_dim = self.data[i].__cat_dim__(key, item)
                    if cat_dim is None:
                        cat_dim = 0
                    s[cat_dim] = slice(start, end)
                elif start + 1 == end:
                    s = slices[start]
                else:
                    s = slice(start, end)
                _data[key] = item[s]
            data = (*data, _data)
        if hasattr(self, '_data_list'):
            self._data_list[idx] = copy.copy(data)
        return data
    
    def index_select(self, idx):
        indices = self.indices()
        if isinstance(idx, slice):
            indices = indices[idx]
        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())
        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())
        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())
        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())
        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]
        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def __repr__(self):
        return '{}(space={})'.format(self.__class__.__name__, self.search_space.__class__.__name__)
