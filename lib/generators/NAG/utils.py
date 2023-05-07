import torch
import numpy as np
from numpy.random import shuffle as shffle
from torch.utils.data import Dataset
import copy


INPUT0 = 'input0'
INPUT1 = 'input1'
OUTPUT = 'output'
SEQCONV3X3 = 'seqconv3x3'
SEQCONV5X5 = 'seqconv5x5'
AVGPOOL3X3 = 'avgpool3x3'
MAXPOOL3X3 = 'maxpool3x3'
IDENTITY = 'identity'
CHOICES = [OUTPUT, INPUT0, INPUT1, SEQCONV3X3, SEQCONV5X5, AVGPOOL3X3, MAXPOOL3X3, IDENTITY]
CODE = {x: i for i, x in enumerate(CHOICES)}
CODE_ = {i: x  for i, x in enumerate(CHOICES)}


class NASBenchwithLabel(Dataset):
    def __init__(self, data, stop, num_perf_class, num_params_class):
        super(NASBenchwithLabel, self).__init__()
        self.data = copy.deepcopy(data)
        perf_list = []
        params_list = []
        for x in self.data:
            perf_list.append(x['perf'].item())
            params_list.append(x['params'].item())
        perf_list = sorted(perf_list)
        params_list = sorted(params_list)
        splits_perf = [perf_list[len(perf_list)//num_perf_class*i] for i in range(1, num_perf_class)]
        splits_params = [params_list[len(params_list)//num_params_class*i] for i in range(1, num_params_class)]
        for x in self.data:
            conv_n = x['conv_n'].item()
            reduc_n = x['reduc_n'].item()
            conv_edge_ = torch.zeros([stop, stop], dtype=torch.float)
            conv_edge_[:conv_n, :conv_n] = x['conv_edge']
            x['conv_edge'] = conv_edge_
            reduc_edge_ = torch.zeros([stop, stop], dtype=torch.float)
            reduc_edge_[:reduc_n, :reduc_n] = x['reduc_edge']
            x['reduc_edge'] = reduc_edge_
            conv_node_ = torch.zeros([stop, len(CODE)], dtype=torch.float)
            conv_node_[:conv_n, :] = x['conv_node']
            x['conv_node'] = conv_node_
            reduc_node_ = torch.zeros([stop, len(CODE)], dtype=torch.float)
            reduc_node_[:reduc_n, :] = x['reduc_node']
            x['reduc_node'] = reduc_node_
            for i, split in enumerate(splits_perf):
                if x['perf'].item() <= split:
                    x['label_perf'] = torch.LongTensor([i])
                    break
            else:
                x['label_perf'] = torch.LongTensor([num_perf_class-1])
            for i, split in enumerate(splits_params):
                if x['params'].item() <= split:
                    x['label_params'] = torch.LongTensor([i])
                    break
            else:
                x['label_params'] = torch.LongTensor([num_params_class-1])

    def __getitem__(self, index):
        return (self.data[index]['conv_edge'], self.data[index]['conv_node'], self.data[index]['conv_n'], self.data[index]['reduc_edge'], self.data[index]['reduc_node'], self.data[index]['reduc_n'], self.data[index]['label_perf'], self.data[index]['label_params'])

    def __len__(self):
        return len(self.data)


def get_graphs(edge_logits, node_logits, stop):
    batch_size = node_logits.size(0)
    step = node_logits.size(1)
    vocab_size = node_logits.size(2)
    assert batch_size == edge_logits.size(0), 'The batch_size of edges and nodes are different!'
    assert sum(range(step)) == edge_logits.size(1), 'The step of edges and nodes are different!'
    edges = torch.FloatTensor([]).cuda()
    nodes = torch.FloatTensor([]).cuda()
    ns = torch.LongTensor([]).cuda()
    for batch in range(batch_size):
        edge = torch.zeros([stop, stop]).float().cuda()
        node = torch.zeros([stop, vocab_size]).float().cuda()
        for i in range(step):
            edge[i, :i] = edge_logits[batch][sum(range(i)):sum(range(i+1))]
            node[i, :] = node_logits[batch][i, :]
            if node_logits[batch][i].topk(1)[1].item() == CODE['output']:
                break
        n = torch.LongTensor([i+1]).cuda()
        edges = torch.cat([edges, edge.unsqueeze(0)], dim=0)
        nodes = torch.cat([nodes, node.unsqueeze(0)], dim=0)
        ns    = torch.cat([ns,    n.unsqueeze(0)],    dim=0)
    return edges, nodes, ns


def edges2index(edges, finish=False):
    batch_size, size = edges.size()
    edge_index = torch.LongTensor(2, 0).cuda()
    num_nodes = int(np.sqrt(2*size+1/4)+.5)
    for idx, batch_edge in enumerate(edges):
        trans = idx*num_nodes
        if finish:
            trans = 0
        i = np.inf
        j = 0
        for k, edge in enumerate(batch_edge):
            if j >= i:
                j -= i
            if edge.item() == 0:
                j += 1
                i = int(np.ceil((np.sqrt(2*k+9/4)-.5)))
            else:
                i = int(np.ceil((np.sqrt(2*k+9/4)-.5)))
                edge_index = torch.cat([edge_index, torch.LongTensor([[j+trans], [i+trans]]).cuda()], 1)
                j += 1
    return edge_index


def graph2arch(edges, nodes, ns):
    edges = edges.cpu().numpy().transpose(0, 2, 1)
    nodes = nodes.cpu().numpy()
    ns    = ns.cpu().numpy()
    num_graph = len(ns)
    archs = list()
    for i in range(num_graph):
        n    = ns[i][0]
        edge = (edges[i][:n, :n]>0.5).astype(int).tolist()
        node = [CODE_[np.argmax(nodes[i][idx])] for idx in range(n)]
        archs.append((edge, node))
    return archs


def sample_random(graph_list, ratio): 
    sampled_list = list()
    shffle(graph_list)
    length = len(graph_list)
    cut = int(np.round(length*ratio))
    sampled_list += graph_list[:cut]
    return sampled_list


class ModelSpec(object):
  """Model specification given adjacency matrix and labeling."""

  def __init__(self, matrix, ops, data_format='channels_last'):
    """Initialize the module spec.

    Args:
      matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
      ops: V-length list of labels for the base ops used. The first and last
        elements are ignored because they are the input and output vertices
        which have no operations. The elements are retained to keep consistent
        indexing.
      data_format: channels_last or channels_first.

    Raises:
      ValueError: invalid matrix or ops
    """
    if not isinstance(matrix, np.ndarray):
      matrix = np.array(matrix)
    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
      raise ValueError('matrix must be square')
    if shape[0] != len(ops):
      raise ValueError('length of ops must match matrix dimensions')
    if not is_upper_triangular(matrix):
      raise ValueError('matrix must be upper triangular')

    # Both the original and pruned matrices are deep copies of the matrix and
    # ops so any changes to those after initialization are not recognized by the
    # spec.
    self.original_matrix = copy.deepcopy(matrix)
    self.original_ops = copy.deepcopy(ops)

    self.matrix = copy.deepcopy(matrix)
    self.ops = copy.deepcopy(ops)
    self.valid_spec = True
    self._prune()

    self.data_format = data_format

  def _prune(self):
    """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(self.original_matrix)[0]

    # DFS forward from input
    visited_from_input = set([0, 1])
    frontier = [0, 1]
    while frontier:
      top = frontier.pop()
      for v in range(top + 1, num_vertices):
        if self.original_matrix[top, v] and v not in visited_from_input:
          visited_from_input.add(v)
          frontier.append(v)

    # DFS backward from output
    visited_from_output = set([num_vertices - 1])
    frontier = [num_vertices - 1]
    while frontier:
      top = frontier.pop()
      for v in range(0, top):
        if self.original_matrix[v, top] and v not in visited_from_output:
          visited_from_output.add(v)
          frontier.append(v)

    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
        visited_from_input.intersection(visited_from_output))

    # If the non-extraneous graph is less than 2 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 3:
      self.matrix = None
      self.ops = None
      self.valid_spec = False
      return

    self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
    self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
      del self.ops[index]

  def __str__(self):
    return f'matrix: {self.matrix}, ops: {self.ops}'


def is_upper_triangular(matrix):
  """True if matrix is 0 on diagonal and below."""
  for src in range(np.shape(matrix)[0]):
    for dst in range(0, src + 1):
      if matrix[src, dst] != 0:
        return False
  return True
