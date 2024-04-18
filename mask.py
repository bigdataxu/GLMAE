import torch
import torch.nn as nn
try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
    
from typing import Optional, Tuple

from torch import Tensor
from torch_geometric.utils import to_undirected, sort_edge_index, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor
import numpy as np
def mask_path(edge_index: Tensor, p: float = 0.3, walks_per_node: int = 1,
              walk_length: int = 3, num_nodes: Optional[int] = None,
              start: str = 'node',
              is_sorted: bool = False,
              training: bool = True) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Sample probability has to be between 0 and 1 '
                         f'(got {p}')

    assert start in ['node', 'edge']
    num_edges = edge_index.size(1)
    edge_mask = edge_index.new_ones(num_edges, dtype=torch.bool)
    
    if not training or p == 0.0:
        return edge_index, edge_mask

    if random_walk is None:
        raise ImportError('`dropout_path` requires `torch-cluster`.')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    if not is_sorted:
        edge_index = sort_edge_index(edge_index, num_nodes=num_nodes)

    row, col = edge_index
    if start == 'edge':
        sample_mask = torch.rand(row.size(0), device=edge_index.device) <= p
        start = row[sample_mask].repeat(walks_per_node)
    else:
        start = torch.randperm(num_nodes, device=edge_index.device)[:round(num_nodes*p)].repeat(walks_per_node)
    
    deg = degree(row, num_nodes=num_nodes)
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    n_id, e_id = random_walk(rowptr, col, start, walk_length, 1.0, 1.0)
    e_id = e_id[e_id != -1].view(-1)  # filter illegal edges过滤非法边缘
    edge_mask[e_id] = False

    return edge_index[:, edge_mask], edge_index[:, ~edge_mask]




class MaskPath(nn.Module):
    def __init__(self, p: float = 0.7, 
                 walks_per_node: int = 1,
                 walk_length: int = 3, 
                 start: str = 'node',
                 num_nodes: Optional[int]=None,
                 is_sorted: bool = False,
                 undirected: bool=True,
                 training: bool = True):
        super().__init__()
        self.p = p
        self.walks_per_node = walks_per_node   #行走次数
        self.walk_length = walk_length    #行走长度
        self.start = start
        self.num_nodes = num_nodes   #2708
        self.undirected = undirected
        self.training = training
        self.is_sorted = is_sorted

    def forward(self, edge_index):
        if self.p < 0. or self.p > 1.:
            raise ValueError(f'Sample probability has to be between 0 and 1 '
                             f'(got {self.p}')

        assert self.start in ['node', 'edge']
        num_edges = edge_index.size(1)
        edge_mask = edge_index.new_ones(num_edges, dtype=torch.bool)

        if not self.training or self.p == 0.0:
            return edge_index, edge_mask

        if random_walk is None:
            raise ImportError('`dropout_path` requires `torch-cluster`.')

        num_nodes = maybe_num_nodes(edge_index, self.num_nodes)

        if not self.is_sorted:
            edge_index = sort_edge_index(edge_index, num_nodes=num_nodes)

        row, col = edge_index
        if self.start == 'edge':
            sample_mask = torch.rand(row.size(0), device=edge_index.device) <= self.p
            start = row[sample_mask].repeat(self.walks_per_node)
        else:
            start = torch.randperm(num_nodes, device=edge_index.device)[:round(num_nodes * self.p)].repeat(self.walks_per_node)

        deg = degree(row, num_nodes=num_nodes)
        rowptr = row.new_zeros(num_nodes + 1)
        torch.cumsum(deg, 0, out=rowptr[1:])
        n_id, e_id = random_walk(rowptr, col, start, self.walk_length, 1.0, 1.0)
        e_id = e_id[e_id != -1].view(-1)  # filter illegal edges过滤非法边缘
        edge_mask[e_id] = False
        remaining_edges = edge_index[:, edge_mask]
        masked_edges = edge_index[:, ~edge_mask]
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges

    def extra_repr(self):
        return f"p={self.p}, walks_per_node={self.walks_per_node}, walk_length={self.walk_length}, \n"\
            f"start={self.start}, undirected={self.undirected}"


class MaskEdge(nn.Module):
    def __init__(self, p: float=0.7, undirected: bool=True):
        super().__init__()
        self.p = p
        self.undirected = undirected

    def forward(self, edge_index):
        if self.p < 0. or self.p > 1.:
            raise ValueError(f'Mask probability has to be between 0 and 1 '
                             f'(got {self.p}')
        e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        mask = torch.full_like(e_ids, self.p, dtype=torch.float32)
        mask = torch.bernoulli(mask).to(torch.bool)
        remaining_edges = edge_index[:, ~mask]
        masked_edges = edge_index[:, mask]
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges

    def extra_repr(self):
        return f"p={self.p}, undirected={self.undirected}"

class MaskumEdge(nn.Module):
    def __init__(self, p: float=0.7, num_nodes: Optional[int]=None,):
        super().__init__()
        self.p = p
        self.num_nodes = num_nodes

    def forward(self, edge_index):
        edge_index = edge_index.t()
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index
        else:
            edge_index = edge_index['train']['edge']
        num_edge = len(edge_index)
        index = np.arange(num_edge)
        np.random.shuffle(index)
        mask_num = int(num_edge * self.p)
        pre_index = torch.from_numpy(index[0:-mask_num])
        mask_index = torch.from_numpy(index[-mask_num:])
        edge_index_train = edge_index[pre_index].t()
        edge_index_mask = edge_index[mask_index].t()
        edge_index = to_undirected(edge_index_train)
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        adj = SparseTensor.from_edge_index(edge_index).t()
        return edge_index, adj, edge_index_mask

class MaskdmEdge(nn.Module):
    def __init__(self, p: float=0.7, num_nodes: Optional[int]=None,):
        super().__init__()
        self.p = p
        self.num_nodes = num_nodes

    def forward(self, edge_index):
        if isinstance(edge_index, torch.Tensor):
            edge_index = to_undirected(edge_index.t()).t()
        else:
            edge_index = torch.stack([edge_index['train']['edge'][:, 1], edge_index['train']['edge'][:, 0]], dim=1)
            edge_index = torch.cat([edge_index['train']['edge'], edge_index], dim=0)

        num_edge = len(edge_index)
        index = np.arange(num_edge)
        np.random.shuffle(index)
        mask_num = int(num_edge * self.p)
        pre_index = torch.from_numpy(index[0:-mask_num])
        mask_index = torch.from_numpy(index[-mask_num:])
        edge_index_train = edge_index[pre_index.type(torch.long)].t()
        edge_index_mask = edge_index[mask_index.type(torch.long)]

        edge_index = edge_index_train
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        adj = SparseTensor.from_edge_index(edge_index).t()
        return edge_index, adj, edge_index_mask