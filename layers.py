from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from .utils import *
import time

class GAT_gate(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        nhop: int = 1,
        gpu: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.nhop = nhop
        self.gpu = gpu

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_node = Parameter(torch.Tensor(1, heads, out_channels, out_channels))
        self.att_gate = Parameter(torch.Tensor(1, heads, out_channels*2))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_node)
        glorot(self.att_edge)
        glorot(self.att_gate)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha = (x_src, x_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, origin_x=x, alpha=alpha, size=size)
        
        for _ in range(self.nhop-1):
            out = self.propagate(edge_index, x=out, origin_x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        
        if alpha_i is None:
            alpha = alpha_j
        else:
            # num_edges, num_heads, num_channels
            alpha = ((torch.unsqueeze(alpha_j, dim=-1) * self.att_node).sum(dim=-1) * alpha_i).sum(dim=-1) + \
                    ((torch.unsqueeze(alpha_i, dim=-1) * self.att_node).sum(dim=-1) * alpha_j).sum(dim=-1)
        
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha = alpha * (edge_attr * self.att_edge).sum(dim=-1)
        
        # if edge_attr is not None:
        #     alpha = alpha * edge_attr    

        return alpha


    def message(self, x_j: Tensor, origin_x_j: Tensor, alpha: Tensor) -> Tensor:
        ret = alpha.unsqueeze(-1) * x_j
        ret = ret.relu()
        coeff = (torch.cat([origin_x_j, ret], dim=-1) * self.att_gate).sum(dim=-1, keepdim=True)
        coeff = torch.sigmoid(coeff)
        ret = coeff * ret + (1-coeff) * origin_x_j
        return ret

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
        
# class GAT_gate(torch.nn.Module):
#     def __init__(self, n_in_feature, n_out_feature, nhop, gpu = 0):
#         super(GAT_gate, self).__init__()
#         self.W = nn.Linear(n_in_feature, n_out_feature)
#         #self.A = nn.Parameter(torch.Tensor(n_out_feature, n_out_feature))
#         self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
#         self.gate = nn.Linear(n_out_feature*2, 1)
#         self.leakyrelu = nn.LeakyReLU(0.2)
#         self.zeros = torch.zeros(1)
#         self.nhop = nhop
#         if gpu > 0:
#             self.zeros = self.zeros.cuda()

#     def forward(self, x, adj, get_attention=False):
#         h = self.W(x)
#         # batch_size = h.size()[0]
#         N = h.size()[1]
#         e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h,self.A), h))
#         e = e + e.permute((0,2,1))
#         # zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, self.zeros)
#         attention = F.softmax(attention, dim=1)
#         #attention = F.dropout(attention, self.dropout, training=self.training)
#         #h_prime = torch.matmul(attention, h)
#         attention = attention*adj
#         # h_prime = F.relu(torch.einsum('aij,ajk->aik',(attention, h)))
#         # coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1))
#         # retval = coeff*x+(1-coeff)*h_prime

#         retval = h
#         for _ in range(self.nhop):
#             az = F.relu(torch.einsum('aij,ajk->aik',(attention, retval)))
#             coeff = torch.sigmoid(self.gate(torch.cat([h, az], -1))).repeat(1, 1, h.size(-1))
#             retval = coeff * h + (1 - coeff) * az
        
#         if not get_attention:
#             return retval

#         return retval, attention
