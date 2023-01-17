import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import GATConv, GATv2Conv, NNConv
from .utils import N_atom_features
from .layers import GAT_gate

class gnn(torch.nn.Module):
    def __init__(self, args):
        super(gnn, self).__init__()
        n_graph_layer = args.n_graph_layer
        d_graph_layer = args.d_graph_layer
        n_FC_layer = args.n_FC_layer
        d_FC_layer = args.d_FC_layer
        self.dropout_rate = args.dropout_rate 
        cal_nhop = None

        if args.tatic == "static":
            cal_nhop = lambda x: args.nhop
        elif args.tatic == "cont":
            cal_nhop = lambda x: x + 1
        elif args.tatic == "jump":
            cal_nhop = lambda x: 2 * x + 1

        self.layers1 = [d_graph_layer for i in range(n_graph_layer+1)]
        self.gconv1 = nn.ModuleList([
                                GAT_gate(in_channels=self.layers1[i], out_channels=self.layers1[i+1], 
                                         heads=args.n_att_heads, dropout=self.dropout_rate,
                                         nhop=cal_nhop(i), concat=False, add_self_loops=False,
                                         edge_dim=1, gpu=args.ngpu>0)
                                for i in range(len(self.layers1)-1)
                                ])

        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1]*2, d_FC_layer) if i==0 else
                                 nn.Linear(d_FC_layer, 2) if i==n_FC_layer-1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])
        
        self.mu = nn.Parameter(torch.Tensor([args.initial_mu]).float())
        self.dev = nn.Parameter(torch.Tensor([args.initial_dev]).float())
        self.embede = nn.Linear(2*N_atom_features, d_graph_layer, bias = False)
        

    def embede_graph(self, data, chunk_idx=None):
        c_hs, c_adjs, c_attr, c_valid = data
        c_hs = self.embede(c_hs)
        c_attr1, c_attr2 = torch.split(c_attr, [1,1], dim=1)
        c_attr2 = torch.exp(-torch.pow(c_attr2-self.mu.expand_as(c_attr2), 2)/self.dev) + c_attr1
        
        # regularization = torch.empty(len(self.gconv1), device=c_hs.device)
        for k in range(len(self.gconv1)):
            c_hs1 = self.gconv1[k](x=c_hs, edge_index=c_adjs, edge_attr=c_attr1)
            c_hs2 = self.gconv1[k](x=c_hs, edge_index=c_adjs, edge_attr=c_attr2)
            c_hs =  c_hs2 - c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
        
        c_hs_l = c_hs*c_valid[:,0].unsqueeze(-1)
        c_hs_p = c_hs*c_valid[:,1].unsqueeze(-1)
        
        if chunk_idx is not None:
            c_hs_l = torch.split(c_hs_l, chunk_idx, dim=0)
            c_hs_p = torch.split(c_hs_p, chunk_idx, dim=0)
            
            c_hs_l = torch.stack([x.sum(0) for x in c_hs_l], dim=0)
            c_hs_p = torch.stack([x.sum(0) for x in c_hs_p], dim=0)
            
        output = torch.cat([c_hs_l, c_hs_p], dim=-1)
        # output = c_hs_l
        return output

    def fully_connected(self, c_hs):
        # regularization = torch.empty(len(self.FC)*1-1, device=c_hs.device)

        for k in range(len(self.FC)):
            if k<len(self.FC)-1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)
                
        # For classification
        # c_hs = torch.relu(c_hs)

        return c_hs

    def forward(self, data, chunk_idx=None):
        #embede a graph to a vector
        c_hs = self.embede_graph(data, chunk_idx=chunk_idx)

        #fully connected NN
        c_hs = self.fully_connected(c_hs)
        mean = c_hs[:,0]
        std = c_hs[:,1]
        std = torch.exp(std)
        c_hs =torch.stack([mean, std], dim=1)

        #note that if you don't use concrete dropout, regularization 1-2 is zero
        return c_hs

    # def get_refined_adjs2(self, data):
    #     c_hs, c_adjs1, c_adjs2, c_valid = data
    #     c_hs = self.embede(c_hs)
    #     c_adjs2 = torch.exp(-torch.pow(c_adjs2-self.mu.expand_as(c_adjs2), 2)/self.dev) + c_adjs1
    #     # regularization = torch.empty(len(self.gconv1), device=c_hs.device)

    #     for k in range(len(self.gconv1)):
    #         c_hs1 = self.gconv1[k](c_hs, c_adjs1)
    #         if k==len(self.gconv1)-1:
    #             c_hs2, attention = self.gconv1[k](c_hs, c_adjs2, True)
    #             return F.normalize(attention)
    #         else:
    #             c_hs2 = self.gconv1[k](c_hs, c_adjs2)
    #         c_hs = c_hs2-c_hs1
    #         c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)

    #     return c_adjs2
