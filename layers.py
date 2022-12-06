import torch
import torch.nn.functional as F
import torch.nn as nn
from .utils import *
import time

class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature, nhop, gpu = 0):
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        #self.A = nn.Parameter(torch.Tensor(n_out_feature, n_out_feature))
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature*2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.zeros = torch.zeros(1)
        self.nhop = nhop
        if gpu > 0:
            self.zeros = self.zeros.cuda()

    def forward(self, x, adj, get_attention=False):
        h = self.W(x)
        # batch_size = h.size()[0]
        N = h.size()[1]
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h,self.A), h))
        e = e + e.permute((0,2,1))
        # zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, self.zeros)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        #h_prime = torch.matmul(attention, h)
        attention = attention*adj
        # h_prime = F.relu(torch.einsum('aij,ajk->aik',(attention, h)))
        # coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1))
        # retval = coeff*x+(1-coeff)*h_prime

        retval = h
        for _ in range(self.nhop):
            az = F.relu(torch.einsum('aij,ajk->aik',(attention, retval)))
            coeff = torch.sigmoid(self.gate(torch.cat([h, az], -1))).repeat(1, 1, h.size(-1))
            retval = coeff * h + (1 - coeff) * az
        
        if not get_attention:
            return retval

        return retval, attention
