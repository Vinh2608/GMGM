import os
import torch
import time
import pickle

import numpy as np
from tqdm import tqdm

from torch.distributions.normal import Normal
from torch_geometric.data import  Batch           
from torch_geometric.utils import from_networkx

from GMGM.gnn import gnn
from GMGM.dataset import get_atom_feature
from GMGM.utils import  initialize_model, merge_graphs

class GMGM:
    def __init__(self, args, mid=1, optimizer=None, loss_fn=None):
        self.args = args
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        
        self.train_dataset = None
        self.test_dataset = None
        
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.mid=mid
        self.ckpt = ""
            
        self.build_model(mid, args)
        
    def build_model(self, mid, args):
        self.model = gnn(args)
        self.device = torch.device(f'cuda:{(mid-1)}' if torch.cuda.is_available() else "cpu")
        
        # print(self.model)
        # print ('Number of parameters : ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        # print('Device', self.device)
        
        # Optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        # Loss function
        if self.loss_fn is None:
            self.loss_fn = torch.nn.L1Loss()
    
    def step(self, sample, training=True):
        # forward code
        self.model.zero_grad()
        H, A, E, Y, V, chunk_idx = sample 
        H, A, E, Y, V = H.to(self.device), A.to(self.device), E.to(self.device),\
                        Y.to(self.device), V.to(self.device)
                        
        #train neural network
        out = self.model((H, A, E, V), chunk_idx)
        dist = Normal(out[:,0], out[:,1])
        
        if training:
            pred = dist.rsample()
            loss = self.loss_fn(pred, Y)# + \
                #    self.loss_fn(out[:,1], torch.zeros_like(Y).to(self.device))
                
            loss.backward()
            self.optimizer.step()
        else:
            loss = self.loss_fn(out[:,0], Y)
            
        H, A, E, Y, V = H.to("cpu"), A.to("cpu"), E.to("cpu"), Y.to("cpu"), V.to("cpu")
            
        return dist, loss
    
    def prepare_single_input(self, m1, m2, m1_feature=None, m2_feature=None, label=None):
        #prepare ligand
        c1 = m1.GetConformers()[0]
        d1 = np.array(c1.GetPositions())
        if m1_feature == None:
            m1_feature = self.get_feature_dict(m1)
        H1 = get_atom_feature(m1, m1_feature, True)

        #prepare protein
        c2 = m2.GetConformers()[0]
        d2 = np.array(c2.GetPositions())
        if m2_feature == None:
            m2_feature = self.get_feature_dict(m2)
        H2 = get_atom_feature(m2, m2_feature, False)
        
        #aggregation
        G = merge_graphs((m1, H1, d1), (m2, H2, d2))

        sample = {
                  'G': G,
                  }
        
        if label is not None:
            sample['Y'] = label

        return sample

    def input_to_tensor(self, batch_input):
        batch = Batch.from_data_list([from_networkx(sample['G']) for sample in batch_input])
          
        H, A, E = batch.x, batch.edge_index, batch.edge_attr
        V = batch.valid
        
        Y = np.zeros((len(batch_input),))
        for i in range(len(batch_input)):
            Y[i] = batch_input[i]['Y']
            
        H = H.float()
        A = A.long()
        E = E.float()
        V = V.long()
        Y = torch.from_numpy(Y).float()
        chunk_idx = [sample['G'].number_of_nodes() for sample in batch_input]
        
        return H, A, E, Y, V, chunk_idx
    
    def prepare_batch_input(self, batch, batch_size=32):
        # Yield batches of data until all batches have been yielded
        loop_times = int(np.ceil(len(batch)/batch_size))
        
        if isinstance(batch, dict):
            use_dict = True
            list_keys = list(batch.keys())
        else:
            use_dict = False
            
        total_sample = len(batch)
        
        for i in range(loop_times):
            batch_input = []
            for j in range(batch_size):
                sample_idx = i*batch_size+j
                if sample_idx < total_sample:
                    if use_dict:
                        sample_idx = list_keys[sample_idx]
                    m1, m2, m1_feature, m2_feature, label = batch[sample_idx]
                    
                    single_input = self.prepare_single_input(m1, m2, m1_feature, m2_feature, label)
                    batch_input.append(single_input)
            
            if use_dict:
                yield list_keys[i*batch_size:(i+1)*batch_size], self.input_to_tensor(batch_input) 
            else:
                yield None, self.input_to_tensor(batch_input)
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
            
    def init(self, path=""):
        if path:
            self.ckpt = path
        self.model = initialize_model(self.model, self.device, 
                                      load_save_file=self.ckpt, gpu=self.args.ngpu>0)
    
        