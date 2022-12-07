import os
import torch
import time
import pickle

import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader                                     
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from .gnn import gnn
from .dataset import MolDataset, collate_fn, get_atom_feature
from .utils import N_atom_features, set_cuda_visible_device, initialize_model

class GMGM:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        
        self.train_dataset = None
        self.test_dataset = None
        
        self.build_model(args)
        
    def build_model(self, args):
        if args.ngpu > 0:
            cmd = set_cuda_visible_device(args.ngpu)
            os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]
            
        self.model = gnn(args)
        print ('number of parameters : ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = initialize_model(self.model, self.device, load_save_file=args.ckpt, gpu=self.args.ngpu>0)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        # Loss function
        self.loss_fn = torch.nn.L1Loss()
        
    def train(self, keys_file=None, epochs=None):
        # train code
        if keys_file is None:
            keys_file = self.args.train_keys
            
        if epochs is None:
            epochs = self.args.epochs
            
        with open (keys_file, 'rb') as fp:
            keys = pickle.load(fp)
            
        if self.train_dataset is None:
            self.train_dataset = MolDataset(keys, self.args.data_fpath, train=True)
            
        train_dataloader = DataLoader(self.train_dataset, self.args.batch_size, \
        shuffle=False, num_workers = self.args.num_workers, collate_fn=collate_fn)
        
       
        for epoch in range(epochs):
            st = time.time()
            print("EPOCH %d:"%epoch)
            #collect losses of each iteration
            train_losses = [] 

            #collect true label of each iteration
            train_true = []
            
            #collect predicted label of each iteration
            train_pred = []
            
            self.model.train()
            for sample in tqdm(train_dataloader):
                #train neural network
                pred, loss = self.step(sample, training=True)
                
                #collect loss, true label and predicted label
                train_losses.append(loss.data.cpu().numpy())
                train_true.append(Y.data.cpu().numpy())
                train_pred.append(pred.data.cpu().numpy())
                
            et = time.time()
            
            #calculate average loss
            avg_train_loss = np.mean(train_losses)
            print("Average train loss: %.4f" % avg_train_loss)
            print("Time taken: %.2f" % (et-st))
            
    def test(self, keys_file=None):
        # test code
        if keys_file is None:
            keys_file = self.args.train_keys
            
        with open (keys_file, 'rb') as fp:
            keys = pickle.load(fp)
            
        if self.test_dataset is None:
            self.test_dataset = MolDataset(keys, self.args.data_fpath, train=False)
            
        test_dataloader = DataLoader(self.test_dataset, self.args.batch_size, \
        shuffle=False, num_workers = self.args.num_workers, collate_fn=collate_fn)
        
       
        st = time.time()
        #collect losses of each iteration
        test_losses = [] 

        #collect true label of each iteration
        test_true = []
        
        #collect predicted label of each iteration
        test_pred = []
        
        self.model.eval()
        for sample in tqdm(test_dataloader):
            #train neural network
            pred, loss = self.step(sample, training=False)
            
            #collect loss, true label and predicted label
            test_losses.append(loss.data.cpu().numpy())
            test_true.append(Y.data.cpu().numpy())
            test_pred.append(pred.data.cpu().numpy())
            
        et = time.time()
        
        #calculate average loss
        avg_train_loss = np.mean(test_losses)
        print("Average test loss: %.4f" % avg_train_loss)
        print("Time taken: %.2f" % (et-st))
        
    
    def step(self, sample, training=True):
        # forward code
        self.model.zero_grad()
        H, A1, A2, Y, V = sample 
        H, A1, A2, Y, V = H.to(self.device), A1.to(self.device), A2.to(self.device),\
                            Y.to(self.device), V.to(self.device)
        
        #train neural network
        pred = self.model.train_model((H, A1, A2, V))
        loss = self.loss_fn(pred, Y)

        H, A1, A2, Y, V = H.to("cpu"), A1.to("cpu"), A2.to("cpu"),\
                                Y.to("cpu"), V.to("cpu")
        
        if training:
            loss.backward()
            self.optimizer.step()
        
        return pred, loss
    
    def prepare_single_input(self, m1, m2, m1_feature=None, m2_feature=None, label=None):
        #prepare ligand
        n1 = m1.GetNumAtoms()
        c1 = m1.GetConformers()[0]
        d1 = np.array(c1.GetPositions())
        adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
        if m1_feature == None:
            m1_feature = self.get_feature_dict(m1)
        H1 = get_atom_feature(m1, m1_feature, True)

        #prepare protein
        n2 = m2.GetNumAtoms()
        c2 = m2.GetConformers()[0]
        d2 = np.array(c2.GetPositions())
        adj2 = GetAdjacencyMatrix(m2)+np.eye(n2)
        if m2_feature == None:
            m2_feature = self.get_feature_dict(m2)
        H2 = get_atom_feature(m2, m2_feature, False)
        
        #aggregation
        H = np.concatenate([H1, H2], 0)
        agg_adj1 = np.zeros((n1+n2, n1+n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        agg_adj2 = np.copy(agg_adj1)
        dm = distance_matrix(d1,d2)
        agg_adj2[:n1,n1:] = np.copy(dm)
        agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))

        #node indice for aggregation
        valid = np.zeros((2, n1+n2,))
        valid[0,:n1] = 1
        valid[1,np.unique(np.where(dm < 5)[1])] = 1

        sample = {
                  'H': H, \
                  'A1': agg_adj1, \
                  'A2': agg_adj2, \
                  'V': valid, \
                  }
        
        if label is not None:
            sample['Y'] = label

        return sample

    def input_to_tensor(self, batch_input):
        max_natoms = max([len(item['H']) for item in batch_input if item is not None])
        batch_size = len(batch_input)
    
        H = np.zeros((batch_size, max_natoms, 2*N_atom_features))
        A1 = np.zeros((batch_size, max_natoms, max_natoms))
        A2 = np.zeros((batch_size, max_natoms, max_natoms))
        V = np.zeros((batch_size, 2, max_natoms))
        Y = np.zeros((batch_size,))
        
        for i in range(batch_size):
            natom = len(batch_input[i]['H'])
            
            H[i,:natom] = batch_input[i]['H']
            A1[i,:natom,:natom] = batch_input[i]['A1']
            A2[i,:natom,:natom] = batch_input[i]['A2']
            V[i,:,:natom] = batch_input[i]['V']
            Y[i] = batch_input[i]['Y']

        H = torch.from_numpy(H).float().to(self.device)
        A1 = torch.from_numpy(A1).float().to(self.device)
        A2 = torch.from_numpy(A2).float().to(self.device)
        V = torch.from_numpy(V).float().to(self.device)
        Y = torch.from_numpy(Y).float().to(self.device)

        return H, A1, A2, Y, V
    
    def prepare_batch_input(self, batch, batch_size=32):
        # Yield batches of data until all batches have been yielded
        loop_times = int(np.ceil(len(batch)/batch_size))
        
        for i in range(loop_times):
            batch_input = []
            for j in range(batch_size):
                if i*batch_size+j < len(batch):
                    m1, m2, m1_feature, m2_feature, label = batch[i*batch_size+j]
                    
                    single_input = self.prepare_single_input(m1, m2, m1_feature, m2_feature, label)
                    batch_input.append(single_input)
                    
            yield self.input_to_tensor(batch_input)
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
            
    def load(self, path):
        self.args.ckpt = path
        self.model = initialize_model(self.model, self.device, load_save_file=self.args.ckpt, gpu=self.args.ngpu>0)
    
        