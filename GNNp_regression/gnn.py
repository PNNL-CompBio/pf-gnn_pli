import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool

protein_features = 31
ligand_features = 43

class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature*2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        h = self.W(x)
        batch_size = h.size()[0]
        N = h.size()[1]
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h,self.A), h))
        e = e + e.permute((0,2,1))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = attention*adj
        h_prime = F.relu(torch.einsum('aij,ajk->aik',(attention, h)))

        coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1))
        retval = coeff*x+(1-coeff)*h_prime
        return retval

class gnn(torch.nn.Module):
    def __init__(self, args):
        super(gnn, self).__init__()
        n_graph_layer = args.n_graph_layer
        d_graph_layer = args.d_graph_layer
        n_FC_layer = args.n_FC_layer
        d_FC_layer = args.d_FC_layer
        self.dropout_rate = args.dropout_rate 
  
        self.batch_size = args.batch_size
        self.layers1 = [d_graph_layer for i in range(n_graph_layer+1)]
        self.gconvprotein = nn.ModuleList([GAT_gate(self.layers1[i], self.layers1[i+1]) for i in range(len(self.layers1)-1)]) 
        self.gconvligand = nn.ModuleList([GAT_gate(self.layers1[i], self.layers1[i+1]) for i in range(len(self.layers1)-1)]) 
        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1]*2, d_FC_layer) if i==0 else
                                 nn.Linear(d_FC_layer, 1) if i==n_FC_layer-1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])
        
        self.densaft = nn.Linear(self.batch_size,self.layers1[-1]) 
        self.mu = nn.Parameter(torch.Tensor([args.initial_mu]).float())
        self.dev = nn.Parameter(torch.Tensor([args.initial_dev]).float())
        self.embede_protein = nn.Linear(protein_features, d_graph_layer, bias = False)
        
        self.embede_ligand = nn.Linear(ligand_features, d_graph_layer, bias = False)

    def embede_graph_protein(self, data):
        c_hs1, c_hs2, c_adjs1, c_adjs2, c_valid = data
        c_hs2 = self.embede_protein(c_hs2)
        hs_size = c_hs2.size()
        c_adjs2 = torch.exp(-torch.pow(c_adjs2-self.mu.expand_as(c_adjs2), 2)/self.dev) 
        regularization = torch.empty(len(self.gconvprotein), device=c_hs2.device)

        for k in range(len(self.gconvprotein)):
            c_hs2 = self.gconvprotein[k](c_hs2, c_adjs2)
            c_hs2 = c_hs2
            c_hs2 = F.dropout(c_hs2, p=self.dropout_rate, training=self.training)
        c_hs2 = c_hs2.sum(1)
        return c_hs2

    def embede_graph_ligand(self, data):
        c_hs1, c_hs2, c_adjs1, c_adjs2, c_valid = data
        c_hs1 = self.embede_ligand(c_hs1)
        hs_size = c_hs1.size()
        c_adjs1 = torch.exp(-torch.pow(c_adjs1-self.mu.expand_as(c_adjs1), 2)/self.dev)
        regularization = torch.empty(len(self.gconvligand), device=c_hs1.device)
        for k in range(len(self.gconvligand)):
            c_hs1 = self.gconvligand[k](c_hs1, c_adjs1)
            c_hs1 = c_hs1
            c_hs1 = F.dropout(c_hs1, p=self.dropout_rate, training=self.training)
        c_hs1 = c_hs1.sum(1)
        return c_hs1

    def fully_connected(self, c_hs):
        regularization = torch.empty(len(self.FC)*1-1, device=c_hs.device)
        for k in range(len(self.FC)):
            if k<len(self.FC)-1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)


        return c_hs

    def train_model(self, data):
        #embede a graph to a vector
        c_hs2 = self.embede_graph_protein(data)
        c_hs1 = self.embede_graph_ligand(data)
        c_hs3 = torch.cat((c_hs1, c_hs2), 1)
        #fully connected NN
        c_hs = self.fully_connected(c_hs3)
        c_hs = c_hs.view(-1) 
        #note that if you don't use concrete dropout, regularization 1-2 is zero
        return c_hs
    
    def test_model(self,data1 ):
        c_hs2 = self.embede_graph_protein(data1)
        c_hs1 = self.embede_graph_ligand(data1)
        c_hs3 = torch.cat((c_hs1, c_hs2), 0)
        c_hs = self.fully_connected(c_hs3)
        c_hs = c_hs.view(-1)
        return c_hs
