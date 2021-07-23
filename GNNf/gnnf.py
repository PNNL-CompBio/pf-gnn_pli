import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool

class gnnf(torch.nn.Module):
    def __init__(self, args):
        super(gnnf, self).__init__()
        n_graph_layer = args.n_graph_layer
        d_graph_layer = args.d_graph_layer
        n_FC_layer = args.n_FC_layer
        d_FC_layer = args.d_FC_layer
        self.dropout_rate = args.dropout_rate 


        self.layers1 = [d_graph_layer for i in range(n_graph_layer+1)]
        self.gconv1 = nn.ModuleList([GAT_gate(self.layers1[i], self.layers1[i+1]) for i in range(len(self.layers1)-1)]) 
        
        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1], d_FC_layer) if i==0 else
                                 nn.Linear(d_FC_layer, 1) if i==n_FC_layer-1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])
        
        self.mu = nn.Parameter(torch.Tensor([args.initial_mu]).float())
        self.dev = nn.Parameter(torch.Tensor([args.initial_dev]).float())
        self.embede = nn.Linear(74, d_graph_layer, bias = False)
        

    def embede_graph(self, data):
        c_hs, c_adjs1, c_adjs2, c_valid = data
        c_hs = self.embede(c_hs)
        hs_size = c_hs.size()
        c_adjs2 = torch.exp(-torch.pow(c_adjs2-self.mu.expand_as(c_adjs2), 2)/self.dev) + c_adjs1
        regularization = torch.empty(len(self.gconv1), device=c_hs.device)
        
        for k in range(len(self.gconv1)):
            c_hs1 = self.gconv1[k](c_hs, c_adjs1)
            c_hs2 = self.gconv1[k](c_hs, c_adjs2)
            c_hs = c_hs2-c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
        c_hs = c_hs*c_valid.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
        c_hs = c_hs.sum(1)
               
        return c_hs

    def fully_connected(self, c_hs):
        regularization = torch.empty(len(self.FC)*1-1, device=c_hs.device)

        for k in range(len(self.FC)):
            if k<len(self.FC)-1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)

        c_hs = torch.sigmoid(c_hs)

        return c_hs

    def train_model(self, data):
        #embede a graph to a vector
        c_hs = self.embede_graph(data)

        #fully connected NN
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1) 
        return c_hs
    
    def test_model(self,data1 ):
        c_hs = self.embede_graph(data1)
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)
        return c_hs

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
        e = torch.einsum('ijl, ikl->ijk', (torch.matmul(h,self.A), h))
        e = e + e.permute((0,2,1))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = attention*adj
        h_prime = F.relu(torch.einsum('aij,ajk->aik',(attention, h)))

        coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1))
        retval = coeff*x+(1-coeff)*h_prime
        return retval
