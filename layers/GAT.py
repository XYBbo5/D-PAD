import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, input_dim, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)    #Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))) 
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)  (B, N, 2^(level-1)*K, T=input_len)*(T=input_len, out_features) -> (B, N, 2^(level-1)*K, out_features)
        # print(Wh.shape)
        a_input = self._prepare_attentional_mechanism_input(Wh) 
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1)) # (B, N, Nodes, Nodes)
        

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        #h_prime.shape=(N,out_features)

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        B, N, Nodes, OF = Wh.size()
        # Nodes = Wh.size()[2] # number of nodes # (B, N, 2^(level-1)*K, T=out_features)

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(Nodes, dim=2) # (B, N, Nodes*Nodes, T=out_features)
        
        Wh_repeated_alternating = Wh.repeat(1, 1, Nodes, 1) # (B, N, Nodes, Nodes, T=out_features)
        
        
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1) # (B, N, Nodes, Nodes, 2*out_features)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(B, N, Nodes, Nodes, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MultiGraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, input_dim, in_features, out_features, dropout, alpha, concat=True):
        super(MultiGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)    
        self.a = nn.Parameter(torch.empty(size=(2, 2*out_features, 1))) 
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)  (B, N, 2^(level-1)*K, T=input_len)*(T=input_len, out_features) -> (B, N, 2^(level-1)*K, out_features)
        # print(Wh.shape)
        h_prime = [] # (2,6,out_features)

        for i in range(int(h.shape[2]//6)):
            a_input = self._prepare_attentional_mechanism_input(Wh[:,:,i*6:(i+1)*6,:]) 
            e = self.leakyrelu(torch.matmul(a_input, self.a[i]).squeeze(-1)) # (B, N, Nodes, Nodes)
            zero_vec = -9e15*torch.ones_like(e[i])
            attention = torch.where(adj[i] > 0, e[i], zero_vec) 
            attention = F.softmax(attention, dim=-1) 
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime.append(torch.matmul(attention, Wh[:,:,i*6:(i+1)*6,:])) 


        h_prime = torch.cat(h_prime, dim=2)
        #h_prime.shape=(N,out_features)

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        B, N, Nodes, OF = Wh.size()

        Wh_repeated_in_chunks = Wh.repeat_interleave(Nodes, dim=2) # (B, N, Nodes*Nodes, T=out_features)

        Wh_repeated_alternating = Wh.repeat(1, 1, Nodes, 1) # (B, N, Nodes, Nodes, T=out_features)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1) # (B, N, Nodes, Nodes, 2*out_features)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(B, N, Nodes, Nodes, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class GAT(nn.Module):
    def __init__(self, input_dim, n_feature, n_hid,dropout, alpha, n_heads):
        """
        Dense version of GAT
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(input_dim, n_feature, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        
        # self.attentions = [MultiGraphAttentionLayer(input_dim, n_feature, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
        #                    range(n_heads)]
                           
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention) 


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)  
        x = F.dropout(x, self.dropout, training=self.training)  
        return x




if __name__ == '__main__':
    input_X = torch.randn(3,5)
    adj = torch.ones(3,3) - torch.eye(3)
    model0 = GAT(n_feature=5, n_hid=5, dropout=0.1, alpha=0.1, n_heads=2)
    out = model0(input_X, adj)
    print(out.shape)

