import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.embed1 = nn.Linear(in_features, hidden_features)
        self.embed2 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.linear3 = nn.Linear(in_features, out_features)

    def forward(self, x):
        e1 = self.embed1(x) # (B,N,K,T)
        e2 = self.embed2(x).permute(0,1,3,2) # (B,N,T,K)
        adj = self.relu(torch.matmul(e1, e2))
        # x = self.linear(x)
        x = torch.matmul(adj, x) # (B,N,K,K)*(B,N,K,T) --> (B,N,K,T)
        x = self.linear3(x)
        # x = F.leaky_relu(x)
        # x = self.linear2(x)

        return x


if __name__ == '__main__':

    features = torch.randn(32,8,6,96)
    gcn = GCN(in_features=96, hidden_features=16, out_features=2)
    output = gcn(features)

    print(output.shape)
