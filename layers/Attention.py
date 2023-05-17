from math import sqrt
 
import torch
import torch.nn as nn
 
class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads
 
    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
 
    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        B, N, K, dim_in = x.shape
        assert dim_in == self.dim_in
 
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
 
        q = self.linear_q(x).reshape(B, N, K, nh, dk).transpose(2, 3)  # (B, N, nh, K, dk)
        k = self.linear_k(x).reshape(B, N, K, nh, dk).transpose(2, 3)  # (B, N, nh, K, dk)
        v = self.linear_v(x).reshape(B, N, K, nh, dk).transpose(2, 3)  # (B, N, nh, K, dk)
 
        dist = torch.matmul(q, k.transpose(3, 4)) * self._norm_fact  # # (B, N, nh, K, dk)*(B, N, nh, dk, K) --> (B, N, nh, K, K)
        dist = torch.softmax(dist, dim=-1)  # (B, N, nh, K, K)
 
        att = torch.matmul(dist, v)  # (B, N, nh, K, K)*(B, N, nh, K, dv) --> (B, N, nh, K, dv)
        att = att.transpose(1, 2).reshape(B, N, K, self.dim_v)  # (B, N, nh, K, dv) --> (B, N, K, nh, dv) --> (B, N, K, dim_v)

        return att




if __name__ == '__main__':

    model = MultiHeadSelfAttention(dim_in=96, dim_k=256*8, dim_v=256*8).cuda()
    
    x = torch.randn(32, 321, 12, 96).cuda() # (B,N,K,T)

    y = model(x) # (B, N, K, dim_v)

    y = torch.sum(y, 2)


    print(y.shape)