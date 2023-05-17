import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, mode, K_IMP, ratio=1):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d((K_IMP, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d((K_IMP, 1))
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling

        self.fc = nn.Sequential(
            nn.Linear(K_IMP, K_IMP // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(K_IMP // ratio, K_IMP, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, N, K, _ = x.shape  # (B,N,K,T)
        v = self.global_pooling(x).view(B, N, K)  # squeeze
        v = self.fc(v).view(B, N, K, 1)
        v = self.sigmoid(v)
        return x * v  # (B,N,K,T)


class CBAMBlock(nn.Module):
    def __init__(self, spatial_attention_kernel_size: int, K_IMP: int = None, ratio: int = 1):
        super(CBAMBlock, self).__init__()
        self.channel_attention_block = Channel_Attention_Module_FC(
            K_IMP, ratio=ratio)
        self.spatial_attention_block = Spatial_Attention_Module(
            kernel_size=spatial_attention_kernel_size)

    def forward(self, x):
        x = self.channel_attention_block(x)
        x = self.spatial_attention_block(x)
        return x


class Channel_Attention_Module_FC(nn.Module):
    def __init__(self, K_IMP, ratio=1):
        super(Channel_Attention_Module_FC, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d((K_IMP, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d((K_IMP, 1))
        self.fc = nn.Sequential(
            nn.Linear(K_IMP, K_IMP // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(K_IMP // ratio, K_IMP, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, N, K, _ = x.shape  # (B,N,K,T)
        avg_x = self.avg_pooling(x).view(B, N, K)
        max_x = self.max_pooling(x).view(B, N, K)
        v = self.fc(avg_x) + self.fc(max_x)
        v = self.sigmoid(v).view(B, N, K, 1)
        return x * v


class Spatial_Attention_Module(nn.Module):
    def __init__(self, kernel_size: int):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max

        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network

        # assert kernel_size in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size=(1, kernel_size), stride=1, padding=(
            0, (kernel_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim=2, keepdim=True)  # (B, N, 1, T)
        max_x, _ = self.max_pooling(x, dim=2, keepdim=True)  # (B, N, 1, T)
        # (B, N, 2, T) --> (B, 2, N, T) --> (B, 1, N, T)
        v = self.conv(torch.cat((max_x, avg_x), dim=2).permute(0, 2, 1, 3))
        # (B, 1, N, T) --> (B, N, 1, T)
        v = self.sigmoid(v.permute(0, 2, 1, 3))
        return x * v
