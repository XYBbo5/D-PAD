from torch import nn
import torch
import argparse
# import sys
# sys.path.append("..")
from layers.MCD import MCD
from layers.GAT import GAT
from utils.gumbel_softmax import gumbel_softmax


class D_R(nn.Module):
    def __init__(self, input_dim, input_len, total_level, current_level, dropout, enc_hidden, output_len, K_IMP):
        super().__init__()
        self.current_level = current_level
        self.input_len = input_len
        self.input_dim = input_dim
        self.enc_hidden = enc_hidden
        self.output_len = output_len
        self.dropout = dropout
        self.K_IMP = K_IMP
        self.total_level = total_level

        self.MCD = MCD(K_IMP, kernel_size=(1, 3), soft_max=False)

        if current_level == 0:
            pass
        else:
            self.branch_slelect = nn.Sequential(
                nn.Linear(input_len, 64),
                nn.BatchNorm2d(input_dim, affine=False),
                nn.LeakyReLU(),
                nn.Linear(64, 2),
            )

            self.reconstruct_proj_left = nn.Sequential(
                nn.Linear(input_len, enc_hidden),
                nn.BatchNorm1d(input_dim, affine=False),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(enc_hidden, input_len),
                nn.BatchNorm1d(input_dim, affine=False),
                nn.LeakyReLU()
            )

            self.reconstruct_proj_right = nn.Sequential(
                nn.Linear(input_len, enc_hidden),
                nn.BatchNorm1d(input_dim, affine=False),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(enc_hidden, input_len),
                nn.BatchNorm1d(input_dim, affine=False),
                nn.LeakyReLU()
            )

            self.EMDNet_Tree_left = D_R(
                input_dim, input_len, total_level, current_level-1, dropout, enc_hidden, output_len, K_IMP)
            self.EMDNet_Tree_right = D_R(
                input_dim, input_len, total_level, current_level-1, dropout, enc_hidden, output_len, K_IMP)

            if current_level == total_level:
                self.forecast_proj1 = nn.ModuleList(
                    [nn.Linear(self.input_len, self.enc_hidden)
                     for i in range(2**total_level*self.K_IMP)]
                )
                self.activate = nn.LeakyReLU()
                self.forecast_proj2 = nn.ModuleList(
                    [nn.Linear(self.enc_hidden, self.input_len)
                     for i in range(2**total_level*self.K_IMP)]
                )
            else:
                pass

    def decompose_MCD(self, x):
        x = self.MCD(x)
        return x  # (B, N, K, T)

    def reconstruct(self, x_imf):  # (B, N, K, T)

        select_feature = self.branch_slelect(x_imf)  # (B, N, K, 2)
        # (B, N, K, 2) one_hot if hard = True
        hard_class = gumbel_softmax(select_feature, hard=False)
        x_imf = x_imf.permute(0, 1, 3, 2)  # (B, N, T, K)
        x_summed = torch.matmul(x_imf, hard_class)  # (B, N, T, 2)

        x_left = self.reconstruct_proj_left(
            x_summed[:, :, :, 0])  # (B, N, T) --> # (B, N, T)
        x_right = self.reconstruct_proj_right(x_summed[:, :, :, 1])

        return x_left, x_right

    def forecast(self, total_imf):

        for i in range(2**self.total_level*self.K_IMP):
            y_current_imf = self.forecast_proj2[i](
                self.activate(
                    self.forecast_proj1[i](total_imf[:, :, i, :])
                )
            )
            y_current_imf = y_current_imf.unsqueeze(2)
            if i == 0:
                y_imf = y_current_imf
            else:
                y_imf = torch.cat((y_imf, y_current_imf), axis=2)

        return y_imf  # (B,N,2^level*K,T=input_len)

    def forward(self, x):
        # leaves
        if self.current_level == 0:
            x_imf = self.decompose_MCD(x)
            return x_imf  # (B,N,K,T)
        else:
            x_imf = self.decompose_MCD(x)
            x_left, x_right = self.reconstruct(x_imf)
            imf_left = self.EMDNet_Tree_left(x_left)
            imf_right = self.EMDNet_Tree_left(x_right)
            # 2*(B,N,2^(level-1)*K,T) --> (B,N,2^level*K,T)
            total_imf = torch.cat([imf_left, imf_right], dim=2)

            if self.current_level == self.total_level:
                y = self.forecast(total_imf)
                return y
            else:  # mid-level
                return total_imf


class IFNet(nn.Module):
    def __init__(self, output_len, input_len, input_dim=9, dec_hidden=1024, dropout=0.5, K_IMP=6):
        super(IFNet, self).__init__()
        self.graph_att = GAT(input_dim, n_feature=input_len,
                             n_hid=dec_hidden, dropout=dropout, alpha=0.1, n_heads=1)
        self.predict = nn.Linear(dec_hidden, output_len)

    def forward(self, x, adj):
        x = self.graph_att(x, adj)
        x = torch.sum(x, 2)
        x = self.predict(x)
        return x


class DPAD(nn.Module):
    def __init__(self, input_dim, input_len, num_levels, dropout, enc_hidden, dec_hidden, output_len, K_IMP):
        super().__init__()
        self.num_levels = num_levels
        self.K_IMP = K_IMP
        self.D_R_D = D_R(
            input_dim=input_dim,
            input_len=input_len,
            total_level=num_levels-1,
            current_level=num_levels-1,
            dropout=dropout,
            enc_hidden=enc_hidden,
            output_len=output_len,
            K_IMP=K_IMP
        )
        self.IF = IFNet(
            output_len=output_len,
            input_len=input_len,
            input_dim=input_dim,
            dec_hidden=dec_hidden,
            dropout=dropout,
            K_IMP=K_IMP
        )

    def forward(self, x):

        x = self.TreeBlock(x)  # (B, N, 2^(level-1)*K, T=input_len)

        adj = (torch.ones(2**(self.num_levels-1)*self.K_IMP, 2**(self.num_levels-1)
               * self.K_IMP) - torch.eye(2**(self.num_levels-1)*self.K_IMP)).to(x.device)

        x = self.decoder(x, adj)

        return x


class DPAD_GAT(nn.Module):
    def __init__(self, output_len, input_len, input_dim=9, enc_hidden=1, dec_hidden=1, num_levels=3, dropout=0.5,
                 single_step_output_One=0, K_IMP=6, RIN=0):
        super(DPAD_GAT, self).__init__()

        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.num_levels = num_levels
        self.single_step_output_One = single_step_output_One
        self.dropout = dropout
        self.K_IMP = K_IMP
        self.RIN = RIN

        self.DPAD = DPAD(
            input_dim=self.input_dim,
            input_len=self.input_len,
            num_levels=self.num_levels,
            dropout=self.dropout,
            enc_hidden=self.enc_hidden,
            dec_hidden=dec_hidden,
            output_len=output_len,
            K_IMP=self.K_IMP
        )

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))

    def forward(self, x):

        ### activated when RIN flag is set ###
        if self.RIN:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            x = x * self.affine_weight + self.affine_bias

        x = x.permute(0, 2, 1)  # (B,N,T)
        x = self.DPAD(x)
        x = x.permute(0, 2, 1)   # (B,T,N)

        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means

        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--window_size', type=int, default=96)
    parser.add_argument('--horizon', type=int, default=96)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden-size', default=1, type=int,
                        help='hidden channel of module')
    parser.add_argument('--RIN', default=1, type=int, help='ReVIN')
    parser.add_argument('--num_levels', default=2,
                        type=int, help='number of levels')
    parser.add_argument('--K_IMP', type=int, default=6)
    args = parser.parse_args()
    model = DPAD_GAT(input_len=168, output_len=args.horizon, input_dim=8, enc_hidden=168,
                     dec_hidden=168, dropout=0.5, num_levels=2, K_IMP=6, RIN=1).cuda()

    x = torch.randn(32, 168, 8).cuda()

    y = model(x)

    print(y.shape)
