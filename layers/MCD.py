import torch
import torch.nn as nn
import torch.nn.functional as F

class Morphology(nn.Module):
    '''
    Base class for morpholigical operators 
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''
    def __init__(self, kernel_size=3, soft_max=True, beta=15, optype=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure. 
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation1d or erosion1d.
        '''
        super(Morphology, self).__init__()
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.optype = optype
        self.kernel_size = kernel_size

        # self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        # self.weight = nn.Parameter(torch.zeros(batch_size, input_dim, kernel_size), requires_grad=False)
        self.weight = nn.Parameter(torch.zeros(1, self.kernel_size[-1], 1), requires_grad=False)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)


    def fixed_padding(self, inputs):
        padded_inputs = F.pad(inputs, (1, 1, 0, 0))
        return padded_inputs


    def forward(self, x):
        '''
        x: tensor of shape (B,N,T)
        '''
        B, N, L = x.shape
        # padding
        x = self.fixed_padding(x) # (B,N,T) --> (B,N,L), where L is the padding length

        x = x.unsqueeze(-2) # (B,N,1,L)
        
        # # unfold
        x = self.unfold(x)  # (B, N*1*kernel_len, L), where L is the numbers of patches  kernel_size = (1, kernel_len)
        x = x.reshape(B, N, self.kernel_size[-1], -1) # (B, N, kernel_len, L)

        if self.optype == 'erosion1d':
            x = self.weight - x # (B, N, kernel_len, L)
            # print(self.weight)
        elif self.optype == 'dilation1d':
            x = self.weight + x # (B, N, kernel_len, L)
        else:
            raise ValueError
        
        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False) # (B, N, T)
        else:
            x = torch.logsumexp(x*self.beta, dim=2, keepdim=False) / self.beta # (B, N, T)

        if self.optype == 'erosion1d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        # x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return x 

class Dilation1d(Morphology):
    def __init__(self, kernel_size=5, soft_max=True, beta=20):
        super(Dilation1d, self).__init__(kernel_size, soft_max, beta, 'dilation1d')

class Erosion1d(Morphology):
    def __init__(self, kernel_size=5, soft_max=True, beta=20):
        super(Erosion1d, self).__init__(kernel_size, soft_max, beta, 'erosion1d')



class MorphoEMP1D(nn.Module):
    def __init__(self, kernel_size=5, soft_max=True, beta=20, optype=None):
        super().__init__()
        self.dilation1d = Dilation1d(kernel_size=kernel_size, soft_max=soft_max)
        self.erosion1d = Erosion1d(kernel_size=kernel_size, soft_max=soft_max)

    def forward(self, x):
        xd=self.dilation1d(x)
        xe=self.erosion1d(x)
        avg = (xd+xe)/2
        imf = x - avg
        return imf


class MCD(nn.Module):
    def __init__(self, K_IMP, kernel_size=5, soft_max=True, beta=20, optype=None):
        super().__init__()
        self.morphoEMP1D = MorphoEMP1D(kernel_size, soft_max=soft_max)
        self.K_IMP = K_IMP


    def get_next_imf(self, X):
        continue_imf = True

        while continue_imf:
            x1 = self.morphoEMP1D(X)
            stop, metric = self.sd_stop(x1, X, sd=0.1)
            # print(metric)
            if stop:
                return x1
            else:
                X = x1
    
    def sd_stop(self, proto_imf, prev_imf, sd=0.2, niters=None):
        """Compute the sd sift stopping metric.

        Parameters
        ----------
        proto_imf : ndarray
            A signal which may be an IMF
        prev_imf : ndarray
            The previously identified IMF
        sd : float
            The stopping threshold
        niters : int
            Number of sift iterations currently completed
        niters : int
            Number of sift iterations currently completed

        Returns
        -------
        bool
            A flag indicating whether to stop siftingg
        float
            The SD metric value

        """
        metric = torch.sum((proto_imf - prev_imf)**2) / torch.sum(proto_imf**2)

        stop = metric < sd

        return stop, metric


    def forward(self, X):
        inside_X = X.clone()
        continue_sift = True
        layer = 0

        while continue_sift:
            next_imf = self.get_next_imf(inside_X)
            next_imf = next_imf.unsqueeze(2)
            if layer == 0:
                imf = next_imf
            else:
                imf = torch.cat((imf, next_imf), axis=2)

            layer += 1
            inside_X = X - imf.sum(2)

            if layer == self.K_IMP-1: 
                imf = torch.cat((imf, inside_X.unsqueeze(2)), axis=2)
                continue_sift = False

        return imf # (B, N, K, T)
