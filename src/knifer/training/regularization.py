import torch
import torch.nn as nn
import torch.linalg as linalg

class SpectralRegularization(nn.Module):
    def __init__(self):
        super().__init__()
        self.highest_sigmas = None

    def forward(self, weight: nn.parameter.Parameter):
        with torch.no_grad():
            if self.highest_sigmas is None:
                W = weight.data.reshape(weight.shape[0], -1)#.to('cpu')
                s = linalg.svdvals(W)
                self.highest_sigmas = s/max(s)
            else:
                
                W = weight.data.reshape(weight.shape[0], -1)#.to('cpu')
                U, s, Vt = linalg.svd(W, full_matrices=False)
                s1 = max(s)
                s = s / s1
                self.highest_sigmas = torch.maximum(s, self.highest_sigmas)
                new_s = s1 * self.highest_sigmas
                S = torch.diag(new_s)
                # does assigning W, as a view of the weight matrix
                # allow us to directly change weight data from USVt ?
                # apparently not really, that's only when assigning
                # array items
                W: torch.Tensor = U @ S @ Vt
                weight.data = W.reshape(weight.shape)#.to(DEVICE)
