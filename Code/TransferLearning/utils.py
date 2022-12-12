import torch.nn as nn

## For CNN Flatten()
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

