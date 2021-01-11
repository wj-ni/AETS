import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.firstLayer = nn.Sequential(nn.Linear(32, 64).cuda(),
                                        nn.ReLU().cuda(),
                                        nn.Linear(64, 32).cuda(),
                                        nn.ReLU().cuda(),
                                        nn.Linear(32, 1).cuda(),
                                        ).cuda()

    def forward(self, x):
        first = self.firstLayer(x)
        return first
