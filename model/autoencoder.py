import torch.nn as nn
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(210, 128).cuda(),
            nn.Tanh().cuda(),
            nn.Linear(128, 32).cuda(),
        ).cuda()
        self.decoder = nn.Sequential(
            nn.Linear(32, 128).cuda(),
            nn.Tanh().cuda(),
            nn.Linear(128, 210).cuda(),
            nn.Sigmoid().cuda()
        ).cuda()

    def forward(self, x):
        encoder = self.encoder(x).cuda()
        decoder = self.decoder(encoder).cuda()
        return encoder, decoder

