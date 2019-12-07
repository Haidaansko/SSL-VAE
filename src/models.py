import numpy as np
import torch
import torch.nn as nn


X_DIM = 28 * 28
Y_DIM = 10
Z_DIM = 50
HIDDEN_DIM1 = 600
HIDDEN_DIM2 = 500
INIT_VAR = 0.001


class M1(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.encoder_z = torch.nn.Sequential(
            nn.Linear(X_DIM, HIDDEN_DIM1),
            nn.Softplus(),
            nn.Linear(HIDDEN_DIM1, HIDDEN_DIM1),
            nn.Softplus(),
            nn.Linear(HIDDEN_DIM1, Z_DIM * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(Z_DIM, HIDDEN_DIM1),
            nn.Softplus(),
            nn.Linear(HIDDEN_DIM1, HIDDEN_DIM1),
            nn.Softplus(),
            nn.Linear(HIDDEN_DIM1, X_DIM)
        )

        for p in self.parameters():
            if p.ndim == 1:
                p.data.fill_(0)
            else:
                p.data.normal_(0, INIT_VAR)
        
        self.p_z = torch.distributions.Normal(
            torch.zeros(1, device=device), 
            torch.ones(1, device=device)
        )


    def encode_z(self, x):
        means_z, logsigma_z = self.encoder_z(x).chunck(2, dim=1)
        return torch.distributions.Normal(means_z, torch.exp(logsigma_z))


    def decode(self, z):
        return torch.distributions.Bernoulli(logits=self.decoder(z))



class M2(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.encoder_y = torch.nn.Sequential(
            nn.Linear(X_DIM, HIDDEN_DIM2),
            nn.Softplus(),
            nn.Linear(HIDDEN_DIM2, HIDDEN_DIM2),
            nn.Softplus(),
            nn.Linear(HIDDEN_DIM2, Y_DIM)
        )

        self.encoder_z = torch.nn.Sequential(
            nn.Linear(X_DIM + Y_DIM, HIDDEN_DIM2),
            nn.Softplus(),
            nn.Linear(HIDDEN_DIM2, HIDDEN_DIM2),
            nn.Softplus(),
            nn.Linear(HIDDEN_DIM2, Z_DIM * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(Y_DIM + Z_DIM, HIDDEN_DIM2),
            nn.Softplus(),
            nn.Linear(HIDDEN_DIM2, HIDDEN_DIM2),
            nn.Softplus(),
            nn.Linear(HIDDEN_DIM2, X_DIM)
        )

        for p in self.parameters():
            if p.ndim == 1:
                p.data.fill_(0)
            else:
                p.data.normal_(0, INIT_VAR)

                
        self.p_z = torch.distributions.Normal(
            torch.zeros(1, device=device), torch.ones(1, device=device)
        )
        self.p_y = torch.distributions.Categorical(
            probas=torch.ones((1, Y_DIM), device=device) / Y_DIM)


    def forward(self, x):
        probs = self.encode_y(x).probs
        return 


    def encode_y(self, x):
        return torch.distributions.Categorical(logits=self.encoder_y(x))


    def encode_z(self, x, y):
        means_z, logsigma_z = self.encoder_z(torch.cat([x, y], axis=1)).chunck(2, dim=1)
        return torch.distributions.Normal(means_z, torch.exp(logsigma_z))


    def decode(self, y, z):
        return torch.distributions.Bernoulli(
            logits=self.decoder(torch.cat([y, z], axis=1)))
