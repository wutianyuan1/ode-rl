import torch.nn as nn

class RepresentationMLP(nn.Module):
    def __init__(self, in_features, hiddens, activatition, device) -> None:
        super().__init__()
        self.device = device
        self.activatition = activatition()
        self.fc_in = nn.Linear(in_features, hiddens[0], device=self.device)
        self.hiddens = [nn.Linear(hiddens[i], hiddens[i+1], device=self.device) for i in range(len(hiddens)-1)]
        self.output_dim = hiddens[-1]
    
    def forward(self, obs):
        obs = self.activatition(self.fc_in(obs))
        for i in range(len(self.hiddens)):
            obs = self.activatition(self.hiddens[i](obs))
        return obs
