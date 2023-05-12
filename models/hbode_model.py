import torch
import torch.nn as nn
from models.hbode.base import HeavyBallNODE, ODE_RNN_with_Grad_Listener
from models.recurrent_base import RecurrentActorProb, RecurrentCritic
from typing import Sequence, Union


class ODEFunc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels, in_channels)
        self.dense2 = nn.Linear(in_channels, out_channels)
        self.dense3 = nn.Linear(out_channels, out_channels)

    def forward(self, h, x):
        out = self.dense1(x)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        out = self.dense3(out)
        return out


class RNN(nn.Module):
    def __init__(self, in_channels, out_channels, nhidden, res=False, cont=False):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels + 2 * nhidden, 2 * nhidden)
        self.dense2 = nn.Linear(2 * nhidden, 2 * nhidden)
        self.dense3 = nn.Linear(2 * nhidden, 2 * out_channels)
        self.cont = cont
        self.res = res

    def forward(self, h, x):
        out = torch.cat([h[:, 0], h[:, 1], x], dim=1)
        out = self.dense1(out)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        out = self.dense3(out).reshape(h.shape)
        out = out + h
        return out

class HBOdeModel(nn.Module):
    def __init__(self, state_dim, nhid, res=False, cont=False):
        super(HBOdeModel, self).__init__()
        self.cell = HeavyBallNODE(ODEFunc(nhid, nhid), corr=1, corrf=False, actv_h=nn.Tanh())
        # self.cell = HeavyBallNODE(ODEFunc(nhid, nhid))
        self.rnn = RNN(state_dim, nhid, nhid, res=res, cont=cont)
        self.ode_rnn = ODE_RNN_with_Grad_Listener(self.cell, self.rnn, (2, nhid), None, tol=1e-7)
        self.outlayer = nn.Linear(nhid, state_dim)

    def forward(self, x: torch.Tensor):
        # x: state hidden, shape=(Batch, Length, Hidden)
        x = x.transpose(0, 1) # (Length, Batch, Hidden)
        time = torch.arange(0, x.shape[0], device='cuda').repeat(x.shape[1]).resize((x.shape[1], x.shape[0]))
        out = self.ode_rnn(time, x, retain_grad=True)[0]
        out = self.outlayer(out[:, :, 0])[1:]
        return out


class HBOdeCritic(RecurrentCritic):
    def __init__(
        self,
        preprocess_net: nn.Module,
        layer_num: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int] = [0],
        device: Union[str, int, torch.device] = "cpu",
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__(preprocess_net, 
                         HBOdeModel(hidden_layer_size, hidden_layer_size),
                         state_shape, action_shape, device, hidden_layer_size)
