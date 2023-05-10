import torch.nn as nn
import numpy as np
import torch
from models.s4.s4 import S4Block as S4
from models.s4.s4d import S4D
from models.recurrent_base import RecurrentActorProb, RecurrentCritic
from typing import Sequence, Union, Optional, Dict, Any, Tuple


SIGMA_MIN = -20
SIGMA_MAX = 2


class S4Model(nn.Module):
    def __init__(self, d_input, d_output=10, d_model=256, n_layers=4, dropout=0.2, lr=0.001, prenorm=False, use_s4d=True):
        super().__init__()
        self.prenorm = prenorm
        # Linear encoder (d_input = 17 for walker)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            if use_s4d:
                self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr)))
            else:
                self.s4_layers.append(S4(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr)))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout1d(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x, times=None):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            # Dropout on the output of the S4 block
            z = dropout(z)
            # Residual connection
            x = z + x
            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        # # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)
        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        return x

    def param_count(self):
        params = 0
        for param in self.parameters():
            params += np.prod(param.shape)
        return params


class S4ActorProb(RecurrentActorProb):
    def __init__(
        self,
        preprocess_net: nn.Module,
        layer_num: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        hidden_layer_size: int = 128,
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
    ) -> None:
        super().__init__(preprocess_net,
                         S4(hidden_layer_size, dropout=0.2, lr=1e-4, transposed=False).to(device), 
                         action_shape, hidden_layer_size, max_action, device, unbounded, conditioned_sigma)
        self.recurrent_model.setup_step()

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        obs = self.preprocess(obs)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        if state is None:
            hidden = self.recurrent_model.default_state(obs.shape[0])
            obs, state = self.recurrent_model(obs)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            obs, hidden = self.recurrent_model.step(obs.squeeze(-2), state["hidden"])
        if len(obs.shape) == 3:
            logits = obs[:, -1]
        else:
            logits = obs
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        # please ensure the first dim is batch size: [bsz, len, ...]
        return (mu, sigma), {"hidden": hidden}


class S4Critic(RecurrentCritic):
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
                         S4Model(hidden_layer_size, hidden_layer_size, n_layers=layer_num, dropout=0.2, lr=1e-4).to(device),
                         state_shape, action_shape, device, hidden_layer_size)
