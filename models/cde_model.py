from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from models.neuralcde.neuralcde import NeuralCDE
from models.recurrent_base import RecurrentActorProb, RecurrentCritic

SIGMA_MIN = -20
SIGMA_MAX = 2


class NeuralCDEActorProb(nn.Module):
    """Recurrent version of ActorProb.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess: nn.Module,
        layer_num: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        hidden_layer_size: int = 128,
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess
        self.nn = NeuralCDE(
            input_channels=hidden_layer_size,
            hidden_channels=hidden_layer_size
        )
        output_dim = int(np.prod(action_shape))
        self.tanh = nn.Tanh()
        self.mu = nn.Linear(hidden_layer_size, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layer_size, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self._max = max_action
        self._unbounded = unbounded

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
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        obs = self.preprocess(obs)
        orig_obs = obs.detach()

        if state is None:
            obs, hidden = self.nn(obs)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            obs, hidden = self.nn(
                obs, state["prev_z"], state["prev_s"]
            )

        logits = self.tanh(obs)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        
        obs_shape = list(orig_obs.shape)
        obs_shape[-2] = 4 # time_len
        
        prev_s = torch.zeros(obs_shape, device=self.device) if (state is None) else state["prev_s"]
        prev_s = torch.cat((prev_s[:, 1:, :], orig_obs), dim=1)
        # please ensure the first dim is batch size: [bsz, len, ...]
        return (mu, sigma), {
            "prev_z": hidden.detach(),
            "prev_s": prev_s
        }


class NeuralCDECritic(RecurrentCritic):
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
                         NeuralCDE(hidden_layer_size, hidden_layer_size),
                         state_shape, action_shape, device, hidden_layer_size)

    def forward(self, obs, act=None, info={}):
        info.update({'tanh': True})
        return super().forward(obs, act, info)
