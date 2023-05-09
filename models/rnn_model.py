from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn

SIGMA_MIN = -20
SIGMA_MAX = 2


class RecurrentActorProb(nn.Module):
    """Recurrent version of ActorProb.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
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
        print(hidden_layer_size )
        self.device = device
        self.nn = nn.LSTM(
            input_size=int(np.prod(state_shape)),
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        output_dim = int(np.prod(action_shape))
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
        self.nn.flatten_parameters()
        if state is None:
            obs, (hidden, cell) = self.nn(obs)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            obs, (hidden, cell) = self.nn(
                obs, (state["hidden"].transpose(0, 1).contiguous(),
                      state["cell"].transpose(0, 1).contiguous())
            )
        logits = obs[:, -1]
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
        return (mu, sigma), {
            "hidden": hidden.transpose(0, 1).detach(),
            "cell": cell.transpose(0, 1).detach()
        }


class RecurrentCritic(nn.Module):
    """Recurrent version of Critic.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int] = [0],
        device: Union[str, int, torch.device] = "cpu",
        hidden_layer_size: int = 128,
        target: str = 'v'
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(
            input_size=int(np.prod(state_shape)),
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        if target == 'v':
            self.fc2 = nn.Linear(hidden_layer_size, 1)
        elif target == 'q':
            self.fc2 = nn.Linear(hidden_layer_size + int(np.prod(action_shape)), 1)
        else:
            print("only V(s) or Q(s, a) is accepted")
            raise NotImplementedError

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        assert len(obs.shape) == 3
        self.nn.flatten_parameters()
        obs, (hidden, cell) = self.nn(obs)
        obs = obs[:, -1]
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            )
            obs = torch.cat([obs, act], dim=1)
        obs = self.fc2(obs)
        return obs