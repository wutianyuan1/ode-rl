from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from neuralcde import NeuralCDE
from tianshou.utils.net.common import MLP

SIGMA_MIN = -20
SIGMA_MAX = 2


class NeuralCDEActorProb(nn.Module):
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
        self.device = device
        self.nn = NeuralCDE(
            input_channels=int(np.prod(state_shape)),
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


class NeuralCDECritic(nn.Module):
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
        self.nn = NeuralCDE(
            input_channels=int(np.prod(state_shape)),
            hidden_channels=hidden_layer_size
        )
        self.tanh = nn.Tanh()
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
        obs, hidden = self.nn(obs)
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            )
            obs = torch.cat([obs, act], dim=1)
        obs = self.tanh(obs)
        obs = self.fc2(obs)
        return obs


class MLPActor(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.mu = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
            device=self.device
        )
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(
                input_dim,  # type: ignore
                self.output_dim,
                hidden_sizes,
                device=self.device
            )
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))
        self._max = max_action
        self._unbounded = unbounded

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        if len(obs.shape) == 3:
            ## if there are history trajs, only remain the last frame
            obs = obs[:, -1, :]
        logits, hidden = self.preprocess(obs, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), state


class NerualCDEDQN(nn.Module):
    def __init__(
        self,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        hidden_shape: int,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.cde = NeuralCDE(
            input_channels=int(np.prod(state_shape)),
            hidden_channels=hidden_shape)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_shape, np.prod(action_shape))

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        obs, hidden = self.cde(obs)
        obs = self.tanh(obs)
        obs = self.fc(obs)
        return obs, state


class MLPDQN(nn.Module):
    def __init__(
        self,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        hidden_shape: int,
        device: Union[str, int, torch.device] = "cpu",):
        super(MLPDQN, self).__init__()
        self.device = device
        input_dim = np.prod(state_shape)
        output_dim = np.prod(action_shape)
        self.fc1 = nn.Linear(input_dim, hidden_shape[0])
        self.fc2 = nn.Linear(hidden_shape[0], hidden_shape[1])
        self.fc3 = nn.Linear(hidden_shape[1], output_dim)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        obs = torch.relu(self.fc1(obs))
        obs = torch.relu(self.fc2(obs))
        return self.fc3(obs), state
