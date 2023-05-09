from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union
import numpy as np
import torch
from torch import nn
from tianshou.utils.net.common import MLP


SIGMA_MIN = -20
SIGMA_MAX = 2


class MLPActorProb(nn.Module):
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
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        logits = self.preprocess(obs)
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


class MLPCritic(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            1,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)
            obs = torch.cat([obs, act], dim=1)
        logits, hidden = self.preprocess(obs)
        logits = self.last(logits)
        return logits
