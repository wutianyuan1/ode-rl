import torch
import torch.nn as nn
from torch.nn import GRU, LSTM
from models.recurrent_base import RecurrentActorProb, RecurrentCritic
from typing import Sequence, Union


class GRUWrapper(GRU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, obs, hiddens=None):
        if hiddens is not None:
            hidden, cell = hiddens
        else:
            hidden, cell = None, None
        out, hidden = super().forward(obs, hidden)
        return out, (hidden, hidden)


class GRUActorProb(RecurrentActorProb):
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
                         GRUWrapper(input_size=hidden_layer_size, hidden_size=hidden_layer_size, num_layers=layer_num, batch_first=True, device=device), 
                         action_shape, hidden_layer_size, max_action, device, unbounded, conditioned_sigma)


class GRUCritic(RecurrentCritic):
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
                         GRUWrapper(input_size=hidden_layer_size, hidden_size=hidden_layer_size, num_layers=layer_num, batch_first=True, device=device),
                         state_shape, action_shape, device, hidden_layer_size)



class LSTMActorProb(RecurrentActorProb):
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
                         LSTM(input_size=hidden_layer_size, hidden_size=hidden_layer_size, num_layers=layer_num, batch_first=True, device=device), 
                         action_shape, hidden_layer_size, max_action, device, unbounded, conditioned_sigma)


class LSTMCritic(RecurrentCritic):
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
                         LSTM(input_size=hidden_layer_size, hidden_size=hidden_layer_size, num_layers=layer_num, batch_first=True, device=device),
                         state_shape, action_shape, device, hidden_layer_size)
