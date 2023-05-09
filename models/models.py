import torch
import torch.nn as nn
import baselines.utils as utils
import numpy as np

from s4.s4 import S4Block as S4
from s4.s4d import S4D
from torch.distributions import Normal
from baselines.baselines import ODEFunc, DiffeqSolver, Decoder, Encoder_z0_RNN
from baselines.baselines import ODEGRU, VanillaGRU, VAEGRU, ExpDecayGRU, DeltaTGRU, LatentODE, VanillaLSTM
from baselines.baselines import BaseRecurrentModel, BaseVAEModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'




class BaselineModel(nn.Module):
    def __init__(self, model_name, d_input, d_output, d_model=400, ode_dim=400,
                 n_layers=1, enc_hidden_to_latent_dim=20, ode_tol=1e-5, eps_decay=0):
        super(BaselineModel, self).__init__()
        gen_ode_func = ODEFunc(ode_func_net=utils.create_net(d_model, d_model, n_layers=2, n_units=ode_dim,
                                                             nonlinear=nn.Tanh)).to(device)
        diffq_solver = DiffeqSolver(gen_ode_func, 'dopri5', odeint_rtol=ode_tol, odeint_atol=ode_tol / 10)

        # encoder
        encoder = Encoder_z0_RNN(d_model, d_input, hidden_to_z0_units=enc_hidden_to_latent_dim,
                                 device=device).to(device)
        z0_prior = Normal(torch.tensor([0.]).to(device), torch.tensor([1.]).to(device))

        # decoder
        decoder = Decoder(d_model, d_output, n_layers=0).to(device)
        self.model_name = model_name

        if model_name == 'gru':
            self.model = VanillaGRU(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                decoder=decoder,
                device=device).to(device)
        elif model_name == 'latentode':
            self.model = LatentODE(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                encoder_z0=encoder,
                decoder=decoder,
                diffeq_solver=diffq_solver,
                z0_prior=z0_prior,
                device=device).to(device)
        elif model_name == 'expdecaygru':
            self.model = ExpDecayGRU(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                decoder=decoder,
                device=device).to(device)
        elif model_name == 'odegru':
            self.model = ODEGRU(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                decoder=decoder,
                diffeq_solver=diffq_solver,
                device=device).to(device)
        elif model_name == 'vaegru':
            self.model = VAEGRU(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                encoder_z0=encoder,
                decoder=decoder,
                z0_prior=z0_prior,
                device=device).to(device)
        elif model_name == 'deltatgru':
            self.model = DeltaTGRU(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                decoder=decoder,
                device=device).to(device)
        elif model_name == 'lstm':
            self.model = VanillaLSTM(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                decoder=decoder,
                n_layers=n_layers,
                device=device).to(device)

    def forward(self, x, times):
        # x: (B, L, D_state)
        if isinstance(self.model, BaseVAEModel):
            next_states, _, _, _ = self.model.predict_next_states(
                x, times.squeeze(-1).to(device),
                torch.full([x.shape[0], ], fill_value=x.shape[1]).to(device))
        elif isinstance(self.model, BaseRecurrentModel):
            next_states, _ = self.model.predict_next_states(x, times.squeeze(-1).to(device))
        else:
            raise Exception("Unknown model:" + self.model_name)
        return next_states

    def __str__(self):
        return self.model_name

    def __repr__(self):
        return self.model_name

    def param_count(self):
        params = 0
        for param in self.model.parameters():
            params += np.prod(param.shape)
        return params
