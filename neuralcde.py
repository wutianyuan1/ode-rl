import torch
import torchcde
import torch.nn as nn


class CDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = nn.Linear(hidden_channels, 128)
        self.linear2 = nn.Linear(128, input_channels * hidden_channels)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z



class NeuralCDE(nn.Module):
    def __init__(self, input_channels, hidden_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()
        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = nn.Linear(input_channels, hidden_channels)
        self.interpolation = interpolation
        self.readout = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, input_x, prev_z=None, prev_states=None):
        '''
        shape of input_x: [batch_size, sequence_len, channels]
          - note: channels = (dim[0]=time, dim[1:]=data_channels)
        prev_z: this is used for RL inference, where sequence_len = 1
          because we only have the current observation.
          assume we are at time t_i, we have s_{t_i}, and desires the
          policy a_{t_i}, then:
          RNN: $$h_{t_i} = RNN(h_{t_{i-1}}, s_{t_i})$$
               $$a_{t_i} = readout(h_{t_i})$$
          ODE: $$h_{t_i} = h_{t_{i-1}} + \int_{t_{i-1}}^{t_i} f(h, t)dX_t$$
               $$a_{t_i} = readout(h_{t_i})$$
            X_t is an interpolation of s_t from t_{i-1} to t_i
            so the Actor or Critic should maintain a history of states
            in its inference stage.
        '''
        if (prev_states is None) and (input_x.shape[-2] == 1):
            ## beginning of the inference, must padding along time dimension for interpolation
            input_x = torch.cat([torch.zeros_like(input_x), input_x], dim=-2)
        elif (prev_states is not None) and (input_x.shape[-2] == 1):
            ## evaluation phase, cat input and prev states together
            input_x = torch.cat([prev_states, input_x], dim=-2)
        elif (prev_states is None) and (input_x.shape[-2] != 1):
            ## prev_states is None, time dim != 1 --> training phase
            pass
        else:
            ## (prev_states is not None) but (input_x.shape[-2] != 1)
            raise "NMSL??? why???"
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(input_x)
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        X0 = X.evaluate(X.interval[0])
        if prev_z is None:
            z0 = self.initial(X0)
        else:
            z0 = prev_z

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval,
                              adjoint=False)
        z_T = z_T[:, 1]
        return self.readout(z_T), z_T

