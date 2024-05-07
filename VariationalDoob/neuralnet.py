"""
A module to approximate functions with neural networks.
"""

import torch
import torch.nn.functional as F
from s4.models.s4.s4 import S4Block as S4
from s4.models.s4.s4d import S4D

if tuple(map(int, torch.__version__.split(".")[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = torch.nn.Dropout
if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 12):
    dropout_fn = torch.nn.Dropout1d
else:
    dropout_fn = torch.nn.Dropout2d


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        layer_widths,
        activate_final=False,
        activation_fn=torch.nn.LeakyReLU(),
    ):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x


class RNN(torch.nn.Module):
    def __init__(self, input_size, config):
        super(RNN, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.num_layers = config["num_layers"]
        self.rnn = torch.nn.RNN(
            input_size,
            self.hidden_size,
            self.num_layers,
            nonlinearity="relu",
            batch_first=True,
        )
        self.fc = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, y):
        y = torch.flip(y, dims=(0,))  # backward information filtering
        if y[0, :].dim() == 1:
            y = y.unsqueeze(0)
        h0 = torch.zeros(self.num_layers, y.size(0), self.hidden_size).to(y.device)
        out, _ = self.rnn(y, h0)
        out = self.fc(out)
        return out


class LSTM(torch.nn.Module):
    def __init__(self, input_size, config):
        super(LSTM, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.num_layers = config["num_layers"]
        self.lstm = torch.nn.LSTM(
            input_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.fc = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, y):
        y = torch.flip(y, dims=(0,))  # backward information filtering
        if y[0, :].dim() == 1:
            y = y.unsqueeze(0)
        h0 = torch.zeros(self.num_layers, y.size(0), self.hidden_size).to(y.device)
        c0 = torch.zeros(self.num_layers, y.size(0), self.hidden_size).to(y.device)
        out, _ = self.lstm(y, (h0, c0))
        out = self.fc(out)
        return out


class S4Model(torch.nn.Module):

    def __init__(self, d_input, config, dropout=0.1, prenorm=False):
        super().__init__()

        self.d_model = config["hidden_size"]
        self.d_output = config["output_size"]
        self.n_layers = config["num_layers"]
        self.prenorm = prenorm

        self.relu = torch.nn.ReLU()

        self.encoder = torch.nn.Linear(d_input, self.d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.s4_layers.append(
                S4D(self.d_model, dropout=dropout, transposed=True, lr=0.001)
            )
            self.norms.append(torch.nn.LayerNorm(self.d_model))
            self.dropouts.append(dropout_fn(dropout))

        self.linearMLP = torch.nn.Linear(self.d_model, self.d_model)

        self.decoder = torch.nn.Linear(self.d_model, self.d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            # if self.prenorm:
            # Prenorm
            # z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Is this RELU on each layer (in what sense)?
            z = z.transpose(-1, -2)
            z = self.relu(self.linearMLP(z))
            z = z.transpose(-1, -2)

            # Dropout on the output of the S4 block
            # z = dropout(z)

            # Residual connection
            # x = z + x
            x = z

            # if not self.prenorm:
            # Postnorm
            # x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Decode the outputs
        x = self.decoder(x)  # (B, L, d_model) -> (B, L, d_output) or (L, d_output)

        if (
            x.dim() == 2
        ):  # if (L, d_output) arises because batch size is 1 pop out a singleton dim 0
            x = x.unsqueeze(0)

        return x


class V_Network(torch.nn.Module):
    def __init__(self, dimension_r, dimension_state, config):
        super().__init__()
        input_dimension = dimension_r + dimension_state
        layers = config["layers"]
        self.net = MLP(input_dimension, layer_widths=layers + [1])

    def forward(self, r, x):
        input = torch.cat((r, x), dim=1)
        out = torch.squeeze(self.net(input))
        return out


class MLP_Z_Network(torch.nn.Module):
    def __init__(self, dimension_r, dimension_state, layers):
        super(MLP_Z_Network, self).__init__()
        input_dimension = dimension_r + dimension_state + 1
        self.net = MLP(input_dimension, layer_widths=layers + [dimension_state])

    def forward(self, r, s, x):
        N = x.shape[0]
        if len(s.shape) == 0:
            s_ = s.repeat((N, 1))
        else:
            s_ = s
        h = torch.cat((r, s_, x), dim=1)
        out = self.net(h)
        return out


class Z_Network(torch.nn.Module):
    def __init__(self, dimension_r, dimension_state, config):
        super().__init__()
        layers = config["layers"]
        self.net = MLP_Z_Network(dimension_r, dimension_state, layers)

    def forward(self, r, s, x):
        out = self.net(r, s, x)
        return out
