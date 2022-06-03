"""
Automatic Cryptocurrency trading using Deep RL
Nick Kaparinos
2022
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class MLP(nn.Module):
    """ Q and V Multi Layer Perceptron (MLP) networks with MLP/LSTM/CNN/Attention timeseries encoders """

    def __init__(self, state_shape, action_shape, n_features, n_previous_timesteps, n_timeseries, encoder_type='LSTM',
                 n_neurons=128, encoder_n_linear_layers=2, q_n_linear_layers=2, v_n_linear_layers=2, n_posmlp_layers=2,
                 n_head_layers=2, n_attention_blocks=1, dueling=True, n_cnn_layers=2, device='cpu') -> None:
        assert q_n_linear_layers >= 1 and v_n_linear_layers >= 1
        super().__init__()
        self.encoder_type = encoder_type
        self.n_features = n_features
        self.n_previous_timesteps = n_previous_timesteps
        self.n_timeseries = n_timeseries
        self.device = device
        self.dueling = dueling
        self.action_dim = int(np.prod(action_shape))

        # TimeseriesEncoder
        self.encoder = TimeseriesEncoder(n_features, n_previous_timesteps, n_timeseries, encoder_type, n_neurons,
                                         encoder_n_linear_layers, n_posmlp_layers, n_head_layers, n_attention_blocks,
                                         n_cnn_layers, device)

        # Q, V Networks
        self.q_network = []
        for i in range(q_n_linear_layers):
            input_dim = n_neurons if i != 0 else self.n_timeseries * n_neurons + self.n_timeseries + 1
            output_dim = n_neurons if i != q_n_linear_layers - 1 else self.action_dim
            self.q_network.append(nn.Linear(input_dim, output_dim))

            if i != q_n_linear_layers - 1:
                self.q_network.append(nn.ReLU())
        self.q_network = nn.Sequential(*self.q_network)

        self.v_network = []
        for i in range(v_n_linear_layers):
            input_dim = n_neurons if i != 0 else self.n_timeseries * n_neurons + self.n_timeseries + 1
            output_dim = n_neurons if i != v_n_linear_layers - 1 else 1
            self.v_network.append(nn.Linear(input_dim, output_dim))

            if i != v_n_linear_layers - 1:
                self.v_network.append(nn.ReLU())
        self.v_network = nn.Sequential(*self.v_network)

    def forward(self, s, state, info):
        """ Mapping: s -> logits. """
        # Timeseries Embeddings
        portfolio_state = torch.as_tensor(s[:, -(self.n_timeseries + 1):], device=self.device, dtype=torch.float32)
        s = torch.as_tensor(s[:, :-(self.n_timeseries + 1)].reshape(s.shape[0], self.n_previous_timesteps + 1, -1),
                            device=self.device, dtype=torch.float32)  # type: ignore
        timeseries_embeddings = self.encoder(s)

        # Heads
        input_features = torch.hstack([portfolio_state, timeseries_embeddings])
        q = self.q_network(input_features)
        if self.dueling:
            v = self.v_network(input_features)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        return logits, state


class TimeseriesEncoder(nn.Module):
    """ Encoder networks for each timeseries """

    def __init__(self, n_features, n_previous_timesteps, n_timeseries, encoder_type='LSTM', n_neurons=128,
                 encoder_n_linear_layers=2, n_posmlp_layers=2, n_head_layers=2, n_attention_blocks=1, n_cnn_layers=2,
                 device='cpu') -> None:
        super().__init__()
        self.n_features = n_features
        self.n_previous_timesteps = n_previous_timesteps
        self.n_timeseries = n_timeseries
        self.device = device

        # Encoder network for each timeseries
        self.encoder = nn.ModuleList()
        for i in range(n_timeseries):
            if encoder_type == 'LSTM':
                self.encoder += [LSTMWrapper(n_features, n_neurons)]
            elif encoder_type == 'MLP':
                self.encoder = [nn.Flatten(), nn.Linear(self.n_features * (self.n_previous_timesteps + 1), n_neurons),
                                nn.ReLU()]
                for _ in range(encoder_n_linear_layers - 1):
                    self.encoder.extend([nn.Linear(n_neurons, n_neurons), nn.ReLU()])
                self.encoder += [nn.Sequential(*self.encoder)]
            elif encoder_type == 'Attention':
                self.encoder += [AttentionEncoder(n_features, n_previous_timesteps, n_neurons, n_posmlp_layers,
                                                  n_head_layers, n_attention_blocks)]
            elif encoder_type == 'CNN':
                self.encoder += [CNNEncoder(n_features, n_previous_timesteps, n_neurons, n_cnn_layers)]
            else:
                raise ValueError(f'TimeseriesEncoder type {encoder_type} not supported!')

    def forward(self, x):
        """ Encoder each timeseries and return the embeddings """
        timeseries_embeddings = torch.empty(0, device=self.device, dtype=torch.float32)
        for i in range(self.n_timeseries):
            temp_embeddings = self.encoder[i](x[:, :, self.n_features * i:self.n_features * (i + 1)])  # noqa
            timeseries_embeddings = torch.cat((timeseries_embeddings, temp_embeddings), dim=1)

        return timeseries_embeddings


class AttentionEncoder(nn.Module):
    """ Attention Encoder consisting of Attention Block(s) and an MLP head """

    def __init__(self, n_features, n_previous_timesteps, n_neurons=128, n_posmlp_layers=2, n_head_layers=2,
                 n_attention_blocks=1) -> None:
        assert n_head_layers >= 1
        super().__init__()
        self.attention_blocks = [AttentionBlock(n_features, n_previous_timesteps, n_neurons, n_posmlp_layers) for _ in
                                 range(n_attention_blocks)]
        self.attention_blocks = nn.Sequential(*self.attention_blocks)

        self.mlp_head = nn.ModuleList([nn.Flatten(),
                                       nn.Linear(n_features * (n_previous_timesteps + 1), n_neurons),
                                       nn.ReLU()])
        for _ in range(n_head_layers - 1):
            self.mlp_head.extend([nn.Linear(n_neurons, n_neurons), nn.ReLU()])
        self.mlp_head = nn.Sequential(*self.mlp_head)
        self.pos_encoder = PositionalEncoding(n_features)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.attention_blocks(x)
        x = self.mlp_head(x)
        return x


class AttentionBlock(nn.Module):
    """ Attention Encoder block consisting of Multi head Attention, LayerNorm and Position Wise MLP network
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, n_features, n_previous_timesteps, n_neurons=128, n_posmlp_layers=2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(n_features, 1)
        self.norm1 = nn.LayerNorm([n_previous_timesteps + 1, n_features])
        self.norm2 = nn.LayerNorm([n_previous_timesteps + 1, n_features])
        self.position_wise_mlp = PositionWiseMLP(n_features, n_previous_timesteps, n_neurons, n_posmlp_layers)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.position_wise_mlp(x))
        return x


class PositionWiseMLP(nn.Module):
    """ Position wise MLP network used in transformer block """

    def __init__(self, n_features, n_previous_timesteps, n_neurons=128, n_posmlp_layers=2) -> None:
        assert n_posmlp_layers >= 2
        super().__init__()
        self.n_previous_timesteps = n_previous_timesteps
        self.position_wise_mlp = nn.ModuleList()

        for _ in range(n_previous_timesteps + 1):
            temp_mlp = nn.ModuleList([nn.Linear(n_features, n_neurons), nn.ReLU()])
            for __ in range(n_posmlp_layers - 2):
                temp_mlp.extend([nn.Linear(n_neurons, n_neurons), nn.ReLU()])
            temp_mlp.extend([nn.Linear(n_neurons, n_features), nn.ReLU()])

            self.position_wise_mlp.append(nn.Sequential(*temp_mlp))

    def forward(self, x):
        y = torch.empty(0, dtype=torch.float32)
        for i in range(self.n_previous_timesteps + 1):
            output = self.position_wise_mlp[i](x[:, i, :])
            output = output[:, None, :]
            if i == 0:
                y = output
            else:
                y = torch.cat((y, output), dim=1)
        return y


class LSTMWrapper(nn.Module):
    """ Simple LSTM wrapper class """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)

    def forward(self, x):
        return self.lstm(x)[0][:, -1, :]


class CNNEncoder(nn.Module):
    """ 1D Convolutional timeseries encoder """

    def __init__(self, n_features, n_previous_timesteps, n_neurons=128, n_cnn_layers=2):
        super().__init__()
        assert n_cnn_layers >= 2
        self.encoder = [nn.Conv1d(n_features, n_neurons, kernel_size=2), nn.ReLU()]

        for i in range(n_cnn_layers - 1):
            self.encoder.extend([nn.Conv1d(n_neurons, n_neurons, kernel_size=2), nn.ReLU()])
        self.encoder = nn.Sequential(*self.encoder)

        self.encoder_head = nn.Linear((n_previous_timesteps + 1 - n_cnn_layers) * n_neurons, n_neurons)

    def forward(self, x):
        x = self.encoder(x.permute(0, 2, 1))
        x = self.encoder_head(x.view(x.shape[0], -1))
        return x


class PositionalEncoding(nn.Module):
    """ Positional encoding class """

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        return x


class Actor(nn.Module):
    def __init__(self, net, use_softmax=False):
        super(Actor, self).__init__()
        self.net = net
        self.use_softmax = use_softmax

    def forward(self, obs, state=None, info={}):
        x, state = self.net(obs, state, info)
        if self.use_softmax:
            x = F.softmax(x, dim=-1)
        return x, state


class Critic(nn.Module):
    def __init__(self, net):
        super(Critic, self).__init__()
        self.net = net

    def forward(self, obs, act=None, info={}):
        x, state = self.net(obs, None, info)
        return x
