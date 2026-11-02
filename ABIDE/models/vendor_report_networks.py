"""Reviewed network classes extracted from the user-provided original archive.
Only class ASTs are retained; executable CLI and preprocessing are excluded.
See vendored_networks_provenance.json.
"""

import torch

import torch.nn as nn

class FCN(nn.Module):

    def __init__(self, input_dim: int, hidden1: int=500, hidden2: int=30, dropout: float=0.5):
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(input_dim, hidden1), nn.Dropout(dropout), nn.ReLU(inplace=True), nn.BatchNorm1d(hidden1))
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.block1(x)
        x = self.relu2(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        return x

class LatentFeatureTransformer(nn.Module):

    def __init__(self, input_dim: int, d_model: int=64, nhead: int=4, num_layers: int=2, dim_feedforward: int=128, dropout: float=0.2, num_latents: int=16, mlp_hidden: int=256, feature_dropout: float=0.05):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError('d_model 必须能被 nhead 整除')
        self.input_dim = int(input_dim)
        self.num_latents = int(num_latents)
        self.feature_dropout = float(feature_dropout)
        self.feature_weight = nn.Parameter(torch.empty(1, self.input_dim, d_model))
        self.feature_bias = nn.Parameter(torch.empty(1, self.input_dim, d_model))
        self.latents = nn.Parameter(torch.empty(1, self.num_latents, d_model))
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_kv = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.latent_norm = nn.LayerNorm(d_model)
        self.mlp_branch = nn.Sequential(nn.LayerNorm(self.input_dim), nn.Linear(self.input_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, d_model), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 2))
        nn.init.normal_(self.feature_weight, std=0.02)
        nn.init.normal_(self.feature_bias, std=0.02)
        nn.init.trunc_normal_(self.latents, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            x = torch.flatten(x, start_dim=1)
        if self.training and self.feature_dropout > 0:
            keep = torch.rand_like(x).ge(self.feature_dropout)
            x_attention = x * keep / (1.0 - self.feature_dropout)
        else:
            x_attention = x
        feature_tokens = x_attention.unsqueeze(-1) * self.feature_weight + self.feature_bias
        latent_tokens = self.latents.expand(x.shape[0], -1, -1)
        (attended, _) = self.cross_attention(self.cross_norm_q(latent_tokens), self.cross_norm_kv(feature_tokens), self.cross_norm_kv(feature_tokens), need_weights=False)
        latent_tokens = latent_tokens + attended
        latent_tokens = self.encoder(latent_tokens)
        attention_readout = self.latent_norm(latent_tokens).mean(dim=1)
        mlp_readout = self.mlp_branch(x)
        return self.head(torch.cat([attention_readout, mlp_readout], dim=1))

class DenoisingAutoencoder(nn.Module):

    def __init__(self, input_size: int, code_size: int):
        super().__init__()
        self.encoder = nn.Linear(input_size, code_size)
        self.decoder = nn.Linear(code_size, input_size)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder(values))

    def forward(self, values: torch.Tensor, corruption: float) -> torch.Tensor:
        if self.training and corruption > 0:
            keep = torch.rand_like(values) >= corruption
            values = values * keep.to(values.dtype)
        return self.decoder(self.encode(values))

class PaperClassifier(nn.Module):

    def __init__(self, input_size: int, ae1_size: int, ae2_size: int, hidden_size: int, demographic_size: int, dropout: float):
        super().__init__()
        self.encoder1 = nn.Linear(input_size, ae1_size)
        self.encoder2 = nn.Linear(ae1_size, ae2_size)
        self.hidden = nn.Linear(ae2_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size + demographic_size, 2)

    def forward(self, connections: torch.Tensor, demographics: torch.Tensor) -> torch.Tensor:
        values = self.dropout(torch.tanh(self.encoder1(connections)))
        values = self.dropout(torch.tanh(self.encoder2(values)))
        values = self.dropout(torch.tanh(self.hidden(values)))
        values = torch.cat([values, demographics], dim=1)
        return self.output(values)
