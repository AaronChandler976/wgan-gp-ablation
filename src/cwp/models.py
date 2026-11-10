"""The two forecasters and the critic.

Both forecasters map a history window x [batch, window] (scaled values) to the next value
[batch], so the trainer and the critic do not care which architecture they are given.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """LoadDynamics-style predictor: stacked LSTM, then one fully connected output layer."""

    def __init__(self, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x.unsqueeze(-1))
        return self.out(h[:, -1]).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding added per time step."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : d_model // 2]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [batch, time, d_model]
        return x + self.pe[: x.size(1)]


class TransformerForecaster(nn.Module):
    """Encoder-decoder Transformer used as the paper's generator (1 encoder, 1 decoder layer).

    The encoder reads the whole history. The decoder receives the last observed value and
    attends to the encoder output, then a linear layer gives the next value.
    """

    def __init__(self, d_model: int = 64, nhead: int = 4, enc_layers: int = 1, dec_layers: int = 1,
                 dim_ff: int = 2048, dropout: float = 0.1, input_embed: str = "linear"):
        super().__init__()
        if input_embed not in ("linear", "broadcast"):
            raise ValueError("input_embed must be 'linear' or 'broadcast'")
        self.d_model = d_model
        self.embed = nn.Linear(1, d_model) if input_embed == "linear" else None
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        dec = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec, dec_layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, 1)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        # "broadcast" copies the scalar into every feature, as the released code does.
        x = self.embed(x) if self.embed is not None else x.expand(-1, -1, self.d_model)
        return self.pos(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = self._embed(x)
        memory = self.encoder(src)
        tgt = src[:, -1:, :]                      # the last observed value
        return self.out(self.decoder(tgt, memory)).squeeze(-1).squeeze(-1)


class Critic(nn.Module):
    """WGAN critic: 3 fully connected layers with LeakyReLU, one unbounded score per sequence.

    Its input is a history window followed by the next value (real or predicted).
    """

    def __init__(self, seq_len: int, hidden: int = 64, leaky_slope: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden), nn.LeakyReLU(leaky_slope),
            nn.Linear(hidden, hidden * 2), nn.LeakyReLU(leaky_slope),
            nn.Linear(hidden * 2, 1))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:   # seq: [batch, window + 1]
        return self.net(seq).squeeze(-1)


def build_forecaster(arch: str, cfg: dict) -> nn.Module:
    if arch == "lstm":
        return LSTMForecaster(**cfg["model"]["lstm"])
    if arch == "transformer":
        return TransformerForecaster(**cfg["model"]["transformer"])
    raise ValueError(f"unknown arch '{arch}'")


def build_critic(cfg: dict) -> Critic:
    return Critic(seq_len=cfg["data"]["window"] + 1, **cfg["model"]["critic"])


def build_optimizer(name: str, params, cfg: dict, with_critic: bool) -> torch.optim.Optimizer:
    """Adam or MADGRAD. With a critic, Adam uses the paper's WGAN settings if enabled."""
    if name == "adam":
        o = dict(cfg["optim"]["adam"])
        awc = cfg["optim"]["adam_with_critic"]
        if with_critic and awc["enabled"]:
            o = {"lr": awc["lr"], "betas": awc["betas"]}
        return torch.optim.Adam(params, lr=o["lr"], betas=tuple(o["betas"]))
    if name == "madgrad":
        from madgrad import MADGRAD
        o = cfg["optim"]["madgrad"]
        return MADGRAD(params, lr=o["lr"], momentum=o["momentum"],
                       weight_decay=o["weight_decay"], eps=o["eps"])
    raise ValueError(f"unknown optimizer '{name}'")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
