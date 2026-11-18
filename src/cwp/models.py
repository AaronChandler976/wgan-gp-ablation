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

    def __init__(self, hidden: int = 64, layers: int = 1, dropout: float = 0.0,
                 out_dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(out_dropout) if out_dropout > 0 else None   # None draws no random numbers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x.unsqueeze(-1))
        last = h[:, -1] if self.drop is None else self.drop(h[:, -1])
        return self.out(last).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding added per time step.

    `by="batch"` reproduces the released code, which indexes the encoding by a sample's position
    in its batch: every time step of sample b gets encoding b, so time order is not encoded.
    """

    def __init__(self, d_model: int, max_len: int = 4096, by: str = "time"):
        super().__init__()
        if by not in ("time", "batch"):
            raise ValueError("pos_encoding must be 'time' or 'batch'")
        self.by = by
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : d_model // 2]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [batch, time, d_model]
        if self.by == "batch":
            return x + self.pe[: x.size(0)].unsqueeze(1)
        return x + self.pe[: x.size(1)]


class TransformerForecaster(nn.Module):
    """Encoder-decoder Transformer used as the paper's generator (1 encoder, 1 decoder layer).

    The encoder reads the whole history. The decoder receives the last observed value and
    attends to the encoder output, then a linear layer gives the next value.
    """

    def __init__(self, d_model: int = 64, nhead: int = 4, enc_layers: int = 1, dec_layers: int = 1,
                 dim_ff: int = 2048, dropout: float = 0.1, input_embed: str = "linear",
                 pos_encoding: str = "time"):
        super().__init__()
        if input_embed not in ("linear", "broadcast"):
            raise ValueError("input_embed must be 'linear' or 'broadcast'")
        self.d_model = d_model
        self.embed = nn.Linear(1, d_model) if input_embed == "linear" else None
        self.pos = PositionalEncoding(d_model, by=pos_encoding)
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


def _critic_layers(sizes: list[int], leaky_slope: float, norm: str) -> nn.Sequential:
    """Linear layers of the given sizes with LeakyReLU between them and a linear last layer.

    `norm="spectral"` bounds each layer's largest singular value; `norm="layer"` adds layer
    normalization after each hidden layer. With `norm="none"` the layers are created in the
    same order as before these options existed, so the initial weights are unchanged."""
    if norm not in ("none", "spectral", "layer"):
        raise ValueError("critic norm must be none, spectral or layer")
    layers = []
    for k, (i, o) in enumerate(zip(sizes[:-1], sizes[1:])):
        lin = nn.Linear(i, o)
        layers.append(nn.utils.parametrizations.spectral_norm(lin) if norm == "spectral" else lin)
        if k < len(sizes) - 2:
            if norm == "layer":
                layers.append(nn.LayerNorm(o))
            layers.append(nn.LeakyReLU(leaky_slope))
    return nn.Sequential(*layers)


def _diffs(seq: torch.Tensor) -> torch.Tensor:
    return seq[:, 1:] - seq[:, :-1]


class Critic(nn.Module):
    """WGAN critic: 3 fully connected layers with LeakyReLU, one unbounded score per sequence.

    Its input is a history window followed by the next value (real or predicted). With
    `input="diffs"` it sees the changes between neighbouring values instead of the values.
    """

    def __init__(self, seq_len: int, hidden: int = 64, leaky_slope: float = 0.2,
                 norm: str = "none", input: str = "levels"):
        super().__init__()
        if input not in ("levels", "diffs"):
            raise ValueError("critic input must be levels or diffs")
        self.diffs = input == "diffs"
        self.net = _critic_layers([seq_len - self.diffs, hidden, hidden * 2, 1], leaky_slope, norm)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:   # seq: [batch, window + 1]
        return self.net(_diffs(seq) if self.diffs else seq).squeeze(-1)


class ConditionalCritic(nn.Module):
    """A critic that is handed the history and the next value as two separate inputs.

    The history is summarized by one layer. The next value then enters next to that summary,
    both as it is and as its change from the last observed value, so the part that differs
    between a real and a predicted sequence is not one input among `window + 1`."""

    def __init__(self, window: int, hidden: int = 64, leaky_slope: float = 0.2,
                 norm: str = "none", input: str = "levels"):
        super().__init__()
        if input not in ("levels", "diffs"):
            raise ValueError("critic input must be levels or diffs")
        self.diffs = input == "diffs"
        self.history = _critic_layers([window - self.diffs, hidden, hidden], leaky_slope, norm)
        self.act = nn.LeakyReLU(leaky_slope)
        self.head = _critic_layers([hidden + 2, hidden * 2, 1], leaky_slope, norm)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:   # seq: [batch, window + 1]
        hist, y = seq[:, :-1], seq[:, -1:]
        summary = self.act(self.history(_diffs(hist) if self.diffs else hist))
        return self.head(torch.cat([summary, y, y - hist[:, -1:]], dim=1)).squeeze(-1)


def build_forecaster(arch: str, cfg: dict) -> nn.Module:
    if arch == "lstm":
        return LSTMForecaster(**cfg["model"]["lstm"])
    if arch == "transformer":
        return TransformerForecaster(**cfg["model"]["transformer"])
    raise ValueError(f"unknown arch '{arch}'")


class PointwiseTanhCritic(nn.Module):
    """The released code's critic: the same small network scores every value on its own, with
    Tanh after every layer including the last. Returns one bounded score per value
    [batch, window + 1]; it never sees two values together."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden * 2), nn.Tanh(),
            nn.Linear(hidden * 2, 1), nn.Tanh())

    def forward(self, seq: torch.Tensor) -> torch.Tensor:   # seq: [batch, window + 1]
        return self.net(seq.unsqueeze(-1)).squeeze(-1)


def build_critic(cfg: dict) -> nn.Module:
    c = cfg["model"]["critic"]
    common = {"hidden": c["hidden"], "leaky_slope": c["leaky_slope"], "norm": c["norm"], "input": c["input"]}
    if c["kind"] == "mlp":
        return Critic(seq_len=cfg["data"]["window"] + 1, **common)
    if c["kind"] == "conditional":
        return ConditionalCritic(window=cfg["data"]["window"], **common)
    if c["kind"] == "pointwise_tanh":
        return PointwiseTanhCritic(hidden=c["hidden"])
    raise ValueError(f"unknown critic kind '{c['kind']}'")


def build_optimizer(name: str, params, cfg: dict, with_critic: bool,
                    lr_scale: float = 1.0) -> torch.optim.Optimizer:
    """Adam or MADGRAD. With a critic, Adam uses the paper's WGAN settings if enabled.

    `lr_scale` multiplies the learning rate (used to give the critic its own)."""
    if name == "adam":
        o = dict(cfg["optim"]["adam"])
        awc = cfg["optim"]["adam_with_critic"]
        if with_critic and awc["enabled"]:
            o = {**o, "lr": awc["lr"], "betas": awc["betas"]}
        return torch.optim.Adam(params, lr=o["lr"] * lr_scale, betas=tuple(o["betas"]),
                                weight_decay=o["weight_decay"])
    if name == "madgrad":
        from madgrad import MADGRAD
        o = cfg["optim"]["madgrad"]
        return MADGRAD(params, lr=o["lr"] * lr_scale, momentum=o["momentum"],
                       weight_decay=o["weight_decay"], eps=o["eps"])
    raise ValueError(f"unknown optimizer '{name}'")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
