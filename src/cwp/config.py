"""Paths, YAML config loading, command-line overrides and run identifiers."""
from __future__ import annotations

import copy
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

ARCHS = ("lstm", "transformer")
CRITICS = ("none", "wgan-gp")
OPTIMIZERS = ("adam", "madgrad")


def root() -> Path:
    """Repository root. Set CWP_ROOT to point somewhere else (used by the tests)."""
    env = os.environ.get("CWP_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


def path(*parts: str) -> Path:
    return root().joinpath(*parts)


def load_yaml(*parts: str) -> dict:
    with open(path(*parts)) as f:
        return yaml.safe_load(f) or {}


def _set_dotted(cfg: dict, key: str, value) -> None:
    node = cfg
    keys = key.split(".")
    for k in keys[:-1]:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"unknown config key '{key}' (no '{k}')")
        node = node[k]
    if not isinstance(node, dict) or keys[-1] not in node:
        raise KeyError(f"unknown config key '{key}'")
    node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: dict | list[str] | None) -> dict:
    """Apply `key=value` overrides (dotted keys). Unknown keys raise KeyError."""
    cfg = copy.deepcopy(cfg)
    if not overrides:
        return cfg
    if isinstance(overrides, list):
        pairs = {}
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override '{item}' must look like key=value")
            k, v = item.split("=", 1)
            pairs[k.strip()] = yaml.safe_load(v)
        overrides = pairs
    for k, v in overrides.items():
        _set_dotted(cfg, k, v)
    return cfg


def default_config() -> dict:
    return load_yaml("configs", "default.yaml")


def datasets() -> dict:
    return load_yaml("configs", "datasets.yaml")


@dataclass(frozen=True)
class RunSpec:
    """One training run: a dataset, one of the 8 configurations, and a seed."""

    dataset: str
    interval: int
    arch: str
    critic: str
    optimizer: str
    seed: int

    def __post_init__(self):
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got '{self.arch}'")
        if self.critic not in CRITICS:
            raise ValueError(f"critic must be one of {CRITICS}, got '{self.critic}'")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got '{self.optimizer}'")

    @property
    def data_id(self) -> str:
        return f"{self.dataset}-{self.interval}m"

    @property
    def config_id(self) -> str:
        critic = "critic" if self.critic == "wgan-gp" else "nocritic"
        return f"{self.arch}-{critic}-{self.optimizer}"

    @property
    def run_id(self) -> str:
        return f"{self.data_id}__{self.config_id}__s{self.seed}"

    @staticmethod
    def from_run_id(run_id: str) -> "RunSpec":
        data_id, config_id, seed = run_id.split("__")
        dataset, interval = data_id.rsplit("-", 1)
        arch, critic, optimizer = config_id.split("-")
        return RunSpec(dataset, int(interval.rstrip("m")), arch,
                       "wgan-gp" if critic == "critic" else "none", optimizer, int(seed.lstrip("s")))


def tuned_path(spec: RunSpec) -> Path:
    return path("configs", "tuned", f"{spec.data_id}__{spec.config_id}.yaml")


def resolve_config(spec: RunSpec, overrides=None, lr: float | None = None, use_tuned: bool = True):
    """Default config + tuned file (if any) + command-line overrides. Returns (cfg, tuned_used)."""
    cfg = default_config()
    tuned_used = False
    tp = tuned_path(spec)
    if use_tuned and tp.exists():
        tuned = yaml.safe_load(tp.read_text()) or {}
        cfg = apply_overrides(cfg, tuned.get("overrides"))
        if lr is None and tuned.get("lr") is not None:
            lr = float(tuned["lr"])
        tuned_used = True
    cfg = apply_overrides(cfg, overrides)
    if lr is not None:  # one learning rate for whichever optimizer this run uses
        cfg["optim"][spec.optimizer]["lr"] = float(lr)
        cfg["optim"]["adam_with_critic"]["lr"] = float(lr)
    return cfg, tuned_used


def md_table(df) -> str:
    """A pandas DataFrame as a Markdown table (avoids the optional `tabulate` dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(" --- " for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if v is None or v != v else str(v) for v in row) + " |")
    return "\n".join(lines)


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root(), capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "--short", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", "src", "configs"))}


def owner() -> str:
    """Who is running this: CWP_OWNER, else git user.name, else the login name."""
    if os.environ.get("CWP_OWNER"):
        return os.environ["CWP_OWNER"]
    try:
        name = subprocess.run(["git", "config", "user.name"], cwd=root(), capture_output=True,
                              text=True, timeout=10).stdout.strip()
        if name:
            return name
    except Exception:
        pass
    import getpass
    return getpass.getuser()
