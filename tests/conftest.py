import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A throwaway copy of the repo layout with the two synthetic series generated."""
    shutil.copytree(REPO / "configs", tmp_path / "configs", ignore=shutil.ignore_patterns("tuned"))
    monkeypatch.setenv("CWP_ROOT", str(tmp_path))
    monkeypatch.setenv("CWP_OWNER", "tester")
    from cwp import data
    data.synthetic()
    return tmp_path


TINY = ["train.epochs=2", "train.eval_every=1", "data.window=12", "model.lstm.hidden=8",
        "model.transformer.d_model=8", "model.transformer.nhead=2", "model.transformer.dim_ff=16",
        "model.critic.hidden=8", "eval.timing_reps=5"]
