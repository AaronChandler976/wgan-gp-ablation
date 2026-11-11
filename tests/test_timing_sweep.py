import torch

from cwp import timing_sweep
from cwp.models import LSTMForecaster


def test_step_loop_computes_the_same_as_the_fused_lstm():
    torch.manual_seed(0)
    for hidden, layers in [(8, 1), (8, 2)]:
        fused = LSTMForecaster(hidden=hidden, layers=layers).eval()
        loop = timing_sweep.LSTMStepLoop.from_lstm(fused).eval()
        x = torch.rand(3, 20)
        with torch.no_grad():
            assert torch.allclose(fused(x), loop(x), atol=1e-5)


def test_sweep_writes_csv_and_markdown(repo):
    timing_sweep.main(["--device", "cpu", "--lengths", "8", "16", "--reps", "2"])
    out = repo / "results" / "summary"
    csv = (out / "timing_sweep_cpu.csv").read_text()
    md = (out / "timing_sweep_cpu.md").read_text()
    assert "lstm step loop" in csv and "transformer" in csv and "cudnn off" not in csv   # GPU only
    assert "| model | size | 8 | 16 |" in md
