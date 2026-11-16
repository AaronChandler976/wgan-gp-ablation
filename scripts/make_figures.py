"""Report figures, drawn from the CSV files in results/summary/.

    python scripts/make_figures.py              # all three figures -> figures/*.png and *.pdf
    python scripts/make_figures.py --only mape  # one of: mape, effects, timing

Needs matplotlib (in environment.yml). Nothing is hard-coded: every value is read from
results/summary/all_runs.csv, factorial_effects.csv and timing_sweep_<device>.csv, so re-running
after `python -m cwp.evaluate compare` or `python -m cwp.timing_sweep` updates the figures.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"
OUT = ROOT / "figures"

# Colors (colorblind-safe order) and neutral ink.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd6"

TEXT_WIDTH = 6.5  # inches: a full-width figure in a single-column LaTeX report

DATASET_TITLES = {
    "google": "Google Cluster 2011",
    "azure_functions": "Azure Functions 2019",
}
ARCH_LABELS = {"lstm": "LSTM", "transformer": "Transformer"}
ARCH_COLORS = {"lstm": BLUE, "transformer": ORANGE}
CRITIC_LABELS = {"none": "no critic", "wgan-gp": "critic"}
OPTIMIZER_LABELS = {"adam": "Adam", "madgrad": "MADGRAD"}
EFFECT_LABELS = {
    "arch": "Transformer\n(vs. LSTM)",
    "critic": "WGAN-gp critic\n(vs. none)",
    "optimizer": "MADGRAD\n(vs. Adam)",
}
FAMILY_STYLES = {  # timing sweep: color and label per model family
    "lstm fused": (BLUE, "LSTM, fused"),
    "lstm cudnn off": (AQUA, "LSTM, cuDNN off"),
    "lstm step loop": (VIOLET, "LSTM, step by step"),
    "transformer": (ORANGE, "Transformer"),
}
DEVICE_TITLES = {"cuda": "GPU", "cpu": "CPU"}


def use_report_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,  # embed TrueType fonts so the text stays text in the PDF
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def dataset_title(data_id: str, sep: str = ", ") -> str:
    """'google-10m' -> 'Google Cluster 2011, 10-minute intervals' (sep=", ")."""
    match = re.fullmatch(r"(.+)-(\d+)m", data_id)
    if not match:
        return data_id
    name, minutes = match.groups()
    return f"{DATASET_TITLES.get(name, name)}{sep}{minutes}-minute intervals"


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"{name}.{ext}"
        # no creation date in the PDF, so drawing the same data again gives the same file
        fig.savefig(path, metadata={"CreationDate": None} if ext == "pdf" else None)
        print(f"wrote {path.relative_to(ROOT)}")
    plt.close(fig)


def figure_mape() -> None:
    """Test MAPE of every configuration: one point per seed, mean and standard deviation."""
    runs = pd.read_csv(SUMMARY / "all_runs.csv")
    data_ids = sorted(runs["data_id"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(data_ids), figsize=(TEXT_WIDTH, 3.3), sharey=True)
    order = (runs[["arch", "critic", "optimizer"]].drop_duplicates()
             .sort_values(["arch", "critic", "optimizer"]).itertuples(index=False))
    order = list(order)
    labels = [f"{ARCH_LABELS[a]}, {CRITIC_LABELS[c]}, {OPTIMIZER_LABELS[o]}" for a, c, o in order]
    for ax, data_id in zip(axes, data_ids):
        part = runs[runs["data_id"] == data_id]
        for row, (arch, critic, optimizer) in enumerate(order):
            values = part[(part["arch"] == arch) & (part["critic"] == critic)
                          & (part["optimizer"] == optimizer)]["mape"]
            if values.empty:
                continue
            color = ARCH_COLORS[arch]
            ax.plot(values, [row] * len(values), "o", ms=4.5, mfc=color, mec="white", mew=0.6,
                    alpha=0.55, zorder=2)
            ax.errorbar(values.mean(), row, xerr=values.std(ddof=1), fmt="D", ms=4.5,
                        mfc=INK, mec="white", mew=0.6, ecolor=INK, elinewidth=1.1, capsize=3,
                        zorder=3)
        ax.set_title(dataset_title(data_id, sep="\n"))
        ax.set_xlabel("Test MAPE (%)")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.08)
    axes[0].set_yticks(range(len(order)), labels)
    axes[0].invert_yaxis()
    seeds = runs["seed"].nunique()
    handles = [
        Line2D([], [], marker="o", ls="", ms=4.5, mfc=ARCH_COLORS[arch], mec="white", alpha=0.75,
               label=f"{ARCH_LABELS[arch]}, one seed") for arch in ARCH_COLORS
    ] + [Line2D([], [], marker="D", ls="-", lw=1.1, ms=4.5, color=INK, mec="white",
                label=f"mean ± std. dev. over {seeds} seeds")]
    fig.legend(handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=1.5)
    save(fig, "mape_by_configuration")


def p_label(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def figure_effects() -> None:
    """Average effect of each switch on MAPE (factorial main effects), per dataset."""
    effects = pd.read_csv(SUMMARY / "factorial_effects.csv")
    main = effects[effects["kind"] == "main effect"]
    data_ids = sorted(main["data_id"].unique(), reverse=True)
    names = [name for name in EFFECT_LABELS if name in set(main["effect"])]
    colors = [BLUE, ORANGE, AQUA, VIOLET]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.8, 3.0))
    width = 0.8 / len(data_ids)
    span = main["mape_change_pct"].abs().max()
    for i, data_id in enumerate(data_ids):
        part = main[main["data_id"] == data_id].set_index("effect").reindex(names)
        xs = [j + (i - (len(data_ids) - 1) / 2) * width for j in range(len(names))]
        ax.bar(xs, part["mape_change_pct"], width=width * 0.9, color=colors[i],
               label=dataset_title(data_id), zorder=2)
        for x, change, p in zip(xs, part["mape_change_pct"], part["p"]):
            offset = 0.03 * span * (1 if change >= 0 else -1)
            ax.text(x, change + offset, f"{change:+.1f}%\n{p_label(p)}", ha="center",
                    va="bottom" if change >= 0 else "top", fontsize=7.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=3)
    ax.set_xticks(range(len(names)), [EFFECT_LABELS[name] for name in names])
    ax.set_ylabel("Change in test MAPE (%)\nnegative = lower error")
    ax.set_ylim(-span * 1.45, span * 1.45)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    fig.legend(loc="lower center", ncols=len(data_ids), bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "factorial_main_effects")


def figure_timing() -> None:
    """Inference time against history length on the GPU and the CPU, log scales."""
    paths = sorted(SUMMARY.glob("timing_sweep_*.csv"), reverse=True)  # cuda first, then cpu
    sweep = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    devices = list(dict.fromkeys(sweep["device"]))
    fig, axes = plt.subplots(1, len(devices), figsize=(TEXT_WIDTH, 3.2), sharey=True,
                             squeeze=False)
    axes = axes[0]
    histories = sorted(sweep["history"].unique())
    families = [family for family in FAMILY_STYLES if family in set(sweep["family"])]
    for ax, device in zip(axes, devices):
        part = sweep[sweep["device"] == device]
        for family in families:
            color = FAMILY_STYLES[family][0]
            sizes = list(dict.fromkeys(part[part["family"] == family]["size"]))
            for k, size in enumerate(sizes):  # smaller model solid, larger model dashed
                line = part[(part["family"] == family) & (part["size"] == size)]
                line = line.sort_values("history")
                ax.plot(line["history"], line["ms"], color=color, lw=1.6,
                        ls="-" if k == 0 else (0, (4, 2)), marker="o" if k == 0 else "s",
                        ms=3.5, mec="white", mew=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(histories, [str(h) for h in histories])
        ax.minorticks_off()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(DEVICE_TITLES.get(device, device))
        ax.set_xlabel("History length (time steps)")
    axes[0].set_ylabel("Inference time per prediction (ms)")
    handles = [Line2D([], [], color=FAMILY_STYLES[f][0], lw=1.6, label=FAMILY_STYLES[f][1])
               for f in families]
    handles += [
        Line2D([], [], color=MUTED, lw=1.6, ls="-", marker="o", ms=3.5, mec="white",
               label="smaller model"),
        Line2D([], [], color=MUTED, lw=1.6, ls=(0, (4, 2)), marker="s", ms=3.5, mec="white",
               label="larger model"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(w_pad=1.5)
    save(fig, "timing_sweep")


FIGURES = {"mape": figure_mape, "effects": figure_effects, "timing": figure_timing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(FIGURES), help="draw just one figure")
    args = parser.parse_args()
    use_report_style()
    for name, draw in FIGURES.items():
        if args.only in (None, name):
            draw()


if __name__ == "__main__":
    main()
