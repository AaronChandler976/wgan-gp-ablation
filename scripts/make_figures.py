"""Report figures, drawn from the CSV files in results/summary/ and results/v2/.

    python scripts/make_figures.py              # every figure -> figures/*.png and *.pdf
    python scripts/make_figures.py --only mape  # one of: mape, effects, timing (reported runs),
                                                # released, autoscale, forecasts, critic_weight,
                                                # critic_diagnostics (v2; skipped if their input
                                                # files are not there yet)

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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"
V2 = ROOT / "results" / "v2"
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


# --------------------------------------------------------------------------- v2 figures

CONFIG_LABELS = {"lstm-nocritic-adam": "LSTM baseline", "transformer-critic-madgrad": "Paper's final model"}
CONFIG_COLORS = {"lstm-nocritic-adam": BLUE, "transformer-critic-madgrad": ORANGE}


def missing(*paths: Path) -> bool:
    gone = [p for p in paths if not p.exists()]
    if gone:
        print("skipped: no " + ", ".join(str(p.relative_to(ROOT)) for p in gone))
    return bool(gone)


def figure_released() -> None:
    """The paper's Table 3, our reported runs and the released code's settings, side by side."""
    path, ref = V2 / "summary" / "released_code_runs.csv", ROOT / "configs" / "paper_reference.yaml"
    if missing(path, ref):
        return
    import yaml
    runs, paper = pd.read_csv(path), yaml.safe_load(ref.read_text())
    runs["shown"] = runs["mape_released_units"].fillna(runs["mape"])
    setups = [("paper settings (reported runs)", "Our runs, paper settings\n(fair evaluation)"),
              ("released_data", "Released data handling\n(leaks test statistics)"),
              ("released_full", "Full released settings\n(leaks test statistics)")]
    data_ids = sorted(runs["data_id"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(data_ids), figsize=(TEXT_WIDTH, 2.7), sharey=True)
    for ax, data_id in zip(axes, data_ids):
        part = runs[runs["data_id"] == data_id]
        for k, config_id in enumerate(CONFIG_LABELS):
            color, dy = CONFIG_COLORS[config_id], (k - 0.5) * 0.3
            if data_id in paper:
                ax.plot(paper[data_id][k], dy, "*", ms=9, mfc=color, mec="white", mew=0.6, zorder=3)
            for row, (setup, _) in enumerate(setups, start=1):
                values = part[(part["config_id"] == config_id) & (part["setup"] == setup)]["shown"]
                values = values[np.isfinite(values)]
                if values.empty:
                    continue
                ax.plot(values, [row + dy] * len(values), "o", ms=4, mfc=color, mec="white",
                        mew=0.5, alpha=0.5, zorder=2)
                ax.errorbar(values.mean(), row + dy, xerr=values.std(ddof=1), fmt="D", ms=4,
                            mfc=INK, mec="white", mew=0.6, ecolor=INK, elinewidth=1.0, capsize=2.5,
                            zorder=3)
        ax.set_title(dataset_title(data_id, sep="\n"))
        ax.set_xlabel("Test MAPE (%), as each setup computes it")
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(setups) + 1), ["Paper, Table 3"] + [label for _, label in setups])
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker="o", ls="", ms=4.5, mfc=CONFIG_COLORS[c], mec="white",
                      label=f"{CONFIG_LABELS[c]}, one seed") for c in CONFIG_LABELS]
    handles += [Line2D([], [], marker="*", ls="", ms=9, mfc=MUTED, mec="white", label="value in the paper"),
                Line2D([], [], marker="D", ls="-", lw=1.0, ms=4, color=INK, mec="white",
                       label="mean ± std. dev. over seeds")]
    fig.legend(handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(w_pad=1.5)
    save(fig, "released_code_comparison")


def config_style(config_id: str) -> tuple[str, str, str]:
    """(color, line style, label) of one of the 8 configurations."""
    arch, critic, optimizer = config_id.split("-")
    critic = "wgan-gp" if critic == "critic" else "none"
    color = {("lstm", "none"): BLUE, ("lstm", "wgan-gp"): AQUA,
             ("transformer", "none"): ORANGE, ("transformer", "wgan-gp"): VIOLET}[(arch, critic)]
    return (color, "-" if optimizer == "madgrad" else (0, (4, 2)),
            f"{ARCH_LABELS[arch]}, {CRITIC_LABELS[critic]}, {OPTIMIZER_LABELS[optimizer]}")


def figure_autoscale() -> None:
    """Unused capacity against the share of intervals that run short, as headroom is added."""
    path = V2 / "summary" / "autoscale_reported_curve.csv"
    if missing(path):
        return
    curve = pd.read_csv(path)
    data_ids = sorted(curve["data_id"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(data_ids), figsize=(TEXT_WIDTH, 3.0))
    for ax, data_id in zip(axes, data_ids):
        part = curve[(curve["data_id"] == data_id) & (curve["under_rate"] >= 1)]
        for config_id in sorted(part["config_id"].unique()):
            color, ls, label = config_style(config_id)
            line = part[part["config_id"] == config_id]
            ax.plot(line["under_rate"], line["idle_vm_pct"], color=color, ls=ls, lw=1.3, label=label)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        ax.minorticks_off()
        ax.set_title(dataset_title(data_id, sep="\n"))
        ax.set_xlabel("Intervals with too few VMs (%)")
    axes[0].set_ylabel("Unused VMs (% of all jobs)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(w_pad=1.5)
    save(fig, "autoscale_tradeoff")


def figure_forecasts() -> None:
    """Forecast against the true series on the test part: the best and the worst reported run."""
    path = SUMMARY / "all_runs.csv"
    if missing(path):
        return
    runs = pd.read_csv(path)
    runs = runs[np.isfinite(runs["mape"])]
    data_ids = sorted(runs["data_id"].unique(), reverse=True)
    fig, axes = plt.subplots(len(data_ids), 1, figsize=(TEXT_WIDTH, 2.3 * len(data_ids)), squeeze=False)
    for ax, data_id in zip(axes[:, 0], data_ids):
        part = runs[runs["data_id"] == data_id].sort_values("mape")
        truth = None
        for row, color, word in ((part.iloc[0], BLUE, "best"), (part.iloc[-1], ORANGE, "worst")):
            pred = pd.read_csv(ROOT / "results" / "predictions" / f"{data_id}__{row['config_id']}__s{row['seed']}.csv")
            pred = pred.iloc[:288]
            truth = pred if truth is None else truth
            label = config_style(row["config_id"])[2]
            ax.plot(pred["t"], pred["y_pred"], color=color, lw=1.0,
                    label=f"{word} run: {label}, seed {row['seed']} (MAPE {row['mape']:.2f}%)")
        ax.plot(truth["t"], truth["y_true"], color=INK, lw=0.8, alpha=0.7, label="true value", zorder=1)
        ax.set_title(f"{dataset_title(data_id)}: first {len(truth)} test intervals")
        ax.set_ylabel("Jobs per interval")
        ax.legend(loc="upper left", fontsize=7.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[-1, 0].set_xlabel("Interval index in the series")
    fig.tight_layout(h_pad=1.2)
    save(fig, "forecast_best_worst")


def figure_critic_weight() -> None:
    """Critic screen, stage 1: validation MAPE against the weight of the critic term."""
    path = V2 / "critic_s1_weight" / "screen_cells.csv"
    ref_path = V2 / "nocritic_reference" / "screen_cells.csv"
    if missing(path, ref_path):
        return
    cells, ref = pd.read_csv(path), pd.read_csv(ref_path)
    cells["weight"] = cells["variant"].str[1:].astype(float)
    names = list(dict.fromkeys(cells["cell"]))
    fig, axes = plt.subplots(1, len(names), figsize=(TEXT_WIDTH, 2.3))
    floor = cells[cells["weight"] > 0]["weight"].min() / 3          # where weight 0 is drawn
    for ax, name in zip(axes, names):
        data_id, config_id = name.split("__")
        part = cells[cells["cell"] == name].sort_values("weight")
        x = part["weight"].replace(0, floor)
        ax.errorbar(x, part["val_mape"], yerr=part["seed_gap"] / 2, fmt="o-", ms=3.5, lw=1.0,
                    color=VIOLET, mec="white", mew=0.5, elinewidth=0.8, capsize=2)
        level = ref[ref["cell"] == name.replace("-critic-", "-nocritic-")]
        if not level.empty:
            ax.axhline(level["val_mape"].iloc[0], color=INK, lw=0.9, ls=(0, (4, 2)))
        ax.set_xscale("log")
        ax.set_xticks([floor, 0.001, 0.01, 0.1, 1], ["0", "0.001", "0.01", "0.1", "1"])
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.minorticks_off()
        ax.set_title(f"{DATASET_TITLES[data_id.rsplit('-', 1)[0]].split()[0]}, "
                     f"{ARCH_LABELS[config_id.split('-')[0]]}", fontsize=9)
    axes[0].set_ylabel("Validation MAPE (%)")
    fig.supxlabel("Weight of the critic term in the forecaster's loss", fontsize=9, y=0.1)
    handles = [Line2D([], [], color=VIOLET, marker="o", ms=3.5, lw=1.0, mec="white",
                      label="with critic (bar: the two tuning seeds)"),
               Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 2)), label="same model without a critic")]
    fig.legend(handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=0.8)
    save(fig, "critic_weight_screen")


def figure_critic_diagnostics() -> None:
    """What the critic does during training: its gradient next to the L1 gradient, per epoch."""
    import json
    files = sorted((V2 / "critic_diagnostics").glob("*.json")) if (V2 / "critic_diagnostics").exists() else []
    if not files:
        print("skipped: no runs in results/v2/critic_diagnostics/")
        return
    panels = [("adv_to_l1_grad", "Critic gradient /\nL1 gradient", True),
              ("grad_cosine", "Cosine between\nthe two gradients", False),
              ("wasserstein", "Critic's Wasserstein\nestimate", False)]
    fig, axes = plt.subplots(len(panels), len(files), figsize=(TEXT_WIDTH, 4.6), sharex="col", squeeze=False)
    for col, f in enumerate(files):
        run = json.loads(f.read_text())
        diag = pd.DataFrame(run["diagnostics"])
        low_file = V2 / "critic_diagnostics_w0p01" / f.name
        low = pd.DataFrame(json.loads(low_file.read_text())["diagnostics"]) if low_file.exists() else None
        for row, (key, label, log) in enumerate(panels):
            ax = axes[row, col]
            ax.plot(diag["epoch"], diag[key], color=VIOLET, lw=0.8, label="weight 1 (reported)")
            if low is not None:
                ax.plot(low["epoch"], low[key], color=AQUA, lw=0.8, label="weight 0.01")
            if log:
                ax.set_yscale("log")
            else:
                ax.axhline(0, color=MUTED, lw=0.6)
            if col == 0:
                ax.set_ylabel(label)
        axes[0, col].set_title(dataset_title(f"{run['dataset']}-{run['interval']}m", sep="\n"))
        axes[-1, col].set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(w_pad=1.5, h_pad=0.8)
    save(fig, "critic_diagnostics")


FIGURES = {"mape": figure_mape, "effects": figure_effects, "timing": figure_timing,
           "released": figure_released, "autoscale": figure_autoscale, "forecasts": figure_forecasts,
           "critic_weight": figure_critic_weight, "critic_diagnostics": figure_critic_diagnostics}


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
