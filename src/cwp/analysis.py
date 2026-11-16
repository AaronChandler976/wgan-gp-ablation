"""Extra statistics for the report, written to results/v2/summary/. Reads finished runs only.

    python -m cwp.analysis effects              # factorial effects with intervals and Holm correction
    python -m cwp.analysis pooled               # the same effects across datasets, and whether they differ
    python -m cwp.analysis autoscale            # provisioning cost of each configuration's forecasts
    python -m cwp.analysis effects --exp NAME   # any command on a v2 experiment (results/v2/NAME/)

Without `--exp` the commands read the reported runs in results/ and never write there: every
output goes to results/v2/summary/ with the source in its file name (`reported` or the
experiment's name), so tables from different setups stay apart.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from . import config, metrics
from .config import RunSpec
from .evaluate import FACTORS

PROVISIONING = ("under_rate", "over_rate", "missing_vm_pct", "idle_vm_pct")
HEADROOMS = np.round(np.arange(0.0, 2.0001, 0.01), 2)      # extra capacity on top of the forecast
UNDER_TARGETS = (5.0, 1.0)                                 # % of intervals allowed to run short


def source_name(exp: str | None) -> str:
    return exp or "reported"


def folder(exp: str | None):
    return config.path("results", "v2", exp) if exp else config.path("results")


def load_runs(exp: str | None) -> pd.DataFrame:
    """One row per finished run with every test metric. Runs with a non-finite MAPE are kept."""
    rows = []
    for f in sorted(folder(exp).glob("*.json")):
        r = json.loads(f.read_text())
        spec = RunSpec.from_run_id(r["run_id"])
        rows.append({"data_id": spec.data_id, "config_id": spec.config_id, "arch": r["arch"],
                     "critic": r["critic"], "optimizer": r["optimizer"], "seed": r["seed"],
                     "run_id": r["run_id"], **r["metrics"]["test"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- factorial effects

def contrasts(df: pd.DataFrame, value: str = "mape") -> pd.DataFrame:
    """Per dataset and seed, each main effect and interaction of the 2x2x2 design on log `value`.

    A main effect is the average log change from switching one part on, over the four settings
    of the other two parts. Only seeds with all 8 configurations finished and finite count."""
    rows, names = [], list(FACTORS)
    for data_id, d in df.groupby("data_id", sort=False):
        table = d.pivot_table(index="seed", columns=["arch", "critic", "optimizer"], values=value)
        table = table[np.isfinite(table).all(axis=1)] if table.shape[1] == 8 else table.iloc[:0]
        logm = np.log(table)
        for k in (1, 2, 3):
            for combo in itertools.combinations(range(3), k):
                sign = np.array([np.prod([1 if col[i] == FACTORS[names[i]][1] else -1 for i in combo])
                                 for col in logm.columns])
                per_seed = (logm.to_numpy() * sign).sum(axis=1) / 4.0
                rows += [{"data_id": data_id, "effect": " x ".join(names[i] for i in combo),
                          "kind": "main effect" if k == 1 else "interaction", "seed": s, "value": v}
                         for s, v in zip(logm.index, per_seed)]
    return pd.DataFrame(rows, columns=["data_id", "effect", "kind", "seed", "value"])


def effect_table(c: pd.DataFrame) -> pd.DataFrame:
    """Mean, 95% interval and p-value (t-test over seeds) of every effect, with Holm corrections.

    `p_holm_dataset` corrects over the 7 effects of one dataset; `p_holm_main` over the main
    effects of all datasets together (the family the report's headline claims come from)."""
    rows = []
    for (data_id, effect, kind), g in c.groupby(["data_id", "effect", "kind"], sort=False):
        v = g["value"].to_numpy()
        n, mean = len(v), float(np.mean(v))
        spread = n > 1 and np.ptp(v) > 0
        half = float(stats.t.ppf(0.975, n - 1) * stats.sem(v)) if spread else np.nan
        rows.append({"data_id": data_id, "effect": effect, "kind": kind, "seeds": n,
                     "log_effect": mean, "change_pct": (np.exp(mean) - 1) * 100,
                     "ci_low_pct": (np.exp(mean - half) - 1) * 100,
                     "ci_high_pct": (np.exp(mean + half) - 1) * 100,
                     "same_sign_seeds": f"{int(max((v > 0).sum(), (v < 0).sum()))}/{n}",
                     "p": float(stats.ttest_1samp(v, 0.0).pvalue) if spread else np.nan,
                     "p_wilcoxon": float(stats.wilcoxon(v).pvalue) if spread and n >= 6 else np.nan})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm_dataset"] = np.nan
    for _, idx in out.groupby("data_id", sort=False).groups.items():
        if out.loc[idx, "p"].notna().all():
            out.loc[idx, "p_holm_dataset"] = metrics.holm(out.loc[idx, "p"].tolist())
    main = out.index[out["kind"] == "main effect"]
    out["p_holm_main"] = np.nan
    if out.loc[main, "p"].notna().all():
        out.loc[main, "p_holm_main"] = metrics.holm(out.loc[main, "p"].tolist())
    return out


def pooled_table(c: pd.DataFrame) -> pd.DataFrame:
    """Each effect across datasets.

    `pooled_*` is a random-effects (DerSimonian-Laird) average of the per-dataset means, so it
    describes "a dataset like these", with an interval that widens when the datasets disagree.
    `p_differs` tests whether the effect is the same on every dataset (Welch t-test for two
    datasets, one-way ANOVA for more): this is the effect-by-dataset interaction."""
    rows = []
    for (effect, kind), g in c.groupby(["effect", "kind"], sort=False):
        groups = [x["value"].to_numpy() for _, x in g.groupby("data_id", sort=False)]
        k = len(groups)
        if k < 2 or min(len(x) for x in groups) < 2:
            continue
        m = np.array([x.mean() for x in groups])
        var = np.array([x.var(ddof=1) / len(x) for x in groups])
        w = 1 / var
        fixed = (w * m).sum() / w.sum()
        q = float((w * (m - fixed) ** 2).sum())
        tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        wr = 1 / (var + tau2)
        est, se = float((wr * m).sum() / wr.sum()), float(np.sqrt(1 / wr.sum()))
        p_diff = (stats.ttest_ind(*groups, equal_var=False).pvalue if k == 2
                  else stats.f_oneway(*groups).pvalue)
        rows.append({"effect": effect, "kind": kind, "datasets": k,
                     "per_dataset_pct": ", ".join(f"{(np.exp(x) - 1) * 100:+.1f}" for x in m),
                     "pooled_pct": (np.exp(est) - 1) * 100,
                     "ci_low_pct": (np.exp(est - 1.96 * se) - 1) * 100,
                     "ci_high_pct": (np.exp(est + 1.96 * se) - 1) * 100,
                     "p_pooled": float(2 * stats.norm.sf(abs(est / se))),
                     "between_dataset_sd_pct": float(np.sqrt(tau2) * 100),
                     "p_differs": float(p_diff)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- autoscaling

def headroom_curve(y, p) -> pd.DataFrame:
    """Provisioning outcome when capacity is the forecast plus a fixed share of headroom."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, None)
    rows = []
    for h in HEADROOMS:
        vms = np.ceil(np.round(p * (1 + h), 6))          # rounding first: 90 x 1.10 is 99, not 99.00000000000001
        rows.append({"headroom_pct": h * 100, "under_rate": float(np.mean(y > vms) * 100),
                     "missing_vm_pct": float(np.clip(y - vms, 0, None).sum() / y.sum() * 100),
                     "idle_vm_pct": float(np.clip(vms - y, 0, None).sum() / y.sum() * 100)})
    return pd.DataFrame(rows)


def headroom_for(curve: pd.DataFrame, target: float) -> dict:
    """The smallest headroom that keeps the share of short intervals at or under `target` %."""
    ok = curve[curve["under_rate"] <= target]
    if ok.empty:
        return {"headroom_pct": np.nan, "idle_vm_pct": np.nan}
    return ok.iloc[0][["headroom_pct", "idle_vm_pct"]].to_dict()


def autoscale_tables(exp: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per run, the headroom curves averaged over seeds). Diverged runs are skipped."""
    runs, curves = [], []
    for r in load_runs(exp).itertuples(index=False):
        f = folder(exp) / "predictions" / f"{r.run_id}.csv"
        if not f.exists() or not np.isfinite(r.mape):
            continue
        pred = pd.read_csv(f)
        if not np.isfinite(pred["y_pred"]).all():
            continue
        curve = headroom_curve(pred["y_true"], pred["y_pred"])
        row = {"data_id": r.data_id, "config_id": r.config_id, "seed": r.seed, "mape": r.mape,
               **{k: getattr(r, k) for k in PROVISIONING}}
        for t in UNDER_TARGETS:
            h = headroom_for(curve, t)
            row[f"headroom_pct_for_{t:g}"] = h["headroom_pct"]
            row[f"idle_vm_pct_at_{t:g}"] = h["idle_vm_pct"]
        runs.append(row)
        curves.append(curve.assign(data_id=r.data_id, config_id=r.config_id, seed=r.seed))
    if not runs:
        return pd.DataFrame(), pd.DataFrame()
    curve = (pd.concat(curves).groupby(["data_id", "config_id", "headroom_pct"], sort=False)
             [["under_rate", "missing_vm_pct", "idle_vm_pct"]].mean().reset_index())
    return pd.DataFrame(runs), curve


# --------------------------------------------------------------------------- commands

def pm(x) -> str:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return ""
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}" if len(x) > 1 else f"{x.mean():.2f}"


def save(name: str, md: list[str], **frames: pd.DataFrame) -> None:
    out = config.path("results", "v2", "summary")
    out.mkdir(parents=True, exist_ok=True)
    for suffix, df in frames.items():
        df.to_csv(out / f"{name}{'' if suffix == 'main' else '_' + suffix}.csv", index=False)
    (out / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"saved to {out / (name + '.md')}")


def need_runs(exp: str | None) -> pd.DataFrame:
    df = load_runs(exp)
    if df.empty:
        sys.exit(f"No finished runs in {folder(exp)}")
    return df


def cmd_effects(exp: str | None) -> None:
    src = source_name(exp)
    t = effect_table(contrasts(need_runs(exp)))
    if t.empty:
        sys.exit("No dataset has all 8 configurations for at least one seed.")
    md = [f"# Factorial effects with intervals and multiple-comparison correction ({src} runs)", "",
          "Effects on log test MAPE of the 2x2x2 design, per dataset. `change_pct` is the average "
          "change in MAPE from switching the part on (negative is better), with a 95% interval from "
          "the seed-to-seed spread. `p` is a t-test over seeds. `p_holm_dataset` is corrected for the "
          "7 effects tested on that dataset, `p_holm_main` for the main effects of all datasets "
          "together. `p_wilcoxon` is given only with 6 or more seeds: with 5 its smallest possible "
          "value is 0.0625, so it cannot reach 0.05. "
          f"Generated by `python -m cwp.analysis effects{' --exp ' + exp if exp else ''}`.", ""]
    for data_id, block in t.groupby("data_id", sort=False):
        md += [f"## {data_id}", "", config.md_table(block.drop(columns=["data_id", "log_effect"]).round(4)), ""]
    save(f"effects_{src}", md, main=t)


def cmd_pooled(exp: str | None) -> None:
    src = source_name(exp)
    c = contrasts(need_runs(exp))
    t = pooled_table(c)
    if t.empty:
        sys.exit("Needs at least 2 datasets with all 8 configurations and 2 seeds.")
    ids = list(dict.fromkeys(c["data_id"]))
    md = [f"# Effects across datasets ({src} runs)", "",
          f"Datasets, in the order of `per_dataset_pct`: {', '.join(ids)}. `pooled_pct` is a "
          "random-effects average of the per-dataset effects (change in test MAPE, %, negative is "
          "better) with a 95% interval; it widens when the datasets disagree. "
          "`between_dataset_sd_pct` is the estimated spread of the true effect from dataset to "
          "dataset. `p_differs` tests whether the effect is the same on every dataset (the "
          "effect-by-dataset interaction): a small value means the effect depends on the workload. "
          "With few datasets the pooled interval is rough; read it together with the per-dataset "
          f"values. Generated by `python -m cwp.analysis pooled{' --exp ' + exp if exp else ''}`.", "",
          config.md_table(t.round(4)), ""]
    save(f"pooled_{src}", md, main=t)


def cmd_autoscale(exp: str | None) -> None:
    src = source_name(exp)
    runs, curve = autoscale_tables(exp)
    if runs.empty:
        sys.exit(f"No runs with saved predictions in {folder(exp)}")
    cols = [c for c in runs.columns if c not in ("data_id", "config_id", "seed")]
    table = runs.groupby(["data_id", "config_id"], sort=False).agg(
        seeds=("seed", "count"), **{c: (c, pm) for c in cols}).reset_index()
    md = [f"# Provisioning cost of the forecasts ({src} runs)", "",
          "One VM per forecast job on the test part, mean ± standard deviation over seeds. "
          "`under_rate`: % of intervals with too few VMs. `missing_vm_pct` / `idle_vm_pct`: VMs "
          "short / unused, as % of all jobs. `headroom_pct_for_T`: the smallest fixed extra capacity "
          "(in steps of 1%) on top of the forecast that keeps the share of short intervals at or "
          "under T%, and `idle_vm_pct_at_T` the unused VMs it costs. The headroom is read off the "
          "test part, so it describes each forecast's trade-off curve; it is not a tuned policy. "
          f"Generated by `python -m cwp.analysis autoscale{' --exp ' + exp if exp else ''}`.", ""]
    for data_id, block in table.groupby("data_id", sort=False):
        md += [f"## {data_id}", "", config.md_table(block.drop(columns="data_id")), ""]
    save(f"autoscale_{src}", md, runs=runs, curve=curve)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["effects", "pooled", "autoscale"])
    ap.add_argument("--exp", help="read results/v2/<EXP>/ instead of the reported runs")
    a = ap.parse_args(argv)
    {"effects": cmd_effects, "pooled": cmd_pooled, "autoscale": cmd_autoscale}[a.command](a.exp)


if __name__ == "__main__":
    main()
