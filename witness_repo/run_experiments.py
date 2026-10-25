"""Reproduce every number and figure:  python run_experiments.py [--source synthetic|power]"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from witness.attacks import Episode, apply_episodes
from witness.data import TARGET, load_sites
from witness.detector import Witness
from witness import evaluate as ev

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="synthetic")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--rotate", type=int, default=0, help="rotate which sites are the hidden test sites")
ap.add_argument("--s2", action="store_true", help="add Sentinel-2 cloud-cover context columns")
a = ap.parse_args()
out = Path("results"); out.mkdir(exist_ok=True)
rng = np.random.default_rng(a.seed)

dfs = load_sites(a.source, seed=a.seed, s2=a.s2)
reg, noise, calib, thr, test = ev.split_sites(dfs, a.rotate)
dets = {m: Witness(m, seed=a.seed).fit(reg, noise, calib, thr) for m in ("witness", "no_twin", "physics_only", "own_history")}
print("fitted detectors", flush=True)
R = {"source": a.source, "seed": a.seed, "n_test_sites": len(test)}

TARGET_FA = 2.0   # false alarms per 1000 clean days, used to compare detectors fairly
for d in dets.values():
    d.tau_eval = ev.matched_tau(d, test, TARGET_FA)
reps_std = 4 if a.quick else 12
R["standard_deployed"] = {m: ev.standard_scenario(d, test, np.random.default_rng(a.seed + 7), reps_std)
                          for m, d in dets.items()}
R["standard"] = {m: ev.standard_scenario(d, test, np.random.default_rng(a.seed + 7), reps_std, tau=d.tau_eval)
                 for m, d in dets.items()}
print("standard scenario done", flush=True)
R["layers"] = ev.layer_matrix(dets["witness"], test, np.random.default_rng(a.seed + 8), tau=dets["witness"].tau_eval)
R["overheads"] = ev.overheads(dets["witness"], test)

print("layers/overheads done", flush=True)
# detectability frontier (witness vs univariate baseline)
sweeps = ev.make_sweeps()
R["frontier"] = {}
reps = 1 if a.quick else 2
for name, (mk, vals) in sweeps.items():
    print("frontier", name, flush=True)
    R["frontier"][name] = {m: ev.sweep(dets[m], test, np.random.default_rng(a.seed + 9), mk, vals, reps, tau=dets[m].tau_eval)
                           for m in ("witness", "no_twin", "own_history")}

fig, ax = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
titles = dict(step="Step bias: size (relative)", drift="Drift to 15%: duration (days)",
              replay_random="Replay (random past): jitter (std units)", replay_season="Replay (same season): jitter (std units)")
for a_, name in zip(ax, sweeps):
    for m, c in (("witness", "tab:green"), ("no_twin", "tab:orange"), ("own_history", "tab:red")):
        v = [r["value"] for r in R["frontier"][name][m]]
        d = [r["detect"] for r in R["frontier"][name][m]]
        a_.plot(range(len(v)), d, "o-", color=c, label={"witness": "Witness (with twin test)", "no_twin": "Witness without twin test", "own_history": "univariate baseline"}[m])
    a_.set_xticks(range(len(v))); a_.set_xticklabels(v, fontsize=8)
    a_.set_title(titles[name], fontsize=9); a_.set_ylim(-0.02, 1.02); a_.grid(alpha=.3)
ax[0].set_ylabel("detection rate"); ax[0].legend(fontsize=8)
fig.suptitle("Detectability frontier: where does the defence break?"); fig.tight_layout()
fig.savefig(out / "fig_frontier.png", dpi=140)

# timeline example on one hidden test site
name, df = test[0]; y0 = df[TARGET].values; det = dets["witness"]
eps = [Episode("step", 400, 430, mag=0.12), Episode("drift", 700, 780, mag=0.15),
       Episode("replay", 1100, 1107, src=1100 - 365, jitter=0.01)]
y, m = apply_episodes(y0, eps, np.random.default_rng(1))
p = det.score(df, y); act = det.actions(p)
fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
ax[0].plot(y0, color="gray", lw=.8, label="true"); ax[0].plot(np.where(m, y, np.nan), "r", lw=1.2, label="attacked")
ax[0].set_ylabel("irradiance"); ax[0].legend(loc="upper right")
ax[1].plot(p, "k", lw=.8); ax[1].axhline(det.tau, color="r", ls="--", label="alarm threshold")
ax[1].set_ylabel("P(compromised)"); ax[1].set_yscale("log"); ax[1].legend()
ax[2].plot(det.trust(p), "b", lw=.8); ax[2].fill_between(range(len(act)), 0, 1, where=act == "QUARANTINE", color="r", alpha=.25, label="quarantine")
ax[2].set_ylabel("trust"); ax[2].set_xlabel("day"); ax[2].legend()
ax[0].set_xlim(300, 1250); fig.suptitle("Step bias (12%), drift (to 15%), replay (7 days) on a hidden site"); fig.tight_layout()
fig.savefig(out / "fig_timeline.png", dpi=140)

(out / "results.json").write_text(json.dumps(R, indent=2))
print(json.dumps({k: R[k] for k in ("standard_deployed", "standard", "layers", "overheads")}, indent=2))

# ---- human-readable summary -------------------------------------------------------------------
def row(m, v):
    return (f"| {m} | {v['step']['detect']:.0%} ({v['step']['delay_median']:.0f} d) | "
            f"{v['drift']['detect']:.0%} ({v['drift']['delay_median']:.0f} d) | "
            f"{v['replay']['detect']:.0%} ({v['replay']['delay_median']:.0f} d) | {v['false_alarms_per_1000d']:.1f} |")

names = {"witness": "Witness (full)", "no_twin": "Witness without twin test", "physics_only": "Witness without any replay logic", "own_history": "Univariate baseline"}
md = [f"# Results (source: {R['source']}, seed {R['seed']}, {R['n_test_sites']} hidden test sites)\n"]
if R["source"] == "synthetic":
    md.append("> Synthetic data: shows the pipeline works, NOT real-world performance. Re-run with `--source power`.\n")
for key, title in (("standard", f"Standard scenario at a matched operating point ({TARGET_FA:g} false alarms / 1000 clean days)"),
                   ("standard_deployed", "Standard scenario with the deployed threshold (set on held-out calibration sites)")):
    md += [f"## {title}\n", "| detector | step bias 6-20% | drift to 8-20% | 7-day replay | false alarms /1000 d |",
           "|---|---|---|---|---|"] + [row(names[m], v) for m, v in R[key].items()] + ["", "Cells: detection rate (median delay in days from attack start).\n"]
md += ["## Which layer catches which attacker?\n", "| attack | outsider vs crypto layer | compromised sensor vs crypto layer | consistency layer |", "|---|---|---|---|"]
for k, v in R["layers"].items():
    md.append(f"| {k} | {v['outsider_crypto']:.0%} | {v['insider_crypto']:.0%} | {v['physics']:.0%} |")
o = R["overheads"]
md += ["", "## Overheads\n", f"- detector: {o['detector_ms_per_day']:.2f} ms/day, peak {o['detector_peak_mb']:.2f} MB",
       f"- signing {o['sign_us_per_record']:.1f} us, verifying {o['verify_us_per_record']:.1f} us per record, {o['record_bytes']} bytes/record",
       "", "Chance floor: with ~2 false alarms per 1000 days, a 36-day window is hit by chance ~7% of the time, so detection rates near 10% mean 'not detected'.",
       "", "![frontier](fig_frontier.png)\n\n![timeline](fig_timeline.png)"]
(out / "results.md").write_text("\n".join(md))
