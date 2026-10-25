"""Cross-site robustness: repeat the standard scenario with different sets of hidden test sites.
Detection depends a lot on which sites are hidden, so one split is not enough to report.
    python run_crosssite.py --source power"""
import argparse, json
import numpy as np
from witness.data import load_sites
from witness.detector import Witness
from witness import evaluate as ev

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="synthetic")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--s2", action="store_true")
a = ap.parse_args()
dfs = load_sites(a.source, seed=a.seed, s2=a.s2)
modes = ("witness", "own_history")
R = {m: [] for m in modes}
for rot in (0, 4, 8, 12, 16):
    reg, noise, calib, thr, test = ev.split_sites(dfs, rot)
    for m in modes:
        d = Witness(m, seed=a.seed).fit(reg, noise, calib, thr)
        tau = ev.matched_tau(d, test, 2.0)
        r = ev.standard_scenario(d, test, np.random.default_rng(7), reps=6, tau=tau)
        R[m].append(dict(rotation=rot, step=r["step"]["detect"], drift=r["drift"]["detect"],
                         replay=r["replay"]["detect"], fa=r["false_alarms_per_1000d"],
                         **ev.heldout_scenario(d, test, np.random.default_rng(11), tau=tau)))
        print(rot, m, R[m][-1], flush=True)
json.dump(R, open("results/crosssite.json", "w"), indent=2)
lines = [f"# Cross-site results (source: {a.source}; 5 different sets of hidden test sites, ~2 false alarms/1000 days)\n",
         "| detector | step bias | drift | replay | held-out: stuck sensor | held-out: oscillating bias | range over site sets (standard attacks) |", "|---|---|---|---|---|---|---|"]
for m in modes:
    x = {k: np.array([r[k] for r in R[m]]) for k in ("step", "drift", "replay")}
    h = {k: np.array([r[k] for r in R[m]]) for k in ("freeze", "oscillate")}
    allv = np.concatenate(list(x.values()))
    lines.append(f"| {m} | {x['step'].mean():.0%} | {x['drift'].mean():.0%} | {x['replay'].mean():.0%} | {h['freeze'].mean():.0%} | {h['oscillate'].mean():.0%} | {allv.min():.0%} to {allv.max():.0%} |")
open("results/crosssite.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
