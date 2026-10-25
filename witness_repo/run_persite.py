"""Per-site detection (which climates are easy or hard?).  python run_persite.py --source power"""
import argparse, json
from collections import defaultdict
import numpy as np
from witness.data import load_sites, SITES
from witness.detector import Witness
from witness import evaluate as ev

ap = argparse.ArgumentParser(); ap.add_argument("--source", default="synthetic"); a = ap.parse_args()
dfs = load_sites(a.source)
acc = defaultdict(list)
for rot in (0, 4, 8, 12, 16):
    reg, noise, calib, thr, test = ev.split_sites(dfs, rot)
    d = Witness("witness").fit(reg, noise, calib, thr)
    tau = ev.matched_tau(d, test, 2.0)
    for name, df in test:
        r = ev.standard_scenario(d, [(name, df)], np.random.default_rng(3), reps=8, tau=tau)
        acc[name].append(np.mean([r[k]["detect"] for k in ("step", "drift", "replay")]))
    print("rotation", rot, "done", flush=True)
rows = sorted(((np.mean(v), n) for n, v in acc.items()), reverse=True)
out = ["| site (lat, lon) | mean detection (step, drift, replay) |", "|---|---|"]
out += [f"| {n} ({SITES[n][0]:.1f}, {SITES[n][1]:.1f}) | {m:.0%} |" for m, n in rows]
open("results/persite.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
