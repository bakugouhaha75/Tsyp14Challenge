"""Evaluation: standard scenario, detectability frontier, layer matrix, overheads."""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from .attacks import Episode, apply_episodes, place, random_episodes
from .data import TARGET
from .detector import Witness
from .provenance import Signer, Verifier

GRACE = 3


def split_sites(dfs, rotate=0):
    names = list(dfs)
    names = names[rotate:] + names[:rotate]      # rotate which sites are hidden test sites
    # physics model | noise model | fusion model | alarm threshold | hidden test sites
    return ([dfs[n] for n in names[:5]], [dfs[n] for n in names[5:7]], [dfs[n] for n in names[7:10]],
            [dfs[n] for n in names[10:13]], [(n, dfs[n]) for n in names[13:]])


def outcomes(p, tau, eps):
    alarm = p > tau
    res = []
    for e in eps:
        w = np.flatnonzero(alarm[e.start:e.end + GRACE])
        res.append((e.kind, bool(len(w)), float(w[0]) if len(w) else np.nan))
    return res


def fa_per_1000(p, tau, sep=7):
    idx = np.flatnonzero(p[200:] > tau)
    events = 0 if len(idx) == 0 else 1 + int((np.diff(idx) > sep).sum())
    return 1000 * events / (len(p) - 200)


def matched_tau(det, test, target=2.0):
    """Lowest threshold whose false alarms on the (clean) test sites are <= target per 1000 days.
    Used to compare detectors at the same operating point."""
    ps = [det.score(df, df[TARGET].values) for _, df in test]
    pooled = np.concatenate([p[200:] for p in ps])
    for tau in np.quantile(pooled, 1 - np.logspace(-3.5, -1.0, 80)[::-1]):
        if np.mean([fa_per_1000(p, tau) for p in ps]) <= target:
            return float(tau)
    return float(pooled.max())


def standard_scenario(det, test, rng, reps=12, tau=None):
    tau = det.tau if tau is None else tau
    rows, fa = [], []
    for name, df in test:
        y0 = df[TARGET].values
        fa.append(fa_per_1000(det.score(df, y0), tau))
        for _ in range(reps):
            eps = random_episodes(rng, len(y0), ["step", "step", "drift", "drift", "replay", "replay"])
            y, _ = apply_episodes(y0, eps, rng)
            rows += outcomes(det.score(df, y), tau, eps)
    summ = {}
    for k in ("step", "drift", "replay"):
        r = [x for x in rows if x[0] == k]
        d = [x[2] for x in r if x[1]]
        summ[k] = dict(n=len(r), detect=float(np.mean([x[1] for x in r])),
                       delay_median=float(np.median(d)) if d else float("nan"))
    summ["false_alarms_per_1000d"] = float(np.mean(fa))
    return summ


def sweep(det, test, rng, make_ep, values, reps=2, tau=None):
    tau = det.tau if tau is None else tau
    out = []
    for v in values:
        rows = []
        for name, df in test:
            y0 = df[TARGET].values
            for _ in range(reps):
                eps = make_ep(rng, len(y0), v)
                y, _ = apply_episodes(y0, eps, rng)
                rows += outcomes(det.score(df, y), tau, eps)
        d = [x[2] for x in rows if x[1]]
        out.append(dict(value=v, detect=float(np.mean([x[1] for x in rows])),
                        delay=float(np.median(d)) if d else float("nan")))
    return out


def make_sweeps():
    def step_ep(rng, n, v):
        return [Episode("step", s, s + 30, mag=v) for s in place(rng, n, [30] * 6)]
    def drift_ep(rng, n, v):       # reach 15% over v days (slower = stealthier)
        k = max(1, min(6, 900 // (int(v) + 45)))
        return [Episode("drift", s, s + int(v), mag=0.15) for s in place(rng, n, [int(v)] * k)]
    def replay_ep(src_mode):
        def f(rng, n, v):
            eps = []
            for s in place(rng, n, [7] * 6, warmup=400):
                src = s - 365 if src_mode == "season" else int(rng.integers(0, s - 37))
                eps.append(Episode("replay", s, s + 7, jitter=v, src=src))
            return eps
        return f
    return dict(step=(step_ep, [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]),
                drift=(drift_ep, [30, 60, 120, 240, 480]),
                replay_random=(replay_ep("random"), [0.0, 0.01, 0.03, 0.1, 0.3, 0.6]),
                replay_season=(replay_ep("season"), [0.0, 0.01, 0.03, 0.1, 0.3, 0.6]))


def layer_matrix(det, test, rng, key=b"demo-key-not-secret", tau=None):
    tau = det.tau if tau is None else tau
    """Which layer catches which attacker? Outsider (no key) vs compromised sensor (has key)."""
    res = {k: dict(outsider_crypto=[], insider_crypto=[], physics=[]) for k in ("step", "drift", "replay")}
    for name, df in test:
        y0 = df[TARGET].values
        for _ in range(4):
            eps = random_episodes(rng, len(y0), ["step", "drift", "replay"])
            y, _ = apply_episodes(y0, eps, rng)
            sg = Signer(key)
            recs = [sg.sign(i, v) for i, v in enumerate(y0)]
            p = det.score(df, y)
            for e in eps:
                # outsider cannot sign: tampers with values / replays old signed records
                ov = Verifier(key)
                bad = []
                for i, r in enumerate(recs):
                    r2 = dict(r)
                    if e.start <= i < e.end:
                        if e.kind == "replay":
                            r2 = dict(recs[e.src + i - e.start])
                        else:
                            r2["value"] = float(y[i])
                    bad.append(ov.check(r2))
                res[e.kind]["outsider_crypto"].append(any(b for b in bad[e.start:e.end + 1]))
                # insider holds the key: signs the forged values correctly
                sg2, iv, ibad = Signer(key), Verifier(key), []
                for i, v in enumerate(y):
                    ibad.append(iv.check(sg2.sign(i, v)))
                res[e.kind]["insider_crypto"].append(any(b for b in ibad[e.start:e.end + 1]))
                res[e.kind]["physics"].append(bool((p[e.start:e.end + GRACE] > tau).any()))
    return {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in res.items()}


def overheads(det, test, key=b"k"):
    name, df = test[0]
    y = df[TARGET].values
    tracemalloc.start()
    t0 = time.perf_counter()
    det.score(df, y)
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    sg = Signer(key)
    t0 = time.perf_counter()
    recs = [sg.sign(i, v) for i, v in enumerate(y)]
    sign_us = (time.perf_counter() - t0) / len(y) * 1e6
    vf = Verifier(key)
    t0 = time.perf_counter()
    for r in recs:
        vf.check(r)
    ver_us = (time.perf_counter() - t0) / len(y) * 1e6
    return dict(detector_ms_per_day=dt / len(y) * 1e3, detector_peak_mb=peak,
                sign_us_per_record=sign_us, verify_us_per_record=ver_us,
                record_bytes=len(str(recs[0])))


def heldout_scenario(det, test, rng, tau=None, reps=6):
    """Attacks the detector was never trained on: a stuck sensor and a zero-mean oscillating bias."""
    tau = det.tau if tau is None else tau
    rows = []
    for name, df in test:
        y0 = df[TARGET].values
        for _ in range(reps):
            n_f, n_o = 5, 3
            starts = place(rng, len(y0), [10] * n_f + [56] * n_o)
            eps = [Episode("freeze", s, s + 10) for s in starts[:n_f]]
            eps += [Episode("oscillate", s, s + 56, mag=0.15) for s in starts[n_f:]]
            y, _ = apply_episodes(y0, eps, rng)
            rows += outcomes(det.score(df, y), tau, eps)
    return {k: float(np.mean([r[1] for r in rows if r[0] == k])) for k in ("freeze", "oscillate")}
