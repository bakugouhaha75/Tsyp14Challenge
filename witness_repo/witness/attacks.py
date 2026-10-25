"""Attack simulator: the three standard attacks plus adaptive variants."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    kind: str            # 'step' | 'drift' | 'replay'
    start: int
    end: int             # exclusive
    mag: float = 0.0     # step: relative bias; drift: final relative bias
    jitter: float = 0.0  # replay: noise added to copied samples, in std units of the series
    src: int = -1        # replay: start index of the copied (clean, earlier) window


def place(rng, n_days, lengths, warmup=200, gap=45):
    """Random non-overlapping start positions (the final test hides site and interval)."""
    spans, starts = [], []
    for L in lengths:
        for _ in range(500):
            s = int(rng.integers(warmup, n_days - L - 5))
            if all(s + L + gap <= a or s >= b + gap for a, b in spans):
                spans.append((s, s + L))
                starts.append(s)
                break
        else:
            raise RuntimeError("could not place episodes")
    return starts


def apply_episodes(y0, episodes, rng):
    """Return (attacked series, boolean mask). A replay copies what the attacker saw earlier, i.e.
    the stream as it had been recorded up to that point (episodes are applied in time order)."""
    y = y0.copy()
    mask = np.zeros(len(y0), bool)
    sd = float(np.std(y0))
    for e in sorted(episodes, key=lambda e: e.start):
        s, t = e.start, e.end
        if e.kind == "step":
            y[s:t] = y0[s:t] * (1 + e.mag)
        elif e.kind == "drift":
            y[s:t] = y0[s:t] * (1 + e.mag * np.linspace(0, 1, t - s + 1)[1:])
        elif e.kind == "replay":
            seg = y[e.src:e.src + (t - s)].copy()
            if e.jitter > 0:
                seg = seg + rng.normal(0, e.jitter * sd, len(seg))
            y[s:t] = np.clip(seg, 0.0, None)
        elif e.kind == "freeze":          # held-out attack: sensor stuck at its last good value
            y[s:t] = y0[s - 1]
        elif e.kind == "oscillate":       # held-out attack: zero-mean 14-day oscillating bias
            y[s:t] = y0[s:t] * (1 + e.mag * np.sin(2 * np.pi * np.arange(t - s) / 14))
        else:
            raise ValueError(e.kind)
        mask[s:t] = True
    return y, mask


def random_episodes(rng, n_days, kinds, replay_len=7):
    """Random episodes of the given kinds with magnitudes spread around the standard scenario."""
    lengths = []
    for k in kinds:
        lengths.append({"step": int(rng.integers(20, 46)), "drift": int(rng.integers(40, 121)),
                        "replay": replay_len}[k])
    starts = place(rng, n_days, lengths)
    eps = []
    for k, L, s in zip(kinds, lengths, starts):
        if k == "step":
            eps.append(Episode(k, s, s + L, mag=float(rng.uniform(0.06, 0.20))))
        elif k == "drift":
            eps.append(Episode(k, s, s + L, mag=float(rng.uniform(0.08, 0.20))))
        else:
            if rng.random() < 0.5 and s >= 365:
                src = s - 365
            else:
                src = int(rng.integers(0, s - L - 30))
            eps.append(Episode(k, s, s + L, jitter=float(rng.uniform(0, 0.1)), src=src))
    return eps
