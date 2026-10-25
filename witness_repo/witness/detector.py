"""Layer 2 - physics consistency + temporal evidence, fused into a trust score.

Evidence per day (all causal, so it runs as a stream):
  z        relative residual of the reading vs what the independent context predicts
  cusum    two-sided CUSUM of z          -> step bias
  ewma     exponentially weighted z      -> slow drift
  roll7    7-day mean of z               -> short episodes
  replay   closeness to an earlier 7-day window -> exact / near-exact replay
A small monotone gradient-boosted classifier (more evidence never lowers p) fuses the evidence into p = P(compromised); trust = 1 - p/tau.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .attacks import apply_episodes, random_episodes
from .data import CLEAR, TARGET

K_SLACK, LAMBDA, WIN, CAP = 0.5, 0.15, 7, 30.0
MODES = ("witness", "no_twin", "physics_only", "own_history")


def _design_legacy(df):
    doy = df.index.dayofyear.values
    X = df[[c for c in df.columns if c != TARGET]].copy()
    X["s"] = np.sin(2 * np.pi * doy / 365.25)
    X["c"] = np.cos(2 * np.pi * doy / 365.25)
    return X.values


def _design(df, kind="anomaly"):
    """kind='anomaly': context features built to transfer across sites (anomalies against the
    context's own recent past, causal, not only absolute levels). kind='legacy': raw context + season.
    Extra columns in `df` (e.g. Sentinel-2 cloud cover) are passed through untouched."""
    if kind == "legacy":
        return _design_legacy(df)
    doy = df.index.dayofyear.values
    T, RH, W, P, C = df["T2M"], df["RH2M"], df["WS2M"], df["PRECTOTCORR"], df[CLEAR]
    past = lambda x: x.shift(1).rolling(30, min_periods=5).mean()
    cols = [(C / C.rolling(365, min_periods=100).max()).values,
            (T - past(T)).values, (RH - past(RH)).values, (W - past(W)).values,
            np.log1p(P.values), np.log1p(P.shift(1).fillna(0).values),
            RH.diff().fillna(0).values, T.diff().fillna(0).values, RH.values / 100,
            T.values, C.values]
    known = {TARGET, "T2M", "RH2M", "WS2M", "PRECTOTCORR", CLEAR}
    cols += [df[c].values for c in df.columns if c not in known]
    return np.column_stack(cols)


def replay_features(y, yhat=None, L=WIN, hist=2000):
    """Near-duplicate window evidence.
    rep : -log10 of the distance to the closest earlier non-overlapping L-day window (scaled)
    gap : 'twin test'. Natural look-alike windows also have look-alike context (similar weather),
          a replay does not. gap = how much worse the context matches than the values do."""
    n = len(y)
    rep, gap = np.zeros(n), np.zeros(n)
    if n < 2 * L + 1:
        return rep, gap
    W = sliding_window_view(y, L)
    H = sliding_window_view(yhat, L) if yhat is not None else None
    scale = pd.Series(y).expanding(min_periods=30).std().bfill().values
    for t in range(2 * L - 1, n):
        i = t - L + 1
        lo = max(0, i - hist)
        if i - L + 1 <= lo:
            continue
        d = np.abs(W[lo:i - L + 1] - W[i]).mean(axis=1)
        k = int(d.argmin())
        rep[t] = -np.log10(d[k] / scale[t] + 1e-3)
        if H is not None:
            gap[t] = (np.abs(H[lo + k] - H[i]).mean() - d[k]) / scale[t]
    return np.clip(rep, 0, 3), np.clip(gap, 0, 2)


def replay_feature(y, L=WIN, hist=2000):
    return replay_features(y, None, L, hist)[0]


def false_alarms_per_1000(p, tau, sep=7):
    idx = np.flatnonzero(p[200:] > tau)
    events = 0 if len(idx) == 0 else 1 + int((np.diff(idx) > sep).sum())
    return 1000 * events / (len(p) - 200)


def pick_threshold(ps, target_fa):
    """Lowest threshold whose false-alarm events on clean series stay <= target per 1000 days."""
    pooled = np.concatenate([p[200:] for p in ps])
    for tau in np.quantile(pooled, 1 - np.logspace(-3.5, -1.0, 80)[::-1]):
        if np.mean([false_alarms_per_1000(p, tau) for p in ps]) <= target_fa:
            return float(tau)
    return float(pooled.max())


class Witness:
    def __init__(self, mode="witness", target_fa=2.0, seed=0, design="anomaly", long_windows=False):
        assert mode in MODES
        self.mode, self.target_fa, self.seed = mode, target_fa, seed
        self.design, self.long_windows = design, long_windows
        self.use_replay = mode in ("witness", "no_twin", "own_history")
        self.use_twin = mode in ("witness", "own_history")
        self.reg = None

    # ---- expectation of the reading -------------------------------------------------------
    def _yhat(self, df, y):
        if self.mode == "own_history":     # univariate baseline: lagged rolling median of itself
            s = pd.Series(y).shift(WIN).rolling(60, min_periods=20).median()
            return s.bfill().values
        return self.reg.predict(_design(df, self.design)) * df[CLEAR].values

    def _z(self, df, y, yhat=None):
        yhat = self._yhat(df, y) if yhat is None else yhat
        rel = y / np.maximum(yhat, 0.1) - 1.0
        if getattr(self, "noise", None) is None:
            return rel, rel
        s = np.clip(self.noise.predict(self._noise_X(df, yhat)), 0.01, None)
        return rel / (s * self.zscale), rel

    def _noise_X(self, df, yhat):
        return np.column_stack([_design(df, self.design), yhat])

    # ---- streaming evidence ----------------------------------------------------------------
    def features(self, df, y):
        yhat = self._yhat(df, y)
        z, _ = self._z(df, y, yhat)
        n = len(z)
        cp = cm = ew = 0.0
        cus, ewm = np.zeros(n), np.zeros(n)
        for t in range(n):
            cp = min(CAP, max(0.0, cp + z[t] - K_SLACK))
            cm = min(CAP, max(0.0, cm - z[t] - K_SLACK))
            ew = LAMBDA * z[t] + (1 - LAMBDA) * ew
            cus[t], ewm[t] = max(cp, cm), abs(ew) * np.sqrt((2 - LAMBDA) / LAMBDA)
        roll = np.abs(pd.Series(z).rolling(WIN, min_periods=1).mean().values) * np.sqrt(WIN)
        zs = pd.Series(z)
        long = [np.clip(np.abs(zs.rolling(w, min_periods=1).mean().values) * np.sqrt(w), 0, CAP) for w in ((30, 90) if self.long_windows else ())]
        cols = [np.clip(np.abs(z), 0, CAP), cus, np.clip(ewm, 0, CAP), np.clip(roll, 0, CAP)] + long
        if self.use_replay:
            rep, gap = replay_features(y, yhat if self.use_twin else None)
            cols.append(gap) if self.use_twin else None
            cols.append(rep)          # last column: used by the hard near-duplicate rule
        return np.column_stack(cols)

    # ---- training ----------------------------------------------------------------------------
    def fit(self, reg_dfs, noise_dfs, calib_dfs, thr_dfs, n_series=4, kinds=("step", "drift", "replay", "replay") * 2):
        """reg_dfs: sites for the physics model; noise_dfs: disjoint sites for the conditional
        noise model (how wrong is the physics model *under these conditions*); calib_dfs: disjoint
        sites for the fusion model; thr_dfs: disjoint sites where the alarm threshold is set to the
        target false-alarm rate. Test sites are never touched."""
        if self.mode != "own_history":
            X = np.vstack([_design(d, self.design) for d in reg_dfs])
            y = np.concatenate([d[TARGET].values / d[CLEAR].values for d in reg_dfs])
            self.reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                                     min_samples_leaf=80, l2_regularization=5.0,
                                                     random_state=self.seed).fit(X, y)
        self.noise = None
        Xn, an = [], []
        for d in noise_dfs:
            yh = self._yhat(d, d[TARGET].values)
            rel = d[TARGET].values / np.maximum(yh, 0.1) - 1.0
            Xn.append(self._noise_X(d, yh)[200:])
            an.append(np.abs(rel)[200:])
        self.noise = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3,
                                                   min_samples_leaf=80, l2_regularization=5.0,
                                                   random_state=self.seed).fit(np.vstack(Xn), np.concatenate(an))
        self.zscale = 1.0
        zs = np.concatenate([self._z(d, d[TARGET].values)[0][200:] for d in noise_dfs])
        self.zscale = float(1.4826 * np.median(np.abs(zs - np.median(zs))))
        rng = np.random.default_rng(self.seed + 1)
        Xs, ys = [], []
        for d in calib_dfs:
            y0 = d[TARGET].values
            for _ in range(n_series):
                eps = random_episodes(rng, len(y0), list(kinds))
                y, m = apply_episodes(y0, eps, rng)
                Xs.append(self.features(d, y)[200:])
                ys.append(m[200:])
        X, y = np.vstack(Xs), np.concatenate(ys)
        self.scaler = StandardScaler().fit(X)
        self.fusion = HistGradientBoostingClassifier(max_depth=3, max_iter=120, learning_rate=0.08,
                                                     l2_regularization=1.0, monotonic_cst=[1] * X.shape[1],
                                                     random_state=self.seed).fit(self.scaler.transform(X), y)
        # near-duplicate windows essentially never occur in natural data: hard rule above the
        # clean maximum, calibrated on clean calibration sites
        self.r_thr = np.inf
        if self.use_replay:
            fr = np.concatenate([self.features(d, d[TARGET].values)[200:, -1] for d in thr_dfs])
            self.r_thr = float(max(2.0, np.quantile(fr, 0.9995) * 1.05))
        ps = [self.score(d, d[TARGET].values) for d in thr_dfs]
        self.tau = pick_threshold(ps, self.target_fa)
        return self

    def score(self, df, y):
        """p(compromised) for every day."""
        X = self.features(df, y)
        p = self.fusion.predict_proba(self.scaler.transform(X))[:, 1]
        if self.use_replay:
            p = np.where(X[:, -1] > self.r_thr, np.maximum(p, 0.99), p)
        return p

    def trust(self, p):
        """1 = fully trusted, 0 = alarm threshold reached."""
        return np.clip(1 - p / self.tau, 0, 1)

    def actions(self, p, exit_days=3):
        """PASS / DEWEIGHT / QUARANTINE with hysteresis (stay quarantined until p < tau/2)."""
        out, state, calm = [], "PASS", 0
        for v in p:
            if state == "QUARANTINE":
                calm = calm + 1 if v < self.tau / 2 else 0
                if calm >= exit_days:
                    state, calm = "PASS", 0
            elif v >= self.tau:
                state = "QUARANTINE"
            out.append(state if state == "QUARANTINE" else ("DEWEIGHT" if v >= self.tau / 3 else "PASS"))
        return np.array(out)
