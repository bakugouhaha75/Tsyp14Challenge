"""Data access: real NASA POWER daily series, or a physics-flavoured synthetic stand-in.

The attacked stream is TARGET (all-sky surface solar irradiance). The CONTEXT columns are the
independent evidence it must stay consistent with (clear-sky irradiance, temperature, humidity,
rain, wind). To add more context (e.g. a Sentinel-2 cloud fraction or NDVI column) append its
name to CONTEXT and add the column to the DataFrame; nothing else changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "ALLSKY_SFC_SW_DWN"
CLEAR = "CLRSKY_SFC_SW_DWN"
CONTEXT = [CLEAR, "T2M", "RH2M", "PRECTOTCORR", "WS2M"]
POWER_PARAMS = [TARGET] + CONTEXT

# (latitude, longitude): deliberately diverse climates
SITES = {
    "site_a": (36.8, 10.2), "site_b": (41.9, 12.5), "site_c": (30.0, 31.2),
    "site_d": (-1.3, 36.8), "site_e": (6.5, 3.4), "site_f": (40.4, -3.7),
    "site_g": (28.6, 77.2), "site_h": (13.8, 100.5), "site_i": (-23.5, -46.6),
    "site_j": (-33.9, 151.2), "site_k": (39.7, -105.0), "site_l": (59.9, 10.8),
    "site_m": (64.1, -21.9), "site_n": (-34.6, -58.4), "site_o": (35.7, 139.7),
    "site_p": (-26.2, 28.0), "site_q": (55.8, 37.6), "site_r": (19.4, -99.1),
    "site_s": (1.4, 103.8), "site_t": (24.7, 46.7),
}


def parse_power_json(js: dict) -> pd.DataFrame:
    par = js["properties"]["parameter"]
    df = pd.DataFrame({k: pd.Series(v) for k, v in par.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.sort_index().replace(-999, np.nan)
    df = df.interpolate(limit=5, limit_direction="both").dropna()
    return df[POWER_PARAMS]


def download_power(name, lat, lon, start="20190101", end="20231231", cache_dir="data/cache"):
    """Download one point from the NASA POWER daily API (cached as CSV)."""
    path = Path(cache_dir) / f"{name}_{start}_{end}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import requests

    r = requests.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        params=dict(parameters=",".join(POWER_PARAMS), community="AG", longitude=lon,
                    latitude=lat, start=start, end=end, format="JSON"),
        timeout=180,
    )
    r.raise_for_status()
    df = parse_power_json(r.json())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def _ar1(rng, n, phi, sd):
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def synthetic_site(rng, lat, start="2019-01-01", days=1826, s2=False) -> pd.DataFrame:
    """Daily series with realistic couplings: clear-sky geometry, a latent cloud process that
    drives irradiance, humidity, rain and temperature. Only for development/CI: results on this
    data demonstrate that the pipeline works, not how it performs on real measurements."""
    idx = pd.date_range(start, periods=days)
    doy = idx.dayofyear.values
    phi = np.deg2rad(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = (24 * 60 / np.pi) * 0.082 * dr * (ws * np.sin(phi) * np.sin(dec)
                                           + np.cos(phi) * np.cos(dec) * np.sin(ws))
    clear = np.maximum(0.75 * ra / 3.6, 0.2)
    hemi = 1.0 if lat >= 0 else -1.0
    winter = np.cos(2 * np.pi * (doy - 15) / 365) * hemi
    latent = _ar1(rng, days, 0.75, 1.0)
    cloud = 1 / (1 + np.exp(-(1.2 * latent + rng.uniform(-1, 0.5) + 0.5 * winter)))
    allsky = clear * (1 - 0.75 * cloud ** 3.4) * (1 + rng.normal(0, 0.03, days))
    allsky = np.clip(allsky, 0.05, None)
    tmean = 27 - 0.45 * abs(lat) + rng.normal(0, 1.5)
    amp = 2 + 0.25 * abs(lat)
    temp = tmean - amp * winter + 3.0 * (1 - cloud) - 1.5 + _ar1(rng, days, 0.7, 1.6)
    rh = np.clip(65 + 28 * (cloud - 0.4) - 0.8 * (temp - 15) + rng.normal(0, 6, days), 8, 100)
    rain = np.where((cloud > 0.65) & (rng.random(days) > 0.45), rng.gamma(1.5, 4, days) * cloud, 0.0)
    wind = 3 * np.exp(_ar1(rng, days, 0.5, 0.35))
    df = pd.DataFrame({TARGET: allsky, CLEAR: clear, "T2M": temp, "RH2M": rh,
                       "PRECTOTCORR": rain, "WS2M": wind}, index=idx)[POWER_PARAMS]
    if s2:   # Sentinel-2-like scene cloud cover every ~5 days, noisy (scene-level, not point-level)
        from .sentinel2 import add_s2_context
        pick = np.arange(2, days, 5) + rng.integers(-1, 2, len(np.arange(2, days, 5)))
        pick = np.clip(pick, 0, days - 1)
        obs = pd.DataFrame({"S2_CLOUD": np.clip(100 * (cloud[pick] + rng.normal(0, 0.15, len(pick))), 0, 100)},
                           index=idx[pick])
        df = add_s2_context(df, obs[~obs.index.duplicated()])
    return df


def load_sites(source="synthetic", seed=0, n_sites=20, s2=False, **kw):
    """Return {name: DataFrame}. source='synthetic' or 'power' (needs internet)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, (lat, lon) in list(SITES.items())[:n_sites]:
        if source == "synthetic":
            out[name] = synthetic_site(rng, lat, s2=s2, **kw)
        else:
            out[name] = download_power(name, lat, lon)
            if s2:
                from .sentinel2 import add_s2_context, fetch_s2_cloud
                out[name] = add_s2_context(out[name], fetch_s2_cloud(name, lat, lon))
    return out
