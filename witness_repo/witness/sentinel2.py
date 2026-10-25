"""Sentinel-2 L2A context: per-acquisition cloud cover, aligned to the daily series.

Sentinel-2 revisits a site every ~2-5 days, so the context is *stale* most days. We keep that
honest: the detector sees both the last observation (S2_CLOUD, percent) and how old it is
(S2_AGE, days). Only past observations are used, so the pipeline stays causal.

Real data: `fetch_s2_cloud` queries the public Microsoft Planetary Computer STAC API (metadata
search needs no account or download of imagery; the scene-level `eo:cloud_cover` field is used).
NOTE: written against the documented STAC API but not run against the live service from the build
environment. If it fails, an equivalent query on the Copernicus Data Space STAC works the same way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"


def fetch_s2_cloud(name, lat, lon, start="2019-01-01", end="2023-12-31", cache_dir="data/cache", half_deg=0.05):
    """DataFrame indexed by date with column S2_CLOUD (0-100) for scenes covering the point."""
    path = Path(cache_dir) / f"{name}_s2_{start}_{end}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import requests

    body = {"collections": ["sentinel-2-l2a"], "datetime": f"{start}/{end}", "limit": 500,
            "bbox": [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]}
    rows, url = [], STAC
    for _ in range(40):                                   # follow pagination
        r = requests.post(url, json=body, timeout=120)
        r.raise_for_status()
        js = r.json()
        rows += [(f["properties"]["datetime"][:10], f["properties"].get("eo:cloud_cover"))
                 for f in js.get("features", []) if f["properties"].get("eo:cloud_cover") is not None]
        nxt = next((l for l in js.get("links", []) if l.get("rel") == "next"), None)
        if not nxt:
            break
        url, body = nxt["href"], nxt.get("body", body)
    s2 = pd.DataFrame(rows, columns=["date", "S2_CLOUD"])
    s2["date"] = pd.to_datetime(s2["date"])
    s2 = s2.groupby("date").mean()
    path.parent.mkdir(parents=True, exist_ok=True)
    s2.to_csv(path)
    return s2


def add_s2_context(df: pd.DataFrame, s2: pd.DataFrame, max_age=15) -> pd.DataFrame:
    """Add S2_CLOUD (last observation at or before each day) and S2_AGE (days since it)."""
    out = df.copy()
    s2 = s2.sort_index()
    obs = s2["S2_CLOUD"].reindex(out.index)
    last_date = pd.Series(np.where(obs.notna(), out.index, pd.NaT), index=out.index).ffill()
    out["S2_CLOUD"] = obs.ffill()
    age = (out.index.to_series() - pd.to_datetime(last_date)).dt.days
    stale = age > max_age
    out["S2_AGE"] = age.astype(float)
    out.loc[stale | age.isna(), ["S2_CLOUD", "S2_AGE"]] = np.nan   # trees handle missing values
    return out
