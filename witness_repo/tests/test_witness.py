import numpy as np
import pytest

from witness.attacks import Episode, apply_episodes
from witness.data import TARGET, load_sites
from witness.detector import Witness, replay_feature
from witness.provenance import Signer, Verifier

KEY = b"k"


def _records(n=30):
    s = Signer(KEY)
    return [s.sign(i, 5.0 + i) for i in range(n)]


def test_clean_stream_accepted():
    v = Verifier(KEY)
    assert all(v.check(r) is None for r in _records())


def test_tamper_replay_and_gap_detected():
    recs = _records()
    v = Verifier(KEY)
    bad = dict(recs[0]); bad["value"] = 99.0
    assert v.check(bad) == "bad_mac"
    v = Verifier(KEY)
    assert [v.check(r) for r in recs[:5]] == [None] * 5
    assert v.check(recs[2]) == "stale_or_replayed"
    assert v.check(recs[7]) == "gap"


def test_insider_with_key_passes_crypto():
    s, v = Signer(KEY), Verifier(KEY)
    assert all(v.check(s.sign(i, 1e6)) is None for i in range(10))   # forged but correctly signed


def test_attack_magnitudes():
    y0 = np.full(400, 5.0)
    y, m = apply_episodes(y0, [Episode("step", 100, 130, mag=0.12)], np.random.default_rng(0))
    assert np.allclose(y[100:130], 5.6) and m.sum() == 30
    y, _ = apply_episodes(y0, [Episode("drift", 100, 160, mag=0.15)], np.random.default_rng(0))
    assert y[159] == pytest.approx(5.75) and y[100] > 5.0


def test_exact_replay_gives_max_feature():
    rng = np.random.default_rng(0)
    y0 = rng.normal(5, 1, 400)
    y, _ = apply_episodes(y0, [Episode("replay", 300, 307, src=100)], rng)
    assert replay_feature(y)[306] == pytest.approx(3.0)


def test_detector_end_to_end_smoke():
    d = load_sites("synthetic", seed=1, n_sites=6, days=1500)
    s = list(d.values())
    det = Witness("witness").fit(s[:2], s[2:3], s[3:4], s[4:5], n_series=1)
    df = s[5]; y0 = df[TARGET].values
    y, _ = apply_episodes(y0, [Episode("step", 700, 740, mag=0.3)], np.random.default_rng(0))
    p = det.score(df, y)
    assert p.shape == y.shape and 0 <= p.min() and p.max() <= 1
    assert p[700:745].max() > p[250:650].max()


def test_sentinel2_alignment_is_causal_and_marks_staleness():
    import pandas as pd
    from witness.sentinel2 import add_s2_context
    idx = pd.date_range("2020-01-01", periods=40)
    df = pd.DataFrame({TARGET: 1.0}, index=idx)
    s2 = pd.DataFrame({"S2_CLOUD": [10.0, 80.0]}, index=pd.to_datetime(["2020-01-05", "2020-01-10"]))
    out = add_s2_context(df, s2, max_age=15)
    assert np.isnan(out["S2_CLOUD"].iloc[0])                      # nothing observed yet: no look-ahead
    assert out.loc["2020-01-07", "S2_CLOUD"] == 10.0 and out.loc["2020-01-07", "S2_AGE"] == 2
    assert out.loc["2020-01-10", "S2_CLOUD"] == 80.0 and out.loc["2020-01-10", "S2_AGE"] == 0
    assert np.isnan(out.loc["2020-02-05", "S2_CLOUD"])             # older than max_age: marked missing
