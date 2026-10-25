# Witness: a data stream that carries its own proof and its own trust score

Track 2 (False Data Injection) prototype. Two layers, one trust score:

1. **Provenance** (`witness/provenance.py`): every reading is HMAC-signed and hash-chained. Catches
   tampering and replay by anyone without the key. It cannot catch a compromised sensor that signs lies.
2. **Consistency** (`witness/detector.py`): each reading is compared with what independent context
   predicts (gradient-boosted physics model, context-dependent noise model), then CUSUM (step bias),
   EWMA and rolling evidence (drift), a near-duplicate-window rule (replay) and a **twin test** are
   fused by a monotone classifier into `P(compromised)`. `trust = 1 - P/tau`, with PASS / DEWEIGHT / QUARANTINE.

**Twin test:** a replay copies values from an earlier window, but the weather around it is different.
A natural look-alike window also has look-alike context; a replay does not. The feature measures how
much worse the context matches than the values do. The ablation in `results/` shows it helps on
replays from the same season (about +10 to +14 points on the sweep), but does not solve them.

The evaluation (`run_experiments.py`) is built around a **detectability frontier**: instead of one
accuracy number it sweeps attack size, drift speed and replay disguise to show where the defence breaks.

## Run
```
pip install -r requirements.txt
python run_experiments.py --source power     # REAL NASA POWER data (internet; ~8 min), writes results/
python run_crosssite.py --source power       # 5 different hidden-site sets (the headline numbers)
python run_persite.py --source power         # which climates are easy or hard
python run_experiments.py                    # same on synthetic data (for development)
pytest -q
```
Outputs: `results/results.md`, `results.json`, `fig_frontier.png`, `fig_timeline.png`.

## Evaluation design
Sites are split five ways with no overlap: physics model | noise model | fusion model | alarm
threshold | hidden test sites. Attack positions are random (the final test hides site and interval).
Detectors are compared at a matched false-alarm rate and also with the deployed threshold.

## Real-data results (NASA POWER, 20 sites, solar irradiance)
Detection depends strongly on which sites are hidden, so a single split is not enough. `run_crosssite.py`
repeats the standard scenario over 5 different sets of 7 hidden sites at ~2 false alarms per 1000 clean days
(numbers vary by about 2 points between machines):

| detector | step bias | drift | replay | held-out: stuck sensor | held-out: oscillating bias | range over site sets |
|---|---|---|---|---|---|---|
| Witness | 51-53% | 47-49% | 55-57% | 32% | 29% | 14% to 82% |
| univariate baseline | 6% | 8-9% | 21% | 6% | 10% | 4% to 35% |

The two held-out attacks (never used in training) show partial generalisation. With Sentinel-2 cloud-cover
context (`--s2`) Witness reached 56% / 54% / 61% on the three standard attacks (one run, modest gain).

`results/persite.md` shows why: detection is 70-80% in clear-sky-dominated climates (Mediterranean, arid,
dry-season monsoon) and falls to 20-30% where clouds dominate and context predicts irradiance poorly
(maritime, high latitude), and to ~4% in an equatorial convective climate. This is the honest limit of
using weather context to validate irradiance, and it is the main thing to say in the report.
`results/results.md` (one split, `--rotate 0`) happens to be the hardest set of sites.

What mattered on real data (measured on development sites, not on test sites):
- Context features that transfer across sites (anomalies against the context's own recent past) raised
  cross-site model R2 from about 0.2 to 0.44, and mean detection from 43% to 51%.
- A 30/90-day evidence window and an own-history offset correction did not help and were dropped.

## Honest status
- Sentinel-2 (`--s2`) ran successfully on real data; the gain was small (see above).
- The attacked variable is solar irradiance, "12% bias" applied multiplicatively. Check the spec book.
- `P(compromised)` is a score trained on a simulated attack prior, not a calibrated probability.
- The deployed threshold targets ~2 false alarms per 1000 clean days on held-out calibration sites; on new
  sites the realised rate varies (0.6 to 4).
- Known weakness (see frontier): replay with added noise. Above ~0.3 standard deviations of noise the copied
  values are indistinguishable from natural variability given this context (detection <20%).
- Anonymity: the repo contains no names, logos or identifying metadata. Keep it that way for Phase 1.
