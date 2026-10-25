# Results (source: power, seed 0, 7 hidden test sites)

## Standard scenario at a matched operating point (2 false alarms / 1000 clean days)

| detector | step bias 6-20% | drift to 8-20% | 7-day replay | false alarms /1000 d |
|---|---|---|---|---|
| Witness (full) | 17% (10 d) | 24% (40 d) | 57% (6 d) | 1.8 |
| Witness without twin test | 18% (8 d) | 27% (38 d) | 46% (6 d) | 1.9 |
| Witness without any replay logic | 24% (7 d) | 24% (54 d) | 35% (3 d) | 1.7 |
| Univariate baseline | 4% (22 d) | 10% (30 d) | 20% (6 d) | 1.9 |

Cells: detection rate (median delay in days from attack start).

## Standard scenario with the deployed threshold (set on held-out calibration sites)

| detector | step bias 6-20% | drift to 8-20% | 7-day replay | false alarms /1000 d |
|---|---|---|---|---|
| Witness (full) | 15% (14 d) | 22% (45 d) | 53% (6 d) | 0.6 |
| Witness without twin test | 7% (20 d) | 10% (55 d) | 31% (6 d) | 0.4 |
| Witness without any replay logic | 30% (11 d) | 32% (51 d) | 43% (1 d) | 3.3 |
| Univariate baseline | 0% (nan d) | 0% (nan d) | 1% (6 d) | 0.0 |

Cells: detection rate (median delay in days from attack start).

## Which layer catches which attacker?

| attack | outsider vs crypto layer | compromised sensor vs crypto layer | consistency layer |
|---|---|---|---|
| step | 100% | 0% | 18% |
| drift | 100% | 0% | 18% |
| replay | 100% | 0% | 46% |

## Overheads

- detector: 0.13 ms/day, peak 0.45 MB
- signing 2.9 us, verifying 2.7 us per record, 177 bytes/record

Chance floor: with ~2 false alarms per 1000 days, a 36-day window is hit by chance ~7% of the time, so detection rates near 10% mean 'not detected'.

![frontier](fig_frontier.png)

![timeline](fig_timeline.png)