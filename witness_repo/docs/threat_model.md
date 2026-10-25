# Threat model (draft for the Phase 1 report)

## What is protected
A daily ground-sensor stream (all-sky surface solar irradiance) that feeds an automated downstream
decision (e.g. a solar-yield estimate, a heat/drought alert or a payout trigger). The stream is
validated against **independent context**: clear-sky irradiance, temperature, humidity, rainfall, wind
(NASA POWER) and, optionally, satellite-derived columns such as a Sentinel-2 cloud fraction or NDVI.

## Attacker levels
| level | capability | example | defence |
|---|---|---|---|
| A1 outsider | can modify or replay records in transit, no key | man-in-the-middle on the ground link | layer 1: HMAC + hash chain + index (provenance) |
| A2 compromised sensor | holds the signing key, signs any value | hijacked gateway, malicious operator | layer 2: consistency with independent context (physics) |
| A3 adaptive | knows the detector, shapes the attack to stay under it | low-and-slow drift, jittered or season-matched replay | measured, not claimed: the detectability frontier |

## Attacks (standard scenario)
step bias (relative, 6-20%), gradual drift (to 8-20%), replay of a 7-sample window. Adaptive variants:
drift duration sweep, replay with added noise, replay from the same season one year earlier.

## Goal and harm
Make a downstream consumer act on a false value without anyone noticing (silent failure). Harm scales
with how long the corruption goes unnoticed, so **detection delay** is reported next to detection rate.

## Trust assumptions (explicit limits)
- The context sources are independent of the sensor and not compromised at the same time. An attacker
  who corrupts sensor *and* context consistently is out of scope.
- Keys are not extracted from the verifier. Key management and rotation are out of scope.
- A very small, very slow bias is physically indistinguishable from natural variability; the frontier
  plots show where that boundary lies instead of hiding it.

## Responses
PASS, DEWEIGHT (trust < 1, continuous), QUARANTINE (alarm, with hysteresis) and verification request.
