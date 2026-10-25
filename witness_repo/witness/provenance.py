"""Layer 1 - provenance: HMAC-signed, hash-chained records.

Proves *who* produced a reading and that the sequence is complete and in order. It cannot prove
the reading is *true*: a compromised sensor that holds the key signs its lies correctly. That is
why layer 2 (physics consistency, detector.py) exists.
"""
from __future__ import annotations

import hashlib
import hmac

GENESIS = "0" * 64


def _mac(key: bytes, sensor: str, idx: int, value: float, prev: str) -> str:
    msg = f"{sensor}|{idx}|{value:.6f}|{prev}".encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


class Signer:
    def __init__(self, key: bytes, sensor_id: str = "S1"):
        self.key, self.sensor, self.prev = key, sensor_id, GENESIS

    def sign(self, idx: int, value: float) -> dict:
        mac = _mac(self.key, self.sensor, idx, float(value), self.prev)
        rec = {"idx": idx, "value": float(value), "prev": self.prev, "mac": mac}
        self.prev = hashlib.sha256(mac.encode()).hexdigest()
        return rec


class Verifier:
    """Returns None for an accepted record, otherwise the reason it was rejected."""

    def __init__(self, key: bytes, sensor_id: str = "S1"):
        self.key, self.sensor = key, sensor_id
        self.expected, self.prev = 0, GENESIS

    def check(self, rec: dict):
        mac = _mac(self.key, self.sensor, rec["idx"], rec["value"], rec["prev"])
        if not hmac.compare_digest(mac, rec["mac"]):
            return "bad_mac"                      # value or index altered without the key
        if rec["idx"] != self.expected:
            reason = "stale_or_replayed" if rec["idx"] < self.expected else "gap"
            if rec["idx"] > self.expected:        # resync after a genuine gap
                self.expected, self.prev = rec["idx"] + 1, hashlib.sha256(mac.encode()).hexdigest()
            return reason
        if rec["prev"] != self.prev:
            return "chain_break"
        self.expected += 1
        self.prev = hashlib.sha256(mac.encode()).hexdigest()
        return None
