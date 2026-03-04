from __future__ import annotations


class Slicer:
    """1-bit slicer with optional sensitivity (hysteresis).

    sensitivity is interpreted as a symmetric hysteresis band around ref:
      - if vin > ref + sens/2 => output 1
      - if vin < ref - sens/2 => output 0
      - else hold previous
    """

    def __init__(self, sensitivity: float = 0.0) -> None:
        self.sensitivity = float(sensitivity)
        self._prev = 0

    def run(self, vin: float, ref: float) -> int:
        sens = max(0.0, float(self.sensitivity))
        hi = float(ref) + 0.5 * sens
        lo = float(ref) - 0.5 * sens
        if vin > hi:
            self._prev = 1
        elif vin < lo:
            self._prev = 0
        return int(self._prev)
