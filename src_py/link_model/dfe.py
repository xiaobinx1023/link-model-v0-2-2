from __future__ import annotations

import numpy as np
import numpy.typing as npt


class DFE:
    """Simple decision-feedback equalizer.

    feedback = sum(taps[i] * (2*data[n-1-i]-1))

    taps are assumed in volts (or normalized units matching din).
    """

    def __init__(self) -> None:
        self.enabled = False
        self.taps = np.array([], dtype=np.float64)
        self._hist = np.zeros(64, dtype=np.int8)
        self._idx = 0

    def update(self, decision: int) -> None:
        self._hist[self._idx] = 1 if int(decision) else 0
        self._idx = (self._idx + 1) % self._hist.size

    def feedback(self) -> float:
        if not self.enabled:
            return 0.0
        taps = np.asarray(self.taps, dtype=np.float64).reshape(-1)
        if taps.size == 0:
            return 0.0
        n = min(taps.size, self._hist.size)
        # most recent decision is at idx-1
        out = 0.0
        for i in range(n):
            bit = int(self._hist[(self._idx - 1 - i) % self._hist.size])
            sym = 1.0 if bit else -1.0
            out += float(taps[i]) * sym
        return float(out)
