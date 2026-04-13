from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .clock import Clock
from .data_gen import DataGen, Pattern
from .driver import Driver


class TxFFE2TapQuarterRate:
    """2-tap FFE TX with a quarter-rate *structural* model.

    Behavioral goal (at UI rate):
        y[n] = w0 * b[n] + w1 * b[n-1]

    Quarter-rate structure:
      - Maintain 4 lane state machines (lanes 0..3) selected by ui_idx % 4.
      - Each lane updates its 'current bit' only when selected, i.e. every 4 UIs.
      - The 2-tap FFE is realized using a global 1-UI delay (b[n-1]) so the
        output matches a standard 2-tap FFE.

    This preserves the *link-level* analog behavior while keeping a clear mapping
    to a quarter-rate architecture.
    """

    AVDD = 0.75

    def __init__(self) -> None:
        self.clk = Clock()
        self.data_gen_pattern = Pattern.ALL_ZEROS

        # Tap weights (cursor/post), user sets directly or via controller mapping.
        self.ffe_weights = np.array([1.0, 0.0], dtype=np.float64)
        self.enabled = False

        self.out = 0.0

        self._dg = DataGen()
        self._ui_idx = 0
        self._lane_bits = np.zeros(4, dtype=np.uint8)
        self._b_prev = 0

        # Analog driver for ZOH + fractional delay support.
        self._drv = Driver()
        self._drv_in = 0.0

    def set_ffe_weights(self, weights: npt.ArrayLike) -> None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != 2:
            raise ValueError("FFE2Tap expects exactly 2 weights [w0, w1].")
        self.ffe_weights = w

    def run(self) -> None:
        self._dg.clk = self.clk
        self._dg.pattern = self.data_gen_pattern

        # Advance bits only on *positive* edge: one new bit per UI.
        if self.clk.is_pos_edge:
            self._dg.run()
            b = int(self._dg.out)
            lane = int(self._ui_idx % 4)
            self._lane_bits[lane] = b

            w0, w1 = (float(self.ffe_weights[0]), float(self.ffe_weights[1]))
            y = w0 * b + w1 * self._b_prev
            self._b_prev = b
            self._ui_idx += 1

            # Keep as normalized analog level. Driver scales by AVDD.
            self._drv_in = float(y)

        # Hold output between UI updates.
        self._drv.clk = self.clk
        self._drv.in_ = float(self._drv_in)
        self._drv.run()
        self.out = self._drv.out if self.enabled else 0.0
