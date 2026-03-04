from __future__ import annotations

from .clock import Clock


class Driver:
    AVDD = 0.75

    def __init__(self) -> None:
        self.clk = Clock()
        self.in_ = 0
        self.out = 0.0
        self._prev_in = 0

    def run(self) -> None:
        if self.clk.is_edge:
            self.out = (
                self.clk.frac_dly * self._prev_in + (1.0 - self.clk.frac_dly) * self.in_
            ) * self.AVDD
            self._prev_in = self.in_
        else:
            self.out = self._prev_in * self.AVDD

