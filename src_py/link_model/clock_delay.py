from __future__ import annotations

from typing import List

from .clock import Clock


class ClockDelay:
    def __init__(self, delay: float = 0.0) -> None:
        self._delay = 0.0
        self.delay = delay
        self.clk_in = Clock()
        self.clk_out = Clock()
        self._timers: List[float] = []
        self._clk_buf: List[Clock] = []

    @property
    def delay(self) -> float:
        return self._delay

    @delay.setter
    def delay(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"Trying to set negative delay: {value}")
        self._delay = float(value)

    def run(self) -> None:
        if self.clk_in.is_edge:
            countdown_time = self.delay + self.clk_in.frac_dly
            self._timers.append(countdown_time)
            self._clk_buf.append(self.clk_in.copy())

        self.clk_out.is_edge = False
        self.clk_out.is_pos = False
        self.clk_out.frac_dly = 0.0
        self.clk_out.period = 0.0

        for i in range(len(self._timers) - 1, -1, -1):
            timer = self._timers[i]
            if timer < 1:
                src = self._clk_buf[i]
                self.clk_out.is_edge = True
                self.clk_out.is_pos = src.is_pos
                self.clk_out.frac_dly = timer
                self.clk_out.period = src.period
                del self._timers[i]
                del self._clk_buf[i]

        self._timers = [x - 1 for x in self._timers]

