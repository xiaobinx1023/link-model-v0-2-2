from __future__ import annotations

import math

from .clock import Clock
from .clock_delay import ClockDelay


class PI:
    MIN_PHASE_CODE = 0
    MAX_PHASE_CODE = 127
    PHASE_CODE_PER_QUAD = 32

    def __init__(self) -> None:
        self.clk_in_i = Clock()
        self.clk_in_q = Clock()
        self._phase_code = 0
        self.clk_out = Clock()

        self._quadrant_sel = 0
        self._quadrant_phase = 0
        self._clk_delayline = ClockDelay(0.0)
        self._clk_out_state = 0

        self._clk_i = Clock()
        self._clk_q = Clock()

    @property
    def phase_code(self) -> int:
        return self._phase_code

    @phase_code.setter
    def phase_code(self, value: int) -> None:
        value_i = int(value)
        if value_i < self.MIN_PHASE_CODE or value_i > self.MAX_PHASE_CODE:
            raise ValueError(
                f"Phase code must be an integer between {self.MIN_PHASE_CODE} and {self.MAX_PHASE_CODE}."
            )
        self._phase_code = value_i

    @property
    def clk_in_i_bar(self) -> Clock:
        clk = self.clk_in_i.copy()
        clk.is_pos = not clk.is_pos
        return clk

    @property
    def clk_in_q_bar(self) -> Clock:
        clk = self.clk_in_q.copy()
        clk.is_pos = not clk.is_pos
        return clk

    def run(self) -> None:
        if self.clk_in_i.is_pos_edge:
            self._quadrant_sel = math.floor(self._phase_code / self.PHASE_CODE_PER_QUAD)
            self._quadrant_phase = self._phase_code % self.PHASE_CODE_PER_QUAD

        if self._quadrant_sel == 0:
            self._clk_i = self.clk_in_i.copy()
            self._clk_q = self.clk_in_q.copy()
        elif self._quadrant_sel == 1:
            self._clk_i = self.clk_in_q.copy()
            self._clk_q = self.clk_in_i_bar
        elif self._quadrant_sel == 2:
            self._clk_i = self.clk_in_i_bar
            self._clk_q = self.clk_in_q_bar
        else:
            self._clk_i = self.clk_in_q_bar
            self._clk_q = self.clk_in_i.copy()

        delayline_clk_in = Clock()
        delayline_delay = self._clk_delayline.delay
        if self._clk_out_state == 1:
            if self._clk_i.is_neg_edge:
                delayline_delay = (self._quadrant_phase / self.PHASE_CODE_PER_QUAD) * (
                    self._clk_i.period / 4.0
                )
                delayline_clk_in = self._clk_i.copy()
                self._clk_out_state = 0
            if self._clk_q.is_neg_edge:
                delayline_delay = ((self.PHASE_CODE_PER_QUAD - 1 - self._quadrant_phase) / self.PHASE_CODE_PER_QUAD) * (
                    self._clk_q.period / 4.0
                )
                delayline_clk_in = self._clk_q.copy()
                self._clk_out_state = 0
        else:
            if self._clk_i.is_pos_edge:
                delayline_delay = (self._quadrant_phase / self.PHASE_CODE_PER_QUAD) * (
                    self._clk_i.period / 4.0
                )
                delayline_clk_in = self._clk_i.copy()
                self._clk_out_state = 1
            if self._clk_q.is_pos_edge:
                delayline_delay = ((self.PHASE_CODE_PER_QUAD - 1 - self._quadrant_phase) / self.PHASE_CODE_PER_QUAD) * (
                    self._clk_q.period / 4.0
                )
                delayline_clk_in = self._clk_q.copy()
                self._clk_out_state = 1

        self._clk_delayline.delay = delayline_delay
        self._clk_delayline.clk_in = delayline_clk_in
        self._clk_delayline.run()
        self.clk_out = self._clk_delayline.clk_out.copy()

