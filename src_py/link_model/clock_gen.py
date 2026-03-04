from __future__ import annotations

import numpy as np

from .clock import Clock


class ClockGen:
    def __init__(
        self,
        freq_hz: float = 16e9,
        sample_freq_hz: float = 16e9 * 16,
        abs_jitter_std_sec: float = 0.01e-12,
    ) -> None:
        self.freq_hz = float(freq_hz)
        self.sample_freq_hz = float(sample_freq_hz)
        self.abs_jitter_std_sec = float(abs_jitter_std_sec)

        self.clk_i = Clock()
        self.clk_q = Clock()

        self._abs_jitter_prev = 0.0
        self._abs_jitter = np.random.randn() * self.abs_jitter_std_sec * self.sample_freq_hz
        self._period_jitter = self._abs_jitter - self._abs_jitter_prev
        self._period = self.nominal_period + self._period_jitter

        self._timer = 0.0
        self._clk_q_pos_edge_timer_val = self._period / 4.0
        self._clk_i_neg_edge_timer_val = self._period / 2.0
        self._clk_q_neg_edge_timer_val = self._period * 3.0 / 4.0
        self._clk_i_pos_edge_timer_val = self._period

    @property
    def nominal_period(self) -> float:
        return self.sample_freq_hz / self.freq_hz

    @staticmethod
    def _set_clock_edge(clk: Clock, is_pos: bool, frac_dly: float, period: float) -> None:
        clk.is_edge = True
        clk.is_pos = is_pos
        clk.frac_dly = frac_dly
        clk.period = period

    @staticmethod
    def _clear_clock(clk: Clock) -> None:
        clk.is_edge = False
        clk.is_pos = False
        clk.frac_dly = 0.0
        clk.period = 0.0

    def run(self) -> None:
        self._timer += 1.0

        if (self._timer - 1) <= self._clk_q_pos_edge_timer_val < self._timer:
            self._set_clock_edge(
                self.clk_q,
                is_pos=True,
                frac_dly=self._clk_q_pos_edge_timer_val - np.floor(self._clk_q_pos_edge_timer_val),
                period=self._period,
            )
        elif (self._timer - 1) <= self._clk_q_neg_edge_timer_val < self._timer:
            self._set_clock_edge(
                self.clk_q,
                is_pos=False,
                frac_dly=self._clk_q_neg_edge_timer_val - np.floor(self._clk_q_neg_edge_timer_val),
                period=self._period,
            )
        else:
            self._clear_clock(self.clk_q)

        if (self._timer - 1) <= self._clk_i_pos_edge_timer_val < self._timer:
            self._set_clock_edge(
                self.clk_i,
                is_pos=True,
                frac_dly=self._clk_i_pos_edge_timer_val - np.floor(self._clk_i_pos_edge_timer_val),
                period=self._period,
            )
        elif (self._timer - 1) <= self._clk_i_neg_edge_timer_val < self._timer:
            self._set_clock_edge(
                self.clk_i,
                is_pos=False,
                frac_dly=self._clk_i_neg_edge_timer_val - np.floor(self._clk_i_neg_edge_timer_val),
                period=self._period,
            )
        else:
            self._clear_clock(self.clk_i)

        if self.clk_i.is_pos_edge:
            self._abs_jitter = np.random.randn() * self.abs_jitter_std_sec * self.sample_freq_hz
            self._period_jitter = self._abs_jitter - self._abs_jitter_prev
            self._abs_jitter_prev = self._abs_jitter
            self._period = self.nominal_period + self._period_jitter

            self._timer = 0.0
            self._clk_i_pos_edge_timer_val = self._period - (1 - self.clk_i.frac_dly)
            self._clk_q_pos_edge_timer_val = self._period / 4.0 - (1 - self.clk_i.frac_dly)
            self._clk_i_neg_edge_timer_val = self._period / 2.0 - (1 - self.clk_i.frac_dly)
            self._clk_q_neg_edge_timer_val = self._period * 3.0 / 4.0 - (1 - self.clk_i.frac_dly)

