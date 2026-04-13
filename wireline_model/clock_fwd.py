from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .clock import Clock


@dataclass
class ForwardClockJitter:
    """Forwarded-clock jitter model parameters.

    All delays/jitter are expressed in *samples* (not seconds) to match the discrete-time sim.

    delay_samp:
        Mean delay from TX clock observation point to RX clock input.
    dcd_ui:
        Duty-cycle distortion in UI. Positive increases high time and decreases low time.
    rj_rms_sec:
        Random jitter (Gaussian) RMS in seconds, applied independently per edge.
    psij_amp_sec / psij_freq_hz:
        Periodic (sinusoidal) jitter amplitude/frequency.
    """

    delay_samp: float = 0.0
    dcd_ui: float = 0.0
    rj_rms_sec: float = 0.0
    psij_amp_sec: float = 0.0
    psij_freq_hz: float = 0.0


class ForwardClockPath:
    """Edge-accurate forwarding clock path with delay + RJ + PSI-J + DCD.

    This block is intentionally *edge based*, consistent with ClockDelay/PI.
    """

    def __init__(self, *, sample_rate_hz: float, ui_rate_hz: float, jitter: ForwardClockJitter | None = None) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.ui_rate_hz = float(ui_rate_hz)
        self.jitter = jitter if jitter is not None else ForwardClockJitter()

        self.clk_in = Clock()
        self.clk_out = Clock()

        self._timers: list[float] = []
        self._buf: list[Clock] = []
        self._sample_idx = 0

    @property
    def samples_per_ui(self) -> float:
        return self.sample_rate_hz / self.ui_rate_hz

    def _psij_samp(self) -> float:
        if self.jitter.psij_amp_sec == 0.0 or self.jitter.psij_freq_hz == 0.0:
            return 0.0
        t = self._sample_idx / self.sample_rate_hz
        return (self.jitter.psij_amp_sec * math.sin(2.0 * math.pi * self.jitter.psij_freq_hz * t)) * self.sample_rate_hz

    def _rj_samp(self) -> float:
        if self.jitter.rj_rms_sec == 0.0:
            return 0.0
        return float(np.random.randn()) * self.jitter.rj_rms_sec * self.sample_rate_hz

    def _dcd_samp(self, is_pos: bool) -> float:
        dcd_ui = float(self.jitter.dcd_ui)
        if dcd_ui == 0.0:
            return 0.0
        # Apply +/- DCD/2 to rising/falling edges.
        # Positive dcd_ui -> longer high time: rising edge delayed, falling edge advanced.
        dcd_samp = 0.5 * dcd_ui * self.samples_per_ui
        return dcd_samp if is_pos else -dcd_samp

    def run(self) -> None:
        self._sample_idx += 1

        # Enqueue input edges with jittered delay.
        if self.clk_in.is_edge:
            total_delay = (
                float(self.jitter.delay_samp)
                + float(self.clk_in.frac_dly)
                + self._dcd_samp(bool(self.clk_in.is_pos))
                + self._psij_samp()
                + self._rj_samp()
            )
            # Clamp to non-negative to keep ordering sane.
            total_delay = max(0.0, total_delay)
            self._timers.append(total_delay)
            self._buf.append(self.clk_in.copy())

        # Default no edge.
        self.clk_out.is_edge = False
        self.clk_out.is_pos = False
        self.clk_out.frac_dly = 0.0
        self.clk_out.period = 0.0

        # Fire any matured edge (at most one per sample).
        for i in range(len(self._timers) - 1, -1, -1):
            timer = self._timers[i]
            if timer < 1.0:
                src = self._buf[i]
                self.clk_out.is_edge = True
                self.clk_out.is_pos = src.is_pos
                self.clk_out.frac_dly = timer
                self.clk_out.period = src.period
                del self._timers[i]
                del self._buf[i]
                break

        # Count down outstanding edges.
        self._timers = [x - 1.0 for x in self._timers]
