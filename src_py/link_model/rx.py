from __future__ import annotations

import logging
import math
import numpy as np
import numpy.typing as npt

from .aperture import ApertureSampler
from .clock import Clock
from .clock_delay import ClockDelay
from .ctle import CTLE
from .dfe import DFE
from .eye_monitor import EyeMonitor
from .slicer import Slicer


class Rx:
    def __init__(self) -> None:
        self.clk = Clock()
        self.din = 0.0
        self.clk_ofst = 0.0
        self.ref = 0.0
        self.pd_out_gain = 0.0

        # RX digital cascade controls.
        self.ctle_en = False
        self.ctle_dc_gain_db = 0.0
        self.ctle_peaking_gain_db = 0.0
        self.ctle_peaking_freq_hz: float | None = None
        self.ctle_zero_freq_hz = np.array([], dtype=np.float64)
        self.ctle_pole_freq_hz = np.array([], dtype=np.float64)
        self.dfe_en = False
        self.dfe_taps = np.array([], dtype=np.float64)
        self.slicer_sensitivity = 0.0
        self.slicer_aperture_ui = 0.0
        self.samples_per_ui = 16
        self.eye_trace_span_ui = 2.0
        self.sample_rate_hz = 256e9

        self.data = 0
        self.pi_code = 0

        self.din_ctle = 0.0
        self.dfe_feedback = 0.0
        self.din_eq = 0.0
        self.din_apertured = 0.0

        self._din_prev = 0.0
        self._data_prev = 0
        self._data_ana = 0.0
        self._edge_prev = 0
        self._edge = 0
        self._edge_ana = 0.0
        self._pd_out = 0
        self._pi_code_acc = 0.0

        self._clk_edge = Clock()
        self._clk_data = Clock()
        self._clk_delayline = ClockDelay(self.clk_ofst)
        self._eye_monitor = EyeMonitor()
        self._ctle = CTLE()
        self._ctle.set_instance_name("rx")
        self._dfe = DFE()
        self._aperture = ApertureSampler(1)
        self._data_slicer = Slicer()
        self._edge_slicer = Slicer()

    def _configure_blocks(self) -> None:
        self._ctle.enabled = bool(self.ctle_en)
        self._ctle.configure(
            sample_rate_hz=float(self.sample_rate_hz),
            dc_gain_db=float(self.ctle_dc_gain_db),
            peaking_gain_db=float(self.ctle_peaking_gain_db),
            zero_freq_hz=np.asarray(self.ctle_zero_freq_hz, dtype=np.float64),
            pole_freq_hz=np.asarray(self.ctle_pole_freq_hz, dtype=np.float64),
            peaking_freq_hz=self.ctle_peaking_freq_hz,
        )

        self._dfe.enabled = bool(self.dfe_en)
        self._dfe.taps = np.asarray(self.dfe_taps, dtype=np.float64).reshape(-1)

        sens = max(0.0, float(self.slicer_sensitivity))
        self._data_slicer.sensitivity = sens
        self._edge_slicer.sensitivity = sens

        ap_ui = max(0.0, float(self.slicer_aperture_ui))
        ap_samps = int(round(ap_ui * float(self.samples_per_ui)))
        self._aperture.aperture_samples = max(1, ap_samps)

        self._eye_monitor.configure_timing(
            num_samples_per_trace=int(max(1, self.samples_per_ui)),
            sample_rate_hz=float(self.sample_rate_hz),
            trace_span_ui=float(self.eye_trace_span_ui),
        )

    def run(self) -> None:
        self._configure_blocks()

        # Keep PI accumulator aligned to externally forced code when CDR is disabled.
        if float(self.pd_out_gain) == 0.0:
            self._pi_code_acc = float(int(self.pi_code) % 128)

        self._clk_data = self.clk.copy()
        self._clk_delayline.delay = self.clk_ofst
        self._clk_delayline.clk_in = self._clk_data
        self._clk_delayline.run()
        self._clk_edge = self._clk_delayline.clk_out.copy()

        self.din_ctle = self._ctle.run(self.din)
        self.dfe_feedback = self._dfe.feedback()
        self.din_eq = float(self.din_ctle - self.dfe_feedback)
        self.din_apertured = self._aperture.run(self.din_eq)

        if self._clk_edge.is_edge:
            self._edge_ana = (1 - self._clk_edge.frac_dly) * self._din_prev + self._clk_edge.frac_dly * self.din_apertured
            self._edge = self._edge_slicer.run(self._edge_ana, self.ref)

        if self._clk_data.is_edge:
            self._data_prev = self.data
            self._edge_prev = self._edge
            self._data_ana = (1 - self._clk_data.frac_dly) * self._din_prev + self._clk_data.frac_dly * self.din_apertured
            self.data = self._data_slicer.run(self._data_ana, self.ref)

        if self._clk_data.is_edge:
            if self.data != self._data_prev:
                if self._edge_prev == self.data:
                    self._pd_out = 1
                else:
                    self._pd_out = -1
            else:
                self._pd_out = 0

            self._pi_code_acc = float((self._pi_code_acc - self.pd_out_gain * self._pd_out) % 128.0)
            self.pi_code = int(math.floor(self._pi_code_acc)) % 128
            self._dfe.update(self.data)

        self._din_prev = float(self.din_apertured)

        self._eye_monitor.clk_in = self._clk_data
        self._eye_monitor.data_in = float(self.din_apertured)
        self._eye_monitor.run()

    def plot_eye(self, ax=None, **kwargs):
        return self._eye_monitor.plot(ax=ax, **kwargs)

    def plot_eye_plotly(self, **kwargs):
        return self._eye_monitor.plot_plotly(**kwargs)

    def get_eye_metrics(self, **kwargs):
        return self._eye_monitor.get_eye_metrics(**kwargs)

    def get_ctle_impulse_response(self, n_samples: int = 256):
        self._configure_blocks()
        return self._ctle.impulse_response(n_samples=n_samples)

    def get_ctle_frequency_response(self, n_points: int = 1024):
        self._configure_blocks()
        return self._ctle.frequency_response(n_points=n_points)

    def get_ctle_response_metrics(self, n_points: int = 4096):
        self._configure_blocks()
        return self._ctle.response_metrics(n_points=n_points)

    def get_ctle_design_info(self):
        self._configure_blocks()
        return self._ctle.get_design_info()

    def set_ctle_logging(
        self,
        enabled: bool = True,
        level: int = logging.INFO,
        log_response_queries: bool = False,
    ) -> None:
        self._ctle.set_logging(
            enabled=enabled,
            level=level,
            log_response_queries=log_response_queries,
        )

    def set_ctle_name(self, name: str) -> None:
        self._ctle.set_instance_name(name)
