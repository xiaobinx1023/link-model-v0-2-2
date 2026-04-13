from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .fir import FIR
from .stats_monitor import Stats
from .stats_monitor import StatsMonitor


class AFE:
    FILTER_LEN = 128

    def __init__(
        self,
        imp_main_drv_to_bump: npt.ArrayLike | None = None,
        imp_echo_drv_to_bump: npt.ArrayLike | None = None,
        imp_main_drv_to_rx: npt.ArrayLike | None = None,
        imp_echo_drv_to_rx: npt.ArrayLike | None = None,
        imp_bump_to_rx: npt.ArrayLike | None = None,
    ) -> None:
        if imp_main_drv_to_bump is None:
            imp_main_drv_to_bump = np.zeros(self.FILTER_LEN, dtype=np.float64)
        if imp_echo_drv_to_bump is None:
            imp_echo_drv_to_bump = np.zeros(self.FILTER_LEN, dtype=np.float64)
        if imp_main_drv_to_rx is None:
            imp_main_drv_to_rx = np.zeros(self.FILTER_LEN, dtype=np.float64)
        if imp_echo_drv_to_rx is None:
            imp_echo_drv_to_rx = np.zeros(self.FILTER_LEN, dtype=np.float64)
        if imp_bump_to_rx is None:
            imp_bump_to_rx = np.zeros(self.FILTER_LEN, dtype=np.float64)

        self.in_from_main_drv = 0.0
        self.in_from_echo_drv = 0.0
        self.in_from_bump = 0.0
        self.out_to_bump = 0.0
        self.out_to_rx = 0.0

        self._filter_main_drv_to_bump = FIR(imp_main_drv_to_bump)
        self._filter_echo_drv_to_bump = FIR(imp_echo_drv_to_bump)
        self._filter_main_drv_to_rx = FIR(imp_main_drv_to_rx)
        self._filter_echo_drv_to_rx = FIR(imp_echo_drv_to_rx)
        self._filter_bump_to_rx = FIR(imp_bump_to_rx)

        self._in_from_main_drv_stats_monitor = StatsMonitor("in_from_main_drv")
        self._in_from_echo_drv_stats_monitor = StatsMonitor("in_from_echo_drv")
        self._in_from_bump_stats_monitor = StatsMonitor("in_from_bump")
        self._out_to_bump_stats_monitor = StatsMonitor("out_to_bump")
        self._v_bump_stats_monitor = StatsMonitor("v_bump")
        self._out_to_rx_stats_monitor = StatsMonitor("out_to_rx")

    @property
    def v_bump(self) -> float:
        return self.in_from_bump + self.out_to_bump

    def run_outbound(self) -> None:
        self.out_to_bump = self._filter_main_drv_to_bump.run(
            self.in_from_main_drv
        ) + self._filter_echo_drv_to_bump.run(self.in_from_echo_drv)

        self._in_from_main_drv_stats_monitor.din = self.in_from_main_drv
        self._in_from_main_drv_stats_monitor.run()
        self._in_from_echo_drv_stats_monitor.din = self.in_from_echo_drv
        self._in_from_echo_drv_stats_monitor.run()
        self._in_from_bump_stats_monitor.din = self.in_from_bump
        self._in_from_bump_stats_monitor.run()
        self._v_bump_stats_monitor.din = self.v_bump
        self._v_bump_stats_monitor.run()
        self._out_to_bump_stats_monitor.din = self.out_to_bump
        self._out_to_bump_stats_monitor.run()
        self._out_to_rx_stats_monitor.din = self.out_to_rx
        self._out_to_rx_stats_monitor.run()

    def run_inboud(self) -> None:
        self.out_to_rx = (
            self._filter_main_drv_to_rx.run(self.in_from_main_drv)
            + self._filter_echo_drv_to_rx.run(self.in_from_echo_drv)
            + self._filter_bump_to_rx.run(self.in_from_bump)
        )

    def update_filter(
        self,
        imp_main_drv_to_bump: npt.ArrayLike,
        imp_echo_drv_to_bump: npt.ArrayLike,
        imp_main_drv_to_rx: npt.ArrayLike,
        imp_echo_drv_to_rx: npt.ArrayLike,
        imp_bump_to_rx: npt.ArrayLike,
    ) -> None:
        self._filter_main_drv_to_bump.set_coeff(imp_main_drv_to_bump)
        self._filter_echo_drv_to_bump.set_coeff(imp_echo_drv_to_bump)
        self._filter_main_drv_to_rx.set_coeff(imp_main_drv_to_rx)
        self._filter_echo_drv_to_rx.set_coeff(imp_echo_drv_to_rx)
        self._filter_bump_to_rx.set_coeff(imp_bump_to_rx)

    def get_stats(self) -> list[Stats]:
        monitors = [
            self._in_from_main_drv_stats_monitor,
            self._in_from_echo_drv_stats_monitor,
            self._in_from_bump_stats_monitor,
            self._v_bump_stats_monitor,
            self._out_to_bump_stats_monitor,
            self._out_to_rx_stats_monitor,
        ]
        out: list[Stats] = []
        for m in monitors:
            s = m.get_stats()
            s.label = f"AFE.{s.label}"
            out.append(s)
        return out

