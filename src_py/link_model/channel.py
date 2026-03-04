from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .fir import FIR


class Channel:
    FILTER_LEN = 128

    def __init__(
        self,
        imp_chan_21: npt.ArrayLike | None = None,
        imp_chan_12: npt.ArrayLike | None = None,
    ) -> None:
        if imp_chan_21 is None:
            imp_chan_21 = np.zeros(self.FILTER_LEN, dtype=np.float64)
        if imp_chan_12 is None:
            imp_chan_12 = np.zeros(self.FILTER_LEN, dtype=np.float64)

        self.in_from_port_one = 0.0
        self.in_from_port_two = 0.0
        self.out_to_port_one = 0.0
        self.out_to_port_two = 0.0

        self._filter_chan_21 = FIR(imp_chan_21)
        self._filter_chan_12 = FIR(imp_chan_12)

    def run(self) -> None:
        self.out_to_port_one = self._filter_chan_12.run(self.in_from_port_two)
        self.out_to_port_two = self._filter_chan_21.run(self.in_from_port_one)

    def update_filter(self, imp_chan_21: npt.ArrayLike, imp_chan_12: npt.ArrayLike) -> None:
        self._filter_chan_12.set_coeff(imp_chan_12)
        self._filter_chan_21.set_coeff(imp_chan_21)

    def get_filter_chan_12_coeff(self):
        return self._filter_chan_12.get_coeff()

    def get_filter_chan_21_coeff(self):
        return self._filter_chan_21.get_coeff()

