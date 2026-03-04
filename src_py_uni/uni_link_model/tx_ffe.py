from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src_py.link_model.clock import Clock
from src_py.link_model.data_gen import DataGen, Pattern
from src_py.link_model.driver import Driver


class TxFFE:
    """
    Unidirectional TX with 4-tap symbol FFE from segment counts:
      [pre1_seg, main_seg, post1_seg, post2_seg]

    The implementation is causal by keeping one-symbol lookahead and evaluating:
      y[n] = pre1*d[n+1] + main*d[n] + post1*d[n-1] + post2*d[n-2]

    Segment counts are converted to normalized tap weights by:
      w_i = seg_i / sum_j |seg_j|
    """

    NUM_TAPS = 4

    def __init__(self) -> None:
        self.clk = Clock()
        self.data_gen_pattern = Pattern.ALL_ZEROS
        self.enabled = True
        self.ffe_taps = np.asarray([0.0, 63.0, 0.0, 0.0], dtype=np.float64)
        self.ffe_weights = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

        self.out = 0.0
        self.data_center = 0
        self.data_next = 0

        self._data_gen = DataGen(pattern=self.data_gen_pattern)
        self._drivers = [Driver() for _ in range(self.NUM_TAPS)]

        self._d_nm2 = 0
        self._d_nm1 = 0
        self._d_n = 0
        self._d_np1 = 0
        self._tap_bits = np.zeros(self.NUM_TAPS, dtype=np.uint8)

    def set_ffe_taps(self, taps: npt.ArrayLike) -> None:
        arr = np.asarray(taps, dtype=np.float64).reshape(-1)
        if arr.size != self.NUM_TAPS:
            raise ValueError(
                f"TxFFE expects {self.NUM_TAPS} taps [pre1, main, post1, post2], got {arr.size}"
            )
        self.ffe_taps = arr

    @staticmethod
    def normalize_weights_from_segments(segment_taps: npt.ArrayLike) -> npt.NDArray[np.float64]:
        seg = np.asarray(segment_taps, dtype=np.float64).reshape(-1)
        total = float(np.sum(np.abs(seg)))
        if total <= 0.0:
            return np.zeros_like(seg)
        return np.asarray(seg / total, dtype=np.float64)

    def _update_tap_bits_on_edge(self) -> None:
        self._tap_bits[0] = int(self._d_np1)  # pre1
        self._tap_bits[1] = int(self._d_n)    # main
        self._tap_bits[2] = int(self._d_nm1)  # post1
        self._tap_bits[3] = int(self._d_nm2)  # post2
        self.data_center = int(self._d_n)
        self.data_next = int(self._d_np1)

    def _advance_symbol_window(self, new_bit: int) -> None:
        self._d_nm2 = self._d_nm1
        self._d_nm1 = self._d_n
        self._d_n = self._d_np1
        self._d_np1 = int(new_bit)

    def run(self) -> None:
        self._data_gen.clk = self.clk
        self._data_gen.pattern = self.data_gen_pattern
        self._data_gen.run()

        if self.clk.is_edge:
            self._update_tap_bits_on_edge()
            self._advance_symbol_window(int(self._data_gen.out))

        tap_out = np.zeros(self.NUM_TAPS, dtype=np.float64)
        for i, drv in enumerate(self._drivers):
            drv.clk = self.clk
            drv.in_ = int(self._tap_bits[i])
            drv.run()
            tap_out[i] = float(drv.out)

        self.ffe_weights = self.normalize_weights_from_segments(self.ffe_taps)
        if self.enabled:
            self.out = float(np.dot(self.ffe_weights, tap_out))
        else:
            self.out = 0.0
