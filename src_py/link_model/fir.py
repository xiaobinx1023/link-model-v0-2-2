from __future__ import annotations

import numpy as np
import numpy.typing as npt


class FIR:
    def __init__(self, coeff: npt.ArrayLike) -> None:
        self._coeff = np.asarray(coeff, dtype=np.float64).reshape(-1)
        if self._coeff.size == 0:
            raise ValueError("FIR: No coefficients provided.")
        self._buffer = np.zeros_like(self._coeff)
        self.input = 0.0
        self.output = 0.0

    def run(self, sample: float) -> float:
        self.input = float(sample)
        self._buffer[1:] = self._buffer[:-1]
        self._buffer[0] = self.input
        self.output = float(np.sum(self._coeff * self._buffer))
        return self.output

    def reset(self) -> None:
        self._buffer[:] = 0.0

    def set_coeff(self, coeff: npt.ArrayLike) -> None:
        coeff_arr = np.asarray(coeff, dtype=np.float64).reshape(-1)
        if coeff_arr.size != self._coeff.size:
            raise ValueError("coefficient lengths do not match")
        self._coeff = coeff_arr

    def get_coeff(self) -> npt.NDArray[np.float64]:
        return self._coeff.copy()

