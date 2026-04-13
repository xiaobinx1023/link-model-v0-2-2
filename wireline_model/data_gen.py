from __future__ import annotations

from enum import IntEnum
from typing import Dict, Sequence

import numpy as np

from .clock import Clock


class Pattern(IntEnum):
    ALL_ZEROS = 0
    ALL_ONES = 1
    PRBS7 = 2
    PRBS9 = 3
    PRBS13 = 4
    PRBS15 = 5
    PRBS23 = 6
    PRBS31 = 7


class DataGen:
    _TAPS: Dict[Pattern, Sequence[int]] = {
        Pattern.PRBS7: (7, 6),
        Pattern.PRBS9: (9, 5),
        Pattern.PRBS13: (13, 12, 2, 1),
        Pattern.PRBS15: (15, 14),
        Pattern.PRBS23: (23, 18),
        Pattern.PRBS31: (31, 28),
    }
    _ORDER = {
        Pattern.PRBS7: 7,
        Pattern.PRBS9: 9,
        Pattern.PRBS13: 13,
        Pattern.PRBS15: 15,
        Pattern.PRBS23: 23,
        Pattern.PRBS31: 31,
    }

    def __init__(self, pattern: Pattern = Pattern.ALL_ZEROS) -> None:
        self.clk = Clock()
        self._pattern = Pattern(pattern)
        self.out = 0
        self._taps: Sequence[int] = ()
        self._lfsr = np.array([], dtype=np.uint8)
        self.reset()

    @property
    def pattern(self) -> Pattern:
        return self._pattern

    @pattern.setter
    def pattern(self, value: Pattern) -> None:
        value = Pattern(value)
        if value != self._pattern:
            self._pattern = value
            self.reset()

    def reset(self) -> None:
        if self._pattern in (Pattern.ALL_ZEROS, Pattern.ALL_ONES):
            self._taps = ()
            self._lfsr = np.array([], dtype=np.uint8)
            return
        order = self._ORDER[self._pattern]
        self._taps = self._TAPS[self._pattern]
        self._lfsr = np.ones(order, dtype=np.uint8)

    def run(self) -> None:
        if not self.clk.is_edge:
            return
        if self._pattern == Pattern.ALL_ZEROS:
            self.out = 0
            return
        if self._pattern == Pattern.ALL_ONES:
            self.out = 1
            return

        feedback = int(self._lfsr[self._taps[0] - 1])
        for tap in self._taps[1:]:
            feedback ^= int(self._lfsr[tap - 1])

        self.out = int(self._lfsr[-1])
        self._lfsr[1:] = self._lfsr[:-1]
        self._lfsr[0] = feedback

