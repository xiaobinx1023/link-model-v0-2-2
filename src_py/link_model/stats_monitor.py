from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class Stats:
    label: str
    min: Optional[float]
    max: Optional[float]
    avg: Optional[float]
    pwr: Optional[float]
    var: Optional[float]
    std: Optional[float]


class StatsMonitor:
    TYP_WINDOW_SIZE = 4096

    def __init__(self, label: str = "N/A", window_size: int = TYP_WINDOW_SIZE) -> None:
        self.din: Optional[float] = None
        self._label = label
        self._window_size = int(window_size)
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._avg: Optional[float] = None
        self._pwr: Optional[float] = None
        self._avg_accum = 0.0
        self._pwr_accum = 0.0
        self._count = 0

    @property
    def var(self) -> Optional[float]:
        if self._avg is None or self._pwr is None:
            return None
        value = self._pwr - self._avg**2
        return max(value, 0.0)

    @property
    def std(self) -> Optional[float]:
        value = self.var
        if value is None:
            return None
        return math.sqrt(value)

    def run(self) -> None:
        if self.din is None:
            return
        x = float(self.din)

        if self._min is None or self._count == self._window_size - 1:
            self._min = x
        elif x < self._min:
            self._min = x

        if self._max is None or self._count == self._window_size - 1:
            self._max = x
        elif x > self._max:
            self._max = x

        self._avg_accum += x
        if self._count == self._window_size - 1:
            self._avg = self._avg_accum / self._window_size
            self._avg_accum = 0.0

        self._pwr_accum += x * x
        if self._count == self._window_size - 1:
            self._pwr = self._pwr_accum / self._window_size
            self._pwr_accum = 0.0

        if self._count == self._window_size - 1:
            self._count = 0
        else:
            self._count += 1

    def get_stats(self) -> Stats:
        return Stats(
            label=self._label,
            min=self._min,
            max=self._max,
            avg=self._avg,
            pwr=self._pwr,
            var=self.var,
            std=self.std,
        )

    def clear(self) -> None:
        self.din = None
        self._min = None
        self._max = None
        self._avg = None
        self._pwr = None
        self._avg_accum = 0.0
        self._pwr_accum = 0.0
        self._count = 0

