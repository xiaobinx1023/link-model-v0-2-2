from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Clock:
    """Simple clock sample with edge metadata."""

    is_edge: bool = False
    is_pos: bool = False
    frac_dly: float = 0.0
    period: float = 0.0

    @property
    def is_pos_edge(self) -> bool:
        return self.is_edge and self.is_pos

    @property
    def is_neg_edge(self) -> bool:
        return self.is_edge and (not self.is_pos)

    def copy(self) -> "Clock":
        return Clock(
            is_edge=self.is_edge,
            is_pos=self.is_pos,
            frac_dly=float(self.frac_dly),
            period=float(self.period),
        )

