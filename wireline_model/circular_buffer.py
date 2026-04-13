from __future__ import annotations

from typing import Generic, List, Optional, TypeVar


T = TypeVar("T")


class CircularBuffer(Generic[T]):
    def __init__(self, size: int) -> None:
        self._size = int(size)
        self._data: List[Optional[T]] = [None] * self._size
        self._head = 0
        self._tail = 0
        self._count = 0

    @property
    def is_full(self) -> bool:
        return self._count == self._size

    @property
    def is_empty(self) -> bool:
        return self._count == 0

    def add(self, value: T) -> None:
        if self.is_full:
            self._tail = (self._tail + 1) % self._size
        else:
            self._count += 1
        self._data[self._head] = value
        self._head = (self._head + 1) % self._size

    def get_data(self) -> List[T]:
        if self.is_empty:
            return []
        if self._head > self._tail:
            out = self._data[self._tail : self._head]
        else:
            out = self._data[self._tail :] + self._data[: self._head]
        return [x for x in out if x is not None]

    def clear(self) -> None:
        self._head = 0
        self._tail = 0
        self._count = 0

