from __future__ import annotations

import numpy as np


class ApertureSampler:
    """Simple rectangular aperture sampler.

    Model:
      y[n] = mean(x[n-k+1:n+1])  where k = aperture_samples.

    This approximates sampling aperture / integration in a dynamic latch.
    """

    def __init__(self, aperture_samples: int = 1) -> None:
        self._aperture_samples = 1
        self.aperture_samples = aperture_samples
        self._buf = np.zeros(1024, dtype=np.float64)
        self._idx = 0

    @property
    def aperture_samples(self) -> int:
        return self._aperture_samples

    @aperture_samples.setter
    def aperture_samples(self, value: int) -> None:
        v = int(value)
        if v < 1:
            v = 1
        self._aperture_samples = v

    def run(self, x: float) -> float:
        self._buf[self._idx] = float(x)
        self._idx = (self._idx + 1) % self._buf.size
        k = self._aperture_samples
        # gather last k samples (wrap-safe)
        if k >= self._buf.size:
            return float(np.mean(self._buf))
        start = (self._idx - k) % self._buf.size
        if start < self._idx:
            return float(np.mean(self._buf[start : self._idx]))
        # wrap
        return float(np.mean(np.concatenate([self._buf[start:], self._buf[: self._idx]])))
