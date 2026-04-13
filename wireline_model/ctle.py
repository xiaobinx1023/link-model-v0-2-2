from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import signal


@dataclass(frozen=True)
class CTLEConfig:
    sample_rate_hz: float
    dc_gain_db: float
    peaking_gain_db: float
    peaking_freq_hz: float | None
    zero_freq_hz: tuple[float, ...]
    pole_freq_hz: tuple[float, ...]


class CTLE:
    """
    CTLE modeled in serdespy style:
      H(s) = Gdc * prod_i (1 + s/wz_i) / (1 + s/wp_i)^2

    Two configuration modes:
    1) gain-peak mode:
       Provide dc_gain_db + peaking_gain_db + peaking_freq_hz,
       keep zero/pole lists empty. One (zero, pole) pair is derived.
       If peaking_gain_db <= 0 dB, shaping is unity (DC gain only).
    2) zero-pole mode:
       Provide dc_gain_db + explicit zero_freq_hz and pole_freq_hz (same length).
    """

    def __init__(self) -> None:
        self.enabled = False
        self.sample_rate_hz = 256e9
        self.dc_gain_db = 0.0
        self.peaking_gain_db = 0.0
        self.peaking_freq_hz: float | None = None
        self.zero_freq_hz = np.array([], dtype=np.float64)
        self.pole_freq_hz = np.array([], dtype=np.float64)

        self._b_z = np.array([1.0], dtype=np.float64)
        self._a_z = np.array([1.0], dtype=np.float64)
        self._state = np.array([], dtype=np.float64)
        self._configured: CTLEConfig | None = None
        self._design_info: dict[str, Any] = {}
        self.instance_name = "ctle"

        self.logger = logging.getLogger(f"{__name__}.CTLE.{self.instance_name}")
        self.log_enabled = False
        self.log_response_queries = False
        self._log_level = logging.INFO

    def set_instance_name(self, name: str) -> None:
        name_s = str(name).strip() or "ctle"
        self.instance_name = name_s
        self.logger = logging.getLogger(f"{__name__}.CTLE.{self.instance_name}")
        self.logger.setLevel(self._log_level)

    def set_logging(
        self,
        enabled: bool = True,
        level: int = logging.INFO,
        log_response_queries: bool = False,
    ) -> None:
        self.log_enabled = bool(enabled)
        self.log_response_queries = bool(log_response_queries)
        self._log_level = int(level)
        self.logger.setLevel(level)

    @staticmethod
    def _to_tuple(arr: npt.ArrayLike) -> tuple[float, ...]:
        return tuple(float(x) for x in np.asarray(arr, dtype=np.float64).reshape(-1))

    @staticmethod
    def _prewarp_hz_to_rad(freq_hz: float, fs_hz: float) -> float:
        f = float(freq_hz)
        fs = float(fs_hz)
        if f <= 0:
            return 0.0
        return float(2.0 * fs * np.tan(np.pi * f / fs))

    @staticmethod
    def _derive_zero_pole_from_peak(
        peaking_gain_db: float,
        peaking_freq_hz: float,
    ) -> tuple[float, float] | None:
        """
        Solve serdespy-style one-stage CTLE:
          |H(jw)|/|H(0)| = sqrt(1+(w/wz)^2)/(1+(w/wp)^2)
        such that:
          peak occurs at w0 and peak_over_dc = peaking_gain_db.
        """
        f0 = float(peaking_freq_hz)
        if f0 <= 0:
            raise ValueError(f"CTLE peaking_freq_hz must be > 0 for gain-peak mode, got {f0}")

        rmag = 10.0 ** (float(peaking_gain_db) / 20.0)
        if not np.isfinite(rmag) or rmag <= 0:
            raise ValueError(f"CTLE invalid peaking_gain_db: {peaking_gain_db}")

        if rmag <= 1.0 + 1e-12:
            # 0 dB peaking means unity shaping (no CTLE pole/zero stage).
            return None

        # From closed-form:
        # y = r^2 = (p/z)^2 = 2R^2 + 2R*sqrt(R^2 - 1)
        y = 2.0 * rmag * rmag + 2.0 * rmag * np.sqrt(max(rmag * rmag - 1.0, 0.0))
        if y <= 2.0:
            raise ValueError("CTLE gain-peak solve failed (invalid y <= 2).")
        ratio = np.sqrt(y)  # p/z
        z_hz = f0 / np.sqrt(y - 2.0)
        p_hz = ratio * z_hz
        return float(z_hz), float(p_hz)

    @staticmethod
    def _build_serdes_zpk(
        z_rad: npt.NDArray[np.float64],
        p_rad: npt.NDArray[np.float64],
        dc_gain_lin: float,
    ) -> tuple[list[complex], list[complex], float]:
        zeros: list[complex] = []
        poles: list[complex] = []
        gain = float(dc_gain_lin)
        for z_i, p_i in zip(z_rad, p_rad):
            z = float(z_i)
            p = float(p_i)
            zeros.append(complex(-z, 0.0))
            poles.append(complex(-p, 0.0))
            poles.append(complex(-p, 0.0))
            # (1 + s/z)/(1 + s/p)^2 = (p^2/z) * (s+z)/(s+p)^2
            gain *= (p * p / z)
        return zeros, poles, gain

    def _design(self) -> None:
        fs = float(self.sample_rate_hz)
        if fs <= 0:
            raise ValueError(f"CTLE sample_rate_hz must be > 0, got {fs}")
        nyq = 0.5 * fs

        zf_in = np.asarray(self.zero_freq_hz, dtype=np.float64).reshape(-1)
        pf_in = np.asarray(self.pole_freq_hz, dtype=np.float64).reshape(-1)

        if zf_in.size == 0 and pf_in.size == 0 and self.peaking_freq_hz is not None:
            mode = "gain_peak"
            z_p = self._derive_zero_pole_from_peak(
                peaking_gain_db=float(self.peaking_gain_db),
                peaking_freq_hz=float(self.peaking_freq_hz),
            )
            if z_p is None:
                zf = np.asarray([], dtype=np.float64)
                pf = np.asarray([], dtype=np.float64)
            else:
                z_hz, p_hz = z_p
                zf = np.asarray([z_hz], dtype=np.float64)
                pf = np.asarray([p_hz], dtype=np.float64)
        else:
            mode = "zero_pole"
            zf = zf_in
            pf = pf_in

        if zf.size != pf.size:
            raise ValueError(
                f"CTLE requires equal number of zero/pole entries, got {zf.size} and {pf.size}."
            )
        if np.any(zf <= 0):
            raise ValueError(f"CTLE zero_freq_hz must be > 0, got {zf}")
        if np.any(pf <= 0):
            raise ValueError(f"CTLE pole_freq_hz must be > 0, got {pf}")
        if np.any(zf >= nyq):
            raise ValueError(f"CTLE zero_freq_hz must be below Nyquist ({nyq:g} Hz), got {zf}")
        if np.any(pf >= nyq):
            raise ValueError(f"CTLE pole_freq_hz must be below Nyquist ({nyq:g} Hz), got {pf}")

        z_rad = np.asarray([self._prewarp_hz_to_rad(f, fs) for f in zf], dtype=np.float64)
        p_rad = np.asarray([self._prewarp_hz_to_rad(f, fs) for f in pf], dtype=np.float64)

        dc_gain_lin = 10.0 ** (float(self.dc_gain_db) / 20.0)
        zeros_rad, poles_rad, gain = self._build_serdes_zpk(z_rad, p_rad, dc_gain_lin)

        b_s, a_s = signal.zpk2tf(zeros_rad, poles_rad, gain)
        b_z, a_z = signal.bilinear(b_s, a_s, fs=fs)
        if a_z.size == 0 or abs(a_z[0]) < 1e-30:
            raise ValueError("CTLE bilinear transform produced invalid denominator.")

        b_z = np.asarray(b_z, dtype=np.float64)
        a_z = np.asarray(a_z, dtype=np.float64)
        b_z = b_z / a_z[0]
        a_z = a_z / a_z[0]

        self._b_z = b_z
        self._a_z = a_z
        self._state = np.zeros(max(len(a_z), len(b_z)) - 1, dtype=np.float64)

        self._design_info = {
            "mode": mode,
            "sample_rate_hz": fs,
            "dc_gain_db_target": float(self.dc_gain_db),
            "peaking_gain_db_input": float(self.peaking_gain_db),
            "peaking_freq_hz_input": None if self.peaking_freq_hz is None else float(self.peaking_freq_hz),
            "zero_freq_hz_input": zf_in.copy(),
            "pole_freq_hz_input": pf_in.copy(),
            "zero_freq_hz_effective": zf.copy(),
            "pole_freq_hz_effective": pf.copy(),
            "zeros_rad_final": np.asarray([z.real for z in zeros_rad], dtype=np.float64),
            "poles_rad_final": np.asarray([p.real for p in poles_rad], dtype=np.float64),
            "b_s": np.asarray(b_s, dtype=np.float64).copy(),
            "a_s": np.asarray(a_s, dtype=np.float64).copy(),
            "b_z": self._b_z.copy(),
            "a_z": self._a_z.copy(),
        }

        metrics = self.response_metrics(n_points=4096)
        self._design_info.update(
            {
                "dc_gain_db_measured": metrics["dc_gain_db"],
                "peak_gain_db_measured": metrics["peak_gain_db"],
                "peak_over_dc_db_measured": metrics["peak_over_dc_db"],
                "peak_freq_hz_measured": metrics["peak_freq_hz"],
            }
        )

        if self.log_enabled:
            self.logger.info("[%s] CTLE filter redesigned.", self.instance_name)
            self.logger.info(
                "[%s] CTLE mode=%s, targets: dc_gain_db=%.3f",
                self.instance_name,
                mode,
                float(self.dc_gain_db),
            )
            if mode == "gain_peak":
                self.logger.info(
                    "[%s] gain-peak inputs: peaking_gain_db=%.3f, peaking_freq_hz=%s",
                    self.instance_name,
                    float(self.peaking_gain_db),
                    "None" if self.peaking_freq_hz is None else f"{float(self.peaking_freq_hz):.6g}",
                )
                if zf.size == 0:
                    self.logger.info(
                        "[%s] gain-peak resolved to unity shaping (no zero/pole stage).",
                        self.instance_name,
                    )
                self.logger.info(
                    "[%s] derived zero/pole (Hz): zero=%s, pole=%s",
                    self.instance_name,
                    np.array2string(zf, precision=6, separator=", "),
                    np.array2string(pf, precision=6, separator=", "),
                )
            elif self.peaking_freq_hz is not None or abs(float(self.peaking_gain_db)) > 1e-15:
                self.logger.info(
                    "[%s] zero-pole mode active; peaking inputs are not used for shape.",
                    self.instance_name,
                )
            self.logger.info(
                "[%s] CTLE measured: dc_gain_db=%.3f, peak_gain_db=%.3f, peak_over_dc_db=%.3f @ %.6g Hz",
                self.instance_name,
                metrics["dc_gain_db"],
                metrics["peak_gain_db"],
                metrics["peak_over_dc_db"],
                metrics["peak_freq_hz"],
            )
            self.logger.info(
                "[%s] CTLE digital coeff: b_z=%s, a_z=%s",
                self.instance_name,
                np.array2string(self._b_z, precision=6, separator=", "),
                np.array2string(self._a_z, precision=6, separator=", "),
            )
            self.logger.info(
                "[%s] CTLE analog coeff: b_s=%s, a_s=%s",
                self.instance_name,
                np.array2string(np.asarray(b_s, dtype=np.float64), precision=6, separator=", "),
                np.array2string(np.asarray(a_s, dtype=np.float64), precision=6, separator=", "),
            )

    def _configure_if_needed(self) -> None:
        cfg = CTLEConfig(
            sample_rate_hz=float(self.sample_rate_hz),
            dc_gain_db=float(self.dc_gain_db),
            peaking_gain_db=float(self.peaking_gain_db),
            peaking_freq_hz=None if self.peaking_freq_hz is None else float(self.peaking_freq_hz),
            zero_freq_hz=self._to_tuple(self.zero_freq_hz),
            pole_freq_hz=self._to_tuple(self.pole_freq_hz),
        )
        if cfg == self._configured:
            return
        self._configured = cfg
        self._design()

    def configure(
        self,
        sample_rate_hz: float,
        dc_gain_db: float,
        peaking_gain_db: float,
        zero_freq_hz: npt.ArrayLike,
        pole_freq_hz: npt.ArrayLike,
        peaking_freq_hz: float | None = None,
    ) -> None:
        prev_cfg = self._configured
        self.sample_rate_hz = float(sample_rate_hz)
        self.dc_gain_db = float(dc_gain_db)
        self.peaking_gain_db = float(peaking_gain_db)
        self.zero_freq_hz = np.asarray(zero_freq_hz, dtype=np.float64).reshape(-1)
        self.pole_freq_hz = np.asarray(pole_freq_hz, dtype=np.float64).reshape(-1)
        self.peaking_freq_hz = None if peaking_freq_hz is None else float(peaking_freq_hz)
        self._configure_if_needed()
        if self._configured != prev_cfg:
            self.reset()

    def run(self, sample: float) -> float:
        x = float(sample)
        if not self.enabled:
            return x
        self._configure_if_needed()
        y, self._state = signal.lfilter(self._b_z, self._a_z, np.array([x], dtype=np.float64), zi=self._state)
        return float(y[0])

    def impulse_response(self, n_samples: int = 256) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        n = int(max(1, n_samples))
        self._configure_if_needed()
        x = np.zeros(n, dtype=np.float64)
        x[0] = 1.0
        y = signal.lfilter(self._b_z, self._a_z, x)
        t = np.arange(n, dtype=np.float64) / float(self.sample_rate_hz)
        if self.log_enabled and self.log_response_queries:
            self.logger.info(
                "[%s] CTLE impulse_response requested: n_samples=%d, first_samples=%s",
                self.instance_name,
                n,
                np.array2string(y[: min(8, y.size)], precision=6, separator=", "),
            )
        return t, y.astype(np.float64)

    def frequency_response(self, n_points: int = 1024) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
        n = int(max(2, n_points))
        self._configure_if_needed()
        f_hz, h = signal.freqz(self._b_z, self._a_z, worN=n, fs=float(self.sample_rate_hz))
        if self.log_enabled and self.log_response_queries:
            mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-15))
            self.logger.info(
                "[%s] CTLE frequency_response requested: n_points=%d, mag_db[min,max]=[%.3f, %.3f]",
                self.instance_name,
                n,
                float(np.min(mag_db)),
                float(np.max(mag_db)),
            )
        return np.asarray(f_hz, dtype=np.float64), np.asarray(h, dtype=np.complex128)

    def response_metrics(self, n_points: int = 4096) -> dict[str, float]:
        f_hz, h = self.frequency_response(n_points=n_points)
        mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-15))
        dc_gain_db = float(mag_db[0])
        peak_idx = int(np.argmax(mag_db))
        peak_gain_db = float(mag_db[peak_idx])
        peak_freq_hz = float(f_hz[peak_idx])
        out = {
            "dc_gain_db": dc_gain_db,
            "peak_gain_db": peak_gain_db,
            "peak_over_dc_db": float(peak_gain_db - dc_gain_db),
            "peak_freq_hz": peak_freq_hz,
        }
        if self.peaking_freq_hz is not None and float(self.peaking_freq_hz) > 0.0:
            f0 = float(self.peaking_freq_hz)
            idx = int(np.argmin(np.abs(f_hz - f0)))
            at_target_db = float(mag_db[idx])
            out["gain_at_target_freq_db"] = at_target_db
            out["gain_over_dc_at_target_db"] = float(at_target_db - dc_gain_db)
            out["target_freq_hz"] = f0
        return out

    def reset(self) -> None:
        self._state[:] = 0.0

    def get_design_info(self) -> dict[str, Any]:
        return dict(self._design_info)
