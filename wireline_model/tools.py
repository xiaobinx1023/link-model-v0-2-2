from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.signal.windows import tukey


class Tools:
    @staticmethod
    def _touchstone_port_count(path: str | Path) -> int:
        match = re.search(r"\.s(\d+)p$", str(path).lower())
        if match is None:
            raise ValueError(f"Cannot determine port count from file name: {path}")
        return int(match.group(1))

    @staticmethod
    def parse_snp_file(
        data_file_name: str | Path,
        fid_log: int | None = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.complex128],
        npt.NDArray[np.float64],
        npt.NDArray[np.complex128],
        float,
    ]:
        if str(data_file_name).lower() in {"info", "help", "?"}:
            raise ValueError("help/info mode is not implemented in the Python port")

        path = Path(data_file_name)
        if not path.exists():
            raise FileNotFoundError(f"requested parameter file not found: {data_file_name}")

        n_ports = Tools._touchstone_port_count(path)
        opt_multiplier = 1e6
        opt_param = "s"
        opt_format = "ma"
        zo = 50.0

        values: list[float] = []
        with path.open("rt", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.partition("!")[0].strip().lower()
                if not line:
                    continue
                if line.startswith("#"):
                    tokens = line[1:].replace(",", " ").split()
                    i = 0
                    while i < len(tokens):
                        tok = tokens[i]
                        if tok == "mhz":
                            opt_multiplier = 1e6
                        elif tok == "ghz":
                            opt_multiplier = 1e9
                        elif tok == "khz":
                            opt_multiplier = 1e3
                        elif tok == "hz":
                            opt_multiplier = 1.0
                        elif tok in {"s", "z", "y", "g", "h", "abcd"}:
                            opt_param = tok
                        elif tok in {"ri", "ma", "db"}:
                            opt_format = tok
                        elif tok == "r" and i + 1 < len(tokens):
                            zo = float(tokens[i + 1])
                            i += 1
                        i += 1
                    continue

                for tok in line.replace(",", " ").split():
                    values.append(float(tok))

        terms_per_freq = 1 + 2 * n_ports * n_ports
        if len(values) < terms_per_freq:
            raise ValueError(f"not enough S-parameter values in {data_file_name}")
        if len(values) % terms_per_freq != 0:
            raise ValueError(
                f"cannot parse S-parameter file cleanly: token_count={len(values)}, terms_per_freq={terms_per_freq}"
            )

        n_freq = len(values) // terms_per_freq
        raw = np.asarray(values, dtype=np.float64).reshape(n_freq, terms_per_freq)
        freq = raw[:, 0] * opt_multiplier
        raw_data = raw[:, 1:]

        if n_ports == 1:
            raw_a = raw_data[:, [0]].reshape(n_freq, 1, 1)
            raw_b = raw_data[:, [1]].reshape(n_freq, 1, 1)
        elif n_ports == 2:
            raw_a = np.zeros((n_freq, 2, 2), dtype=np.float64)
            raw_b = np.zeros((n_freq, 2, 2), dtype=np.float64)
            raw_a[:, 0, 0] = raw_data[:, 0]
            raw_b[:, 0, 0] = raw_data[:, 1]
            raw_a[:, 1, 0] = raw_data[:, 2]
            raw_b[:, 1, 0] = raw_data[:, 3]
            raw_a[:, 0, 1] = raw_data[:, 4]
            raw_b[:, 0, 1] = raw_data[:, 5]
            raw_a[:, 1, 1] = raw_data[:, 6]
            raw_b[:, 1, 1] = raw_data[:, 7]
        else:
            t_ab = raw_data.reshape(n_freq, n_ports, 2 * n_ports)
            raw_a = t_ab[:, :, 0::2]
            raw_b = t_ab[:, :, 1::2]

        if opt_format == "ri":
            data_nf = raw_a + 1j * raw_b
        elif opt_format == "ma":
            data_nf = raw_a * np.cos(np.deg2rad(raw_b)) + 1j * raw_a * np.sin(np.deg2rad(raw_b))
        elif opt_format == "db":
            mag = 10 ** (raw_a / 20.0)
            ang = np.deg2rad(raw_b)
            data_nf = mag * np.cos(ang) + 1j * mag * np.sin(ang)
        else:
            raise ValueError(f"Unsupported data format: {opt_format}")

        if opt_param != "s":
            raise NotImplementedError(
                f"Python port currently supports only S-parameter files; got '{opt_param}'."
            )

        data = np.transpose(data_nf, (1, 2, 0)).astype(np.complex128, copy=False)
        freq_noise = np.array([], dtype=np.float64)
        data_noise = np.empty((0, 3), dtype=np.complex128)
        return freq, data, freq_noise, data_noise, zo

    @staticmethod
    def extrap_to_dc(
        f: npt.ArrayLike,
        tf: npt.ArrayLike,
        n_fit: int = 2,
        order: int = 1,
    ) -> npt.NDArray[np.complex128]:
        f_arr = np.asarray(f, dtype=np.float64).reshape(-1)
        tf_arr = np.asarray(tf, dtype=np.complex128).reshape(-1)

        mag = np.abs(tf_arr)
        phs = np.unwrap(np.angle(tf_arr))
        mag_p = np.polyfit(f_arr[:n_fit], mag[:n_fit], order)
        phs_p = np.polyfit(f_arr[:n_fit], phs[:n_fit], order)
        mag_dc = np.polyval(mag_p, 0.0)
        phs_dc = np.polyval(phs_p, 0.0)
        tf_dc = np.real(mag_dc * np.exp(1j * phs_dc))
        return np.concatenate(([tf_dc], tf_arr))

    @staticmethod
    def force_causaility(
        imp_old: npt.ArrayLike,
        max_iter: int = int(1e5),
        rel_tol: float = 1e-5,
        diff_tol: float = 1e-6,
    ) -> npt.NDArray[np.float64]:
        imp_old_arr = np.asarray(imp_old, dtype=np.float64).reshape(-1)
        length = int(np.ceil(len(imp_old_arr) / 2) * 2)
        h = np.fft.fft(imp_old_arr, length)
        imp = np.real(np.fft.ifft(h))

        err = np.inf
        half_idx = length // 2
        iter_idx = 0
        while True:
            imp[half_idx:] = 0.0
            h_mod = np.fft.fft(imp)
            h_mod = np.abs(h) * np.exp(1j * np.angle(h_mod))
            imp_mod = np.real(np.fft.ifft(h_mod))
            delta = np.abs(imp - imp_mod)
            err_prev = err
            denom = np.max(np.abs(imp))
            if denom == 0:
                denom = 1.0
            err = np.max(delta) / denom
            imp = imp_mod
            iter_idx += 1
            if err < rel_tol:
                break
            if abs(err_prev - err) < diff_tol or iter_idx >= max_iter:
                break
        return imp

    @staticmethod
    def convert_tf_to_imp(
        f: npt.ArrayLike,
        tf: npt.ArrayLike,
        fs: float,
        delay: float = 0.0,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        f_arr = np.asarray(f, dtype=np.float64).reshape(-1)
        tf_arr = np.asarray(tf, dtype=np.complex128).reshape(-1)

        if f_arr[0] != 0:
            raise ValueError("DC value shall be provided for FD to TD conversion!")
        if not np.allclose(np.diff(np.diff(f_arr)), 0.0):
            raise ValueError("Frequency shall be uniformly distributed!")

        tf_delayed = tf_arr * np.exp(-1j * 2.0 * np.pi * f_arr * delay)

        window = tukey(2 * (len(tf_delayed) - 1), alpha=0.25)
        tf_windowed = tf_delayed * window[len(tf_delayed) - 2 :]

        imp = np.real(np.fft.ifft(np.concatenate([tf_windowed, np.conj(tf_windowed[-2:0:-1])])))

        f_nyq = fs / 2.0
        f_max = f_arr[-1]
        resamp_factor = f_nyq / f_max
        frac = Fraction(float(resamp_factor)).limit_denominator(1000)
        p = frac.numerator
        q = frac.denominator
        imp = signal.resample_poly(imp, p, q) / p * q

        time = np.arange(len(imp), dtype=np.float64) / fs
        return time, imp.astype(np.float64, copy=False)
