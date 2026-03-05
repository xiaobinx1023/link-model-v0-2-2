from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class TxTerminationConfig:
    # Per-segment unit resistance for TX driver bank.
    # Effective TX driver R ~= drv_scale_ohm / total_segments.
    drv_scale_ohm: float = 2700.0
    drv_series_ohm: float = 0.0
    # Minimum non-zero segment count used for divide protection.
    tap_code_floor: float = 1.0

    tx_cap_f: float = 75e-15
    tx_ind_h: float = 0.0


@dataclass
class RxTerminationConfig:
    # RX termination is configured directly from rx_term_code.
    term_scale_ohm: float = 2700.0
    term_series_ohm: float = 0.0
    term_code_gain: float = 1.0
    term_code_floor: float = 1e-3

    sense_res_ohm: float = 0.0
    rx_cap_f: float = 35e-15
    rx_ind_h: float = 0.0


@dataclass
class UniTerminationState:
    r_tx_drv_ohm: float
    r_rx_term_ohm: float
    z_tx_io: npt.NDArray[np.complex128]
    z_rx_io: npt.NDArray[np.complex128]
    z_load_tx: npt.NDArray[np.complex128]
    z_load_rx: npt.NDArray[np.complex128]


class UniIOTerminationModel:
    def __init__(
        self,
        z0_ohm: float = 50.0,
        tx_cfg: TxTerminationConfig | None = None,
        rx_cfg: RxTerminationConfig | None = None,
    ) -> None:
        self.z0_ohm = float(z0_ohm)
        self.tx_cfg = tx_cfg if tx_cfg is not None else TxTerminationConfig()
        self.rx_cfg = rx_cfg if rx_cfg is not None else RxTerminationConfig()

    @staticmethod
    def _par(z1: npt.ArrayLike, z2: npt.ArrayLike):
        z1a = np.asarray(z1)
        z2a = np.asarray(z2)
        return 1.0 / (1.0 / z1a + 1.0 / z2a)

    @staticmethod
    def _series_cap_ind_impedance(
        freq_hz: npt.ArrayLike,
        cap_f: float,
        ind_h: float,
    ) -> npt.NDArray[np.complex128]:
        f = np.asarray(freq_hz, dtype=np.float64).reshape(-1)
        w = 2.0 * np.pi * f

        z_c = np.full(f.size, 1e30, dtype=np.complex128)
        if float(cap_f) > 0.0 and f.size > 1:
            z_c[1:] = 1.0 / (1j * w[1:] * float(cap_f))

        z_l = np.zeros(f.size, dtype=np.complex128)
        if float(ind_h) != 0.0:
            z_l = 1j * w * float(ind_h)

        return z_l + z_c

    @staticmethod
    def _safe_code(code: float, floor: float) -> float:
        return max(float(code), float(floor))

    def normalized_ffe_weights_from_segments(self, tx_ffe_taps: npt.ArrayLike) -> npt.NDArray[np.float64]:
        seg = np.asarray(tx_ffe_taps, dtype=np.float64).reshape(-1)
        seg_total = self._safe_code(np.sum(np.abs(seg)), self.tx_cfg.tap_code_floor)
        return np.asarray(seg / seg_total, dtype=np.float64)

    def tx_driver_res_ohm_from_ffe(self, tx_ffe_taps: npt.ArrayLike) -> float:
        # FFE taps are interpreted as segment counts per cursor.
        # Driver resistance scales with total active segments.
        seg = np.asarray(tx_ffe_taps, dtype=np.float64).reshape(-1)
        total_segments = self._safe_code(np.sum(np.abs(seg)), self.tx_cfg.tap_code_floor)
        return float(self.tx_cfg.drv_scale_ohm / total_segments + self.tx_cfg.drv_series_ohm)

    def rx_term_res_ohm_from_code(self, rx_term_code: float) -> float:
        code = self._safe_code(self.rx_cfg.term_code_gain * float(rx_term_code), self.rx_cfg.term_code_floor)
        return float(self.rx_cfg.term_scale_ohm / code + self.rx_cfg.term_series_ohm)

    def build_state(
        self,
        freq_hz: npt.ArrayLike,
        tx_ffe_taps: npt.ArrayLike,
        rx_term_code: float,
    ) -> UniTerminationState:
        r_tx = self.tx_driver_res_ohm_from_ffe(tx_ffe_taps)
        r_rx = self.rx_term_res_ohm_from_code(rx_term_code)

        z_tx_io = self._series_cap_ind_impedance(
            freq_hz=freq_hz,
            cap_f=self.tx_cfg.tx_cap_f,
            ind_h=self.tx_cfg.tx_ind_h,
        )
        z_rx_io = self._series_cap_ind_impedance(
            freq_hz=freq_hz,
            cap_f=self.rx_cfg.rx_cap_f,
            ind_h=self.rx_cfg.rx_ind_h,
        )

        z_load_tx = self._par(r_tx, z_tx_io)
        z_load_rx = self._par(r_rx, z_rx_io)

        return UniTerminationState(
            r_tx_drv_ohm=float(r_tx),
            r_rx_term_ohm=float(r_rx),
            z_tx_io=np.asarray(z_tx_io, dtype=np.complex128),
            z_rx_io=np.asarray(z_rx_io, dtype=np.complex128),
            z_load_tx=np.asarray(z_load_tx, dtype=np.complex128),
            z_load_rx=np.asarray(z_load_rx, dtype=np.complex128),
        )

    def compute_transfer(
        self,
        s11: npt.ArrayLike,
        s12: npt.ArrayLike,
        s21: npt.ArrayLike,
        s22: npt.ArrayLike,
        state: UniTerminationState,
    ) -> dict[str, npt.NDArray[np.complex128]]:
        z0 = float(self.z0_ohm)
        s11a = np.asarray(s11, dtype=np.complex128)
        s12a = np.asarray(s12, dtype=np.complex128)
        s21a = np.asarray(s21, dtype=np.complex128)
        s22a = np.asarray(s22, dtype=np.complex128)

        gamma_load_rx = (state.z_load_rx - z0) / (state.z_load_rx + z0)

        z_in_chan_tx = z0 * (
            ((1.0 - gamma_load_rx * s22a) * (1.0 + s11a) + gamma_load_rx * s12a * s21a)
            / ((1.0 - gamma_load_rx * s22a) * (1.0 - s11a) - gamma_load_rx * s12a * s21a)
        )
        gamma_in_chan_tx = (z_in_chan_tx - z0) / (z_in_chan_tx + z0)

        tf_chan_tx_to_rx = (
            s21a
            / (1.0 - gamma_load_rx * s22a)
            * (1.0 + gamma_load_rx)
            / (1.0 + gamma_in_chan_tx)
        )

        z_seen_by_tx_drv = self._par(state.z_tx_io, z_in_chan_tx)
        tf_tx_drv_to_chan = z_seen_by_tx_drv / (state.r_tx_drv_ohm + z_seen_by_tx_drv)

        tf_chan_to_rx = state.z_load_rx / (self.rx_cfg.sense_res_ohm + state.z_load_rx)
        tf_tx_drv_to_rx = tf_tx_drv_to_chan * tf_chan_tx_to_rx * tf_chan_to_rx

        return {
            "z_in_chan_tx": np.asarray(z_in_chan_tx, dtype=np.complex128),
            "tf_tx_drv_to_chan": np.asarray(tf_tx_drv_to_chan, dtype=np.complex128),
            "tf_chan_tx_to_rx": np.asarray(tf_chan_tx_to_rx, dtype=np.complex128),
            "tf_chan_to_rx": np.asarray(tf_chan_to_rx, dtype=np.complex128),
            "tf_tx_drv_to_rx": np.asarray(tf_tx_drv_to_rx, dtype=np.complex128),
        }
