from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import interpolate

from src_py.link_model.channel import Channel
from src_py.link_model.clock_gen import ClockGen
from src_py.link_model.data_gen import Pattern
from src_py.link_model.driver import Driver
from src_py.link_model.fir import FIR
from src_py.link_model.pi import PI
from src_py.link_model.rx import Rx
from src_py.link_model.tools import Tools

from .io_termination_uni import UniIOTerminationModel
from .tx_ffe import TxFFE


@dataclass
class ChanData:
    freq: npt.NDArray[np.float64]
    S: npt.NDArray[np.complex128]
    S_full: npt.NDArray[np.complex128]


class UniDirLink:
    """
    Unidirectional link model:
      TX FFE -> TX IO/termination -> channel -> RX IO/termination -> RX

    Victim lane is defined by (chan_port_tx_sel -> chan_port_rx_sel).
    Aggressor lanes are modeled as additional injected sources from selected ports.
    All non-victim lane IO terminations are configured with the same TX/RX settings
    as the victim lane.
    """

    SAMP_FREQ_HZ = 32e9 * 16
    CLK_FREQ_HZ = 32e9
    CHAN_RES_FREQ_HZ = 100e6
    Z0 = 50.0
    IMP_DELAY_PS = 64.0

    def __init__(
        self,
        chan_file: str | Path = "./data/A1_Combined_models_Slice_C_clocks_Slice_D_Data.s10p",
        chan_port_tx_sel: int = 7,
        chan_port_rx_sel: int = 8,
        tx_pattern: Pattern = Pattern.PRBS7,
        tx_ffe_taps: npt.ArrayLike | None = None,
        tx_pi_code: int = 0,
        rx_pi_code: int = 0,
        rx_term_code: float = 63.0,
        rx_clk_ofst: float | None = None,
        rx_slicer_ref: float | None = None,
        rx_pd_out_gain: float = 0.0,
        channel_pairs: list[tuple[int, int]] | None = None,
        aggressor_ports: list[int] | None = None,
        aggressor_enable: bool = True,
    ) -> None:
        self.chan_file = str(chan_file)
        self.chan_port_tx_sel = int(chan_port_tx_sel)
        self.chan_port_rx_sel = int(chan_port_rx_sel)
        self.io_term = UniIOTerminationModel(z0_ohm=self.Z0)

        self.tx = TxFFE()
        self.rx = Rx()
        self.chan = Channel()
        self.clk_src = ClockGen(self.CLK_FREQ_HZ, self.SAMP_FREQ_HZ)
        self.tx_pi = PI()
        self.rx_pi = PI()

        self.tx_pattern = Pattern(tx_pattern)
        self.tx_pi_code = int(tx_pi_code)
        self.rx_pi_code = int(rx_pi_code)
        self.rx_term_code = float(rx_term_code)
        self.rx.pi_code = int(self.rx_pi_code)
        if tx_ffe_taps is not None:
            self.tx.set_ffe_taps(tx_ffe_taps)

        self.rx.samples_per_ui = int(round(self.SAMP_FREQ_HZ / self.CLK_FREQ_HZ))
        self.rx.sample_rate_hz = self.SAMP_FREQ_HZ
        self.rx.clk_ofst = float(self.rx.samples_per_ui / 4.0 if rx_clk_ofst is None else rx_clk_ofst)
        self.rx.ref = float(0.5 * Driver.AVDD if rx_slicer_ref is None else rx_slicer_ref)
        self.rx.pd_out_gain = float(rx_pd_out_gain)

        self.tf_freq = np.array([], dtype=np.float64)
        self.tf_tx_drv_to_chan = np.array([], dtype=np.complex128)
        self.tf_chan_tx_to_rx = np.array([], dtype=np.complex128)
        self.tf_chan_to_rx = np.array([], dtype=np.complex128)
        self.tf_tx_drv_to_rx = np.array([], dtype=np.complex128)
        self.tf_xtalk_to_rx_bump_total = np.array([], dtype=np.complex128)
        self.tf_xtalk_to_rx_bump_by_port: dict[int, npt.NDArray[np.complex128]] = {}

        self.imp_tx_drv_to_chan = np.zeros(Channel.FILTER_LEN, dtype=np.float64)
        self.imp_chan_tx_to_rx = np.zeros(Channel.FILTER_LEN, dtype=np.float64)
        self.imp_chan_to_rx = np.zeros(Channel.FILTER_LEN, dtype=np.float64)
        self.imp_tx_drv_to_rx = np.zeros(Channel.FILTER_LEN, dtype=np.float64)
        self.imp_xtalk_to_rx_bump_total = np.zeros(Channel.FILTER_LEN, dtype=np.float64)
        self.imp_xtalk_to_rx_bump_by_port: dict[int, npt.NDArray[np.float64]] = {}

        self.tx_drv_out = 0.0
        self.tx_to_chan = 0.0
        self.rx_xtalk_bump = 0.0
        self.rx_bump = 0.0
        self.rx_xtalk_in = 0.0
        self.rx_in = 0.0

        self._filt_tx_drv_to_chan = FIR(np.zeros(Channel.FILTER_LEN, dtype=np.float64))
        self._filt_chan_to_rx = FIR(np.zeros(Channel.FILTER_LEN, dtype=np.float64))
        self._filt_chan_to_rx_xtalk = FIR(np.zeros(Channel.FILTER_LEN, dtype=np.float64))
        self._xtalk_filters: dict[int, FIR] = {}
        self._last_impulse_signature: tuple[float, ...] | None = None

        self.chan_data = self.load_chan_data()
        self.channel_pairs = self._normalize_channel_pairs(channel_pairs)
        self.victim_pair = self._find_pair_for_ports(self.chan_port_tx_sel, self.chan_port_rx_sel)
        if self.victim_pair is None:
            raise ValueError(
                f"Victim ports ({self.chan_port_tx_sel}, {self.chan_port_rx_sel}) must belong to one channel pair."
            )
        tx_leg = self._get_leg_in_pair(self.chan_port_tx_sel, self.victim_pair)
        rx_leg = self._get_leg_in_pair(self.chan_port_rx_sel, self.victim_pair)
        if tx_leg is None or rx_leg is None or tx_leg == rx_leg:
            raise ValueError(
                f"Victim TX/RX ports ({self.chan_port_tx_sel}, {self.chan_port_rx_sel}) must be opposite legs in one pair."
            )
        self._victim_tx_leg = int(tx_leg)
        self._victim_rx_leg = int(rx_leg)
        self._port_is_tx_side = self._build_port_tx_side_map()

        self.aggressor_enable = bool(aggressor_enable)
        self.aggressor_ports: list[int] = []
        self.aggressor_port_src: dict[int, float] = {}
        self.set_aggressor_ports(aggressor_ports)

    @staticmethod
    def _interp_complex(
        x_old: npt.ArrayLike,
        y_old: npt.ArrayLike,
        x_new: npt.ArrayLike,
    ) -> npt.NDArray[np.complex128]:
        interp_r = interpolate.interp1d(
            x_old,
            np.real(y_old),
            kind="cubic",
            fill_value="extrapolate",
            bounds_error=False,
        )
        interp_i = interpolate.interp1d(
            x_old,
            np.imag(y_old),
            kind="cubic",
            fill_value="extrapolate",
            bounds_error=False,
        )
        return np.asarray(interp_r(x_new) + 1j * interp_i(x_new), dtype=np.complex128)

    def _normalize_channel_pairs(self, channel_pairs: list[tuple[int, int]] | None) -> list[tuple[int, int]]:
        n_ports = self.chan_data.S_full.shape[0]
        if channel_pairs is None:
            if n_ports % 2 != 0:
                raise ValueError(
                    f"Cannot infer default channel_pairs for odd-port network (N={n_ports}). "
                    "Please pass channel_pairs explicitly."
                )
            channel_pairs = [(i, i + 1) for i in range(1, n_ports, 2)]

        pairs: list[tuple[int, int]] = []
        seen_ports: set[int] = set()
        for a, b in channel_pairs:
            a_i = int(a)
            b_i = int(b)
            if a_i == b_i:
                raise ValueError(f"Invalid channel pair ({a_i}, {b_i}): ports must be different.")
            if a_i < 1 or a_i > n_ports or b_i < 1 or b_i > n_ports:
                raise ValueError(f"Invalid channel pair ({a_i}, {b_i}) for N={n_ports}.")
            if a_i in seen_ports or b_i in seen_ports:
                raise ValueError(f"Port reused across channel_pairs: ({a_i}, {b_i}).")
            seen_ports.add(a_i)
            seen_ports.add(b_i)
            pairs.append((a_i, b_i))
        return pairs

    def _find_pair_for_ports(self, p0: int, p1: int) -> tuple[int, int] | None:
        target = {int(p0), int(p1)}
        for a, b in self.channel_pairs:
            if {a, b} == target:
                return (a, b)
        return None

    def get_channel_pair_for_port(self, port: int) -> tuple[int, int] | None:
        p_i = int(port)
        for a, b in self.channel_pairs:
            if p_i == a or p_i == b:
                return (a, b)
        return None

    @staticmethod
    def _get_leg_in_pair(port: int, pair: tuple[int, int]) -> int | None:
        p_i = int(port)
        if p_i == pair[0]:
            return 0
        if p_i == pair[1]:
            return 1
        return None

    def _build_port_tx_side_map(self) -> dict[int, bool]:
        out: dict[int, bool] = {}
        tx_leg = int(self._victim_tx_leg)
        for a, b in self.channel_pairs:
            if tx_leg == 0:
                out[int(a)] = True
                out[int(b)] = False
            else:
                out[int(a)] = False
                out[int(b)] = True
        n_ports = self.chan_data.S_full.shape[0]
        for p in range(1, n_ports + 1):
            out.setdefault(p, True)
        return out

    def _default_aggressor_ports(self) -> list[int]:
        # One source per aggressor lane: choose TX-side port from each non-victim pair.
        victim_pair_set = {self.victim_pair[0], self.victim_pair[1]}
        ports: list[int] = []
        for a, b in self.channel_pairs:
            if a in victim_pair_set and b in victim_pair_set:
                continue
            a_i = int(a)
            b_i = int(b)
            ports.append(a_i if self._port_is_tx_side.get(a_i, True) else b_i)
        return ports

    def _normalize_aggressor_ports(self, aggressor_ports: list[int] | None) -> list[int]:
        n_ports = self.chan_data.S_full.shape[0]
        victim_ports = {self.chan_port_tx_sel, self.chan_port_rx_sel}
        if aggressor_ports is None:
            ports = self._default_aggressor_ports()
        else:
            ports = [int(p) for p in aggressor_ports]

        cleaned: list[int] = []
        seen: set[int] = set()
        for p in ports:
            if p in seen:
                continue
            if p < 1 or p > n_ports:
                raise ValueError(f"Aggressor port {p} out of range [1, {n_ports}]")
            if p in victim_ports:
                raise ValueError(f"Aggressor port {p} conflicts with victim ports {sorted(victim_ports)}")
            seen.add(p)
            cleaned.append(p)
        return cleaned

    def set_aggressor_ports(self, aggressor_ports: list[int] | None) -> None:
        prev = getattr(self, "aggressor_port_src", {})
        self.aggressor_ports = self._normalize_aggressor_ports(aggressor_ports)
        self.aggressor_port_src = {p: float(prev.get(p, 0.0)) for p in self.aggressor_ports}
        self.update_impulses()

    def set_aggressor_sources(self, sources: dict[int, float]) -> None:
        for p, val in sources.items():
            p_i = int(p)
            if p_i not in self.aggressor_port_src:
                raise ValueError(f"Aggressor port {p_i} is not enabled. Enabled ports: {self.aggressor_ports}")
            self.aggressor_port_src[p_i] = float(val)

    def set_aggressor_enable(self, enabled: bool) -> None:
        self.aggressor_enable = bool(enabled)

    def load_chan_data(self) -> ChanData:
        snp_freq, snp_data, _, _, _ = Tools.parse_snp_file(self.chan_file)
        s_full = snp_data.astype(np.complex128, copy=True)
        n_ports = s_full.shape[0]

        if snp_freq[0] != 0:
            s_full_dc = np.zeros((n_ports, n_ports, s_full.shape[2] + 1), dtype=np.complex128)
            s_full_dc[:, :, 1:] = s_full
            for r in range(n_ports):
                for c in range(n_ports):
                    s_full_dc[r, c, :] = Tools.extrap_to_dc(snp_freq, s_full[r, c, :])
            s_full = s_full_dc
            snp_freq = np.concatenate(([0.0], snp_freq))

        snp_freq_new = np.arange(0.0, snp_freq[-1] + self.CHAN_RES_FREQ_HZ, self.CHAN_RES_FREQ_HZ)
        s_full_i = np.zeros((n_ports, n_ports, snp_freq_new.size), dtype=np.complex128)
        for r in range(n_ports):
            for c in range(n_ports):
                s_full_i[r, c, :] = self._interp_complex(snp_freq, s_full[r, c, :], snp_freq_new)

        p_tx = self.chan_port_tx_sel - 1
        p_rx = self.chan_port_rx_sel - 1
        s11 = s_full_i[p_tx, p_tx, :]
        s12 = s_full_i[p_tx, p_rx, :]
        s21 = s_full_i[p_rx, p_tx, :]
        s22 = s_full_i[p_rx, p_rx, :]

        s = np.zeros((2, 2, snp_freq_new.size), dtype=np.complex128)
        s[0, 0, :] = s11
        s[0, 1, :] = s12
        s[1, 0, :] = s21
        s[1, 1, :] = s22
        return ChanData(freq=snp_freq_new, S=s, S_full=s_full_i)

    def _impulse_signature(self) -> tuple[float, ...]:
        taps = tuple(float(x) for x in np.asarray(self.tx.ffe_taps, dtype=np.float64).reshape(-1))
        aggr = tuple(float(p) for p in self.aggressor_ports)
        return taps + (float(self.rx_term_code),) + aggr

    @staticmethod
    def _par(z1: complex, z2: complex) -> complex:
        return complex(1.0 / (1.0 / z1 + 1.0 / z2))

    def _solve_loaded_network(
        self,
        s_full_f: npt.NDArray[np.complex128],
        gamma_load: npt.NDArray[np.complex128],
        src_idx: int,
    ) -> tuple[npt.NDArray[np.complex128], complex]:
        n_ports = s_full_f.shape[0]
        e_src = np.zeros(n_ports, dtype=np.complex128)
        e_src[src_idx] = 1.0

        g_vec = np.asarray(gamma_load, dtype=np.complex128).copy()
        # Excited source is treated as active source at the port (a_src = 1).
        g_vec[src_idx] = 0.0
        g_mat = np.diag(g_vec)
        mat = np.eye(n_ports, dtype=np.complex128) - s_full_f @ g_mat
        rhs = s_full_f @ e_src
        try:
            b = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            b = np.linalg.pinv(mat) @ rhs
        a = e_src + g_vec * b
        v = a + b
        gamma_in = complex(b[src_idx])
        return v, gamma_in

    def _build_port_load_vector(
        self,
        freq_idx: int,
        z_tx: npt.NDArray[np.complex128],
        z_rx: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.complex128]:
        n_ports = self.chan_data.S_full.shape[0]
        out = np.empty(n_ports, dtype=np.complex128)
        z_tx_i = complex(z_tx[freq_idx])
        z_rx_i = complex(z_rx[freq_idx])
        for p in range(1, n_ports + 1):
            out[p - 1] = z_tx_i if self._port_is_tx_side.get(p, True) else z_rx_i
        return out

    def update_impulses(self) -> None:
        freq = self.chan_data.freq
        self.tf_freq = freq
        n_freq = freq.size
        n_ports = self.chan_data.S_full.shape[0]
        tx_idx = self.chan_port_tx_sel - 1
        rx_idx = self.chan_port_rx_sel - 1

        term_state = self.io_term.build_state(
            freq_hz=freq,
            tx_ffe_taps=self.tx.ffe_taps,
            rx_term_code=self.rx_term_code,
        )
        z_tx = np.asarray(term_state.z_load_tx, dtype=np.complex128)
        z_rx = np.asarray(term_state.z_load_rx, dtype=np.complex128)
        gamma_load_by_freq = np.zeros((n_freq, n_ports), dtype=np.complex128)
        for i in range(n_freq):
            z_load_i = self._build_port_load_vector(i, z_tx=z_tx, z_rx=z_rx)
            gamma_load_by_freq[i, :] = (z_load_i - self.Z0) / (z_load_i + self.Z0)

        self.tf_chan_tx_to_rx = np.zeros(n_freq, dtype=np.complex128)
        self.tf_tx_drv_to_chan = np.zeros(n_freq, dtype=np.complex128)
        for i in range(n_freq):
            s_full_i = self.chan_data.S_full[:, :, i]
            v, gamma_in = self._solve_loaded_network(s_full_i, gamma_load_by_freq[i, :], tx_idx)
            v_src = complex(v[tx_idx])
            self.tf_chan_tx_to_rx[i] = 0.0 + 0.0j if abs(v_src) < 1e-18 else complex(v[rx_idx] / v_src)

            den = (1.0 - gamma_in)
            z_in = complex(1e18) if abs(den) < 1e-18 else complex(self.Z0 * (1.0 + gamma_in) / den)
            z_seen = self._par(complex(term_state.z_tx_io[i]), z_in)
            self.tf_tx_drv_to_chan[i] = complex(z_seen / (term_state.r_tx_drv_ohm + z_seen))

        self.tf_chan_to_rx = np.asarray(
            term_state.z_load_rx / (self.io_term.rx_cfg.sense_res_ohm + term_state.z_load_rx),
            dtype=np.complex128,
        )
        self.tf_tx_drv_to_rx = self.tf_tx_drv_to_chan * self.tf_chan_tx_to_rx * self.tf_chan_to_rx

        self.tf_xtalk_to_rx_bump_total = np.zeros(n_freq, dtype=np.complex128)
        self.tf_xtalk_to_rx_bump_by_port = {}
        self.imp_xtalk_to_rx_bump_by_port = {}
        self._xtalk_filters = {}
        delay_s = self.IMP_DELAY_PS / 1e12
        for port in self.aggressor_ports:
            src_idx = int(port) - 1
            tf_port = np.zeros(n_freq, dtype=np.complex128)
            for i in range(n_freq):
                s_full_i = self.chan_data.S_full[:, :, i]
                v, gamma_in_src = self._solve_loaded_network(s_full_i, gamma_load_by_freq[i, :], src_idx)
                v_src = complex(v[src_idx])
                if abs(v_src) < 1e-18:
                    tf_port[i] = 0.0 + 0.0j
                    continue

                tf_src_port_to_vrx = complex(v[rx_idx] / v_src)
                den = (1.0 - gamma_in_src)
                z_in_src = complex(1e18) if abs(den) < 1e-18 else complex(self.Z0 * (1.0 + gamma_in_src) / den)
                z_seen_src = self._par(complex(term_state.z_tx_io[i]), z_in_src)
                tf_src_drv_to_port = complex(z_seen_src / (term_state.r_tx_drv_ohm + z_seen_src))
                tf_port[i] = tf_src_drv_to_port * tf_src_port_to_vrx

            self.tf_xtalk_to_rx_bump_by_port[port] = tf_port
            self.tf_xtalk_to_rx_bump_total += tf_port
            _, imp_port = Tools.convert_tf_to_imp(freq, tf_port, self.SAMP_FREQ_HZ, delay_s)
            imp_port = imp_port[: Channel.FILTER_LEN]
            self.imp_xtalk_to_rx_bump_by_port[port] = imp_port
            self._xtalk_filters[port] = FIR(imp_port)

        _, self.imp_tx_drv_to_chan = Tools.convert_tf_to_imp(freq, self.tf_tx_drv_to_chan, self.SAMP_FREQ_HZ, delay_s)
        _, self.imp_chan_tx_to_rx = Tools.convert_tf_to_imp(freq, self.tf_chan_tx_to_rx, self.SAMP_FREQ_HZ)
        _, self.imp_chan_to_rx = Tools.convert_tf_to_imp(freq, self.tf_chan_to_rx, self.SAMP_FREQ_HZ, delay_s)
        _, self.imp_tx_drv_to_rx = Tools.convert_tf_to_imp(freq, self.tf_tx_drv_to_rx, self.SAMP_FREQ_HZ, delay_s)
        _, imp_xtalk_total = Tools.convert_tf_to_imp(freq, self.tf_xtalk_to_rx_bump_total, self.SAMP_FREQ_HZ, delay_s)

        self.imp_tx_drv_to_chan = self.imp_tx_drv_to_chan[: Channel.FILTER_LEN]
        self.imp_chan_tx_to_rx = self.imp_chan_tx_to_rx[: Channel.FILTER_LEN]
        self.imp_chan_to_rx = self.imp_chan_to_rx[: Channel.FILTER_LEN]
        self.imp_tx_drv_to_rx = self.imp_tx_drv_to_rx[: Channel.FILTER_LEN]
        self.imp_xtalk_to_rx_bump_total = imp_xtalk_total[: Channel.FILTER_LEN]

        self._filt_tx_drv_to_chan.set_coeff(self.imp_tx_drv_to_chan)
        self._filt_chan_to_rx.set_coeff(self.imp_chan_to_rx)
        self._filt_chan_to_rx_xtalk.set_coeff(self.imp_chan_to_rx)
        self.chan.update_filter(
            imp_chan_21=self.imp_chan_tx_to_rx,
            imp_chan_12=np.zeros(Channel.FILTER_LEN, dtype=np.float64),
        )

        self._last_impulse_signature = self._impulse_signature()

    def _update_impulses_if_needed(self) -> None:
        sig = self._impulse_signature()
        if self._last_impulse_signature is None or sig != self._last_impulse_signature:
            self.update_impulses()

    def run(self) -> None:
        self._update_impulses_if_needed()

        self.clk_src.run()

        self.tx_pi.clk_in_i = self.clk_src.clk_i
        self.tx_pi.clk_in_q = self.clk_src.clk_q
        self.tx_pi.phase_code = int(self.tx_pi_code)
        self.tx_pi.run()

        self.rx_pi.clk_in_i = self.clk_src.clk_i
        self.rx_pi.clk_in_q = self.clk_src.clk_q
        # If PD loop gain is enabled, let RX CDR update PI code.
        # Otherwise keep fixed phase from rx_pi_code.
        if float(self.rx.pd_out_gain) != 0.0:
            self.rx_pi.phase_code = int(self.rx.pi_code)
        else:
            self.rx_pi.phase_code = int(self.rx_pi_code)
        self.rx_pi.run()

        self.tx.clk = self.tx_pi.clk_out
        self.tx.data_gen_pattern = Pattern(self.tx_pattern)
        self.tx.run()

        self.tx_drv_out = float(self.tx.out)
        self.tx_to_chan = float(self._filt_tx_drv_to_chan.run(self.tx_drv_out))

        self.chan.in_from_port_one = self.tx_to_chan
        self.chan.in_from_port_two = 0.0
        self.chan.run()

        xtalk_bump = 0.0
        for port, filt in self._xtalk_filters.items():
            src = float(self.aggressor_port_src.get(port, 0.0)) if self.aggressor_enable else 0.0
            xtalk_bump += float(filt.run(src))
        self.rx_xtalk_bump = float(xtalk_bump)

        self.rx_bump = float(self.chan.out_to_port_two + self.rx_xtalk_bump)
        self.rx_in = float(self._filt_chan_to_rx.run(self.rx_bump))
        self.rx_xtalk_in = float(self._filt_chan_to_rx_xtalk.run(self.rx_xtalk_bump))

        self.rx.clk = self.rx_pi.clk_out
        self.rx.din = self.rx_in
        self.rx.run()

    def get_aggressor_victim_pulse_response(self, aggressor_port: int, include_total: bool = True) -> dict[str, Any]:
        port = int(aggressor_port)
        if port not in self.imp_xtalk_to_rx_bump_by_port:
            raise ValueError(
                f"Aggressor port {port} is unavailable. Enabled aggressor ports: {self.aggressor_ports}"
            )
        t = np.arange(Channel.FILTER_LEN, dtype=np.float64) / self.SAMP_FREQ_HZ
        out: dict[str, Any] = {
            "time_sec": t,
            "xtalk_to_victim_rx_bump_impulse": self.imp_xtalk_to_rx_bump_by_port[port].copy(),
        }
        if include_total:
            out["xtalk_to_victim_rx_bump_total_impulse"] = self.imp_xtalk_to_rx_bump_total.copy()
        return out

    def plot_path_impulses(self, time_unit: str = "ns") -> None:
        unit_scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}
        if time_unit not in unit_scale:
            raise ValueError(f"Unsupported time_unit '{time_unit}'. Use one of {list(unit_scale.keys())}.")
        scale = unit_scale[time_unit]

        t = np.arange(Channel.FILTER_LEN, dtype=np.float64) / self.SAMP_FREQ_HZ * scale
        fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        axs[0, 0].plot(t, self.imp_tx_drv_to_chan)
        axs[0, 0].set_title("TX drv -> channel")
        axs[0, 1].plot(t, self.imp_chan_tx_to_rx, label="Main path")
        if self.imp_xtalk_to_rx_bump_total.size > 0 and np.max(np.abs(self.imp_xtalk_to_rx_bump_total)) > 0:
            axs[0, 1].plot(t, self.imp_xtalk_to_rx_bump_total, "--", label="Aggressor sum")
            axs[0, 1].legend()
        axs[0, 1].set_title("Channel/Aggressor -> RX bump")
        axs[1, 0].plot(t, self.imp_chan_to_rx)
        axs[1, 0].set_title("RX bump -> RX input")
        axs[1, 1].plot(t, self.imp_tx_drv_to_rx)
        axs[1, 1].set_title("Total TX drv -> RX input")
        for ax in axs.reshape(-1):
            ax.set_xlabel(f"Time ({time_unit})")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
