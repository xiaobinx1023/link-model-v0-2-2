from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import interpolate

from .afe import AFE
from .channel import Channel
from .clock_delay import ClockDelay
from .clock_gen import ClockGen
from .controller import Controller
from .fir import FIR
from .io_termination import IOTerminationModel
from .pi import PI
from .rx import Rx
from .tools import Tools
from .tx import Tx


@dataclass
class ChanData:
    freq: npt.NDArray[np.float64]
    S: npt.NDArray[np.complex128]
    Z: npt.NDArray[np.complex128]
    S_full: npt.NDArray[np.complex128]


class Link:
    SAMP_FREQ_HZ = 16e9 * 16
    CLK_FREQ_HZ = 16e9
    CHAN_RES_FREQ_HZ = 100e6
    Z0 = 50.0
    AFE_IMP_DELAY_PS = 64.0

    def __init__(
        self,
        chan_file: str | Path = "./data/A1_Combined_models_Slice_C_clocks_Slice_D_Data.s10p",
        chan_port_one_sel: int = 7,
        chan_port_two_sel: int = 8,
        channel_pairs: list[tuple[int, int]] | None = None,
        aggressor_ports: list[int] | None = None,
    ) -> None:
        self.master_ctrl = Controller()
        self.slave_ctrl = Controller()

        self.master_afe = AFE()
        self.slave_afe = AFE()
        self.chan = Channel()

        self.master_tx = Tx()
        self.master_rx = Rx()
        self.slave_tx = Tx()
        self.slave_rx = Rx()
        self.master_rx.set_ctle_name("master")
        self.slave_rx.set_ctle_name("slave")
        rx_samples_per_ui = int(round(self.SAMP_FREQ_HZ / self.CLK_FREQ_HZ))
        self.master_rx.samples_per_ui = rx_samples_per_ui
        self.slave_rx.samples_per_ui = rx_samples_per_ui
        self.master_rx.sample_rate_hz = self.SAMP_FREQ_HZ
        self.slave_rx.sample_rate_hz = self.SAMP_FREQ_HZ

        self.master_clk_src = ClockGen(self.CLK_FREQ_HZ, self.SAMP_FREQ_HZ)
        self.master_tx_pi = PI()
        self.master_rx_pi = PI()
        self.chan_clk_delay_i = ClockDelay(0.0)
        self.chan_clk_delay_q = ClockDelay(0.0)
        self.slave_tx_pi = PI()
        self.slave_rx_pi = PI()

        self.chan_file = str(chan_file)
        self.chan_port_one_sel = int(chan_port_one_sel)
        self.chan_port_two_sel = int(chan_port_two_sel)
        self.io_termination = IOTerminationModel(z0_ohm=self.Z0)
        self.chan_data = self.load_chan_data()
        self.channel_pairs = self._normalize_channel_pairs(channel_pairs)
        self.victim_pair = self._find_pair_for_ports(self.chan_port_one_sel, self.chan_port_two_sel)
        if self.victim_pair is None:
            raise ValueError(
                f"Victim ports ({self.chan_port_one_sel}, {self.chan_port_two_sel}) must belong to one channel pair."
            )
        self.aggressor_ports: list[int] = self._normalize_aggressor_ports(aggressor_ports)
        self.aggressor_port_src: dict[int, float] = {p: 0.0 for p in self.aggressor_ports}

        self.tf_freq = np.array([], dtype=np.float64)
        self.tf_chan_mas_to_slv = np.array([], dtype=np.complex128)
        self.tf_chan_slv_to_mas = np.array([], dtype=np.complex128)
        self.tf_chan_to_rx_mas = np.array([], dtype=np.complex128)
        self.tf_main_drv_to_chan_mas = np.array([], dtype=np.complex128)
        self.tf_main_drv_to_rx_mas = np.array([], dtype=np.complex128)
        self.tf_echo_drv_to_chan_mas = np.array([], dtype=np.complex128)
        self.tf_echo_drv_to_rx_mas = np.array([], dtype=np.complex128)
        self.tf_chan_to_rx_slv = np.array([], dtype=np.complex128)
        self.tf_main_drv_to_chan_slv = np.array([], dtype=np.complex128)
        self.tf_main_drv_to_rx_slv = np.array([], dtype=np.complex128)
        self.tf_echo_drv_to_chan_slv = np.array([], dtype=np.complex128)
        self.tf_echo_drv_to_rx_slv = np.array([], dtype=np.complex128)
        self.tf_xtalk_next_total = np.array([], dtype=np.complex128)
        self.tf_xtalk_fext_total = np.array([], dtype=np.complex128)
        self.tf_xtalk_next_by_port: dict[int, npt.NDArray[np.complex128]] = {}
        self.tf_xtalk_fext_by_port: dict[int, npt.NDArray[np.complex128]] = {}

        self.imp_chan_mas_to_slv = np.zeros(Channel.FILTER_LEN, dtype=np.float64)
        self.imp_chan_slv_to_mas = np.zeros(Channel.FILTER_LEN, dtype=np.float64)
        self.imp_chan_to_rx_mas = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
        self.imp_main_drv_to_chan_mas = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
        self.imp_main_drv_to_rx_mas = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
        self.imp_echo_drv_to_chan_mas = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
        self.imp_echo_drv_to_rx_mas = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
        self.imp_chan_to_rx_slv = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
        self.imp_main_drv_to_chan_slv = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
        self.imp_main_drv_to_rx_slv = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
        self.imp_echo_drv_to_chan_slv = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
        self.imp_echo_drv_to_rx_slv = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
        self.imp_xtalk_next_total = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
        self.imp_xtalk_fext_total = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
        self.imp_xtalk_next_by_port: dict[int, npt.NDArray[np.float64]] = {}
        self.imp_xtalk_fext_by_port: dict[int, npt.NDArray[np.float64]] = {}
        self._xtalk_next_filters: dict[int, FIR] = {}
        self._xtalk_fext_filters: dict[int, FIR] = {}
        self._xtalk_next_dst_port_by_port: dict[int, int] = {}
        self._xtalk_fext_dst_port_by_port: dict[int, int] = {}

        self.update_chan_afe_impulses()
        self.update_tx_drv_weights()

    @staticmethod
    def _interp_complex(x_old, y_old, x_new):
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
        return interp_r(x_new) + 1j * interp_i(x_new)

    def _normalize_aggressor_ports(self, aggressor_ports: list[int] | None) -> list[int]:
        n_ports = self.chan_data.S_full.shape[0]
        victim_ports = {self.chan_port_one_sel, self.chan_port_two_sel}
        if aggressor_ports is None:
            # Default: all non-victim ports in the N-port network.
            ports = [p for p in range(1, n_ports + 1) if p not in victim_ports]
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
        if len(cleaned) > n_ports - 2:
            raise ValueError(f"At most {n_ports - 2} aggressor ports are allowed for {n_ports}-port channel.")
        return cleaned

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

    def set_aggressor_ports(self, aggressor_ports: list[int] | None) -> None:
        self.aggressor_ports = self._normalize_aggressor_ports(aggressor_ports)
        # Preserve existing source values for retained ports.
        prev = getattr(self, "aggressor_port_src", {})
        self.aggressor_port_src = {p: float(prev.get(p, 0.0)) for p in self.aggressor_ports}
        self.update_chan_afe_impulses()

    def set_aggressor_sources(self, sources: dict[int, float]) -> None:
        for p, val in sources.items():
            p_i = int(p)
            if p_i not in self.aggressor_port_src:
                raise ValueError(f"Aggressor port {p_i} is not enabled. Enabled ports: {self.aggressor_ports}")
            self.aggressor_port_src[p_i] = float(val)

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

        p1 = self.chan_port_one_sel - 1
        p2 = self.chan_port_two_sel - 1
        s_11 = s_full_i[p1, p1, :]
        s_12 = s_full_i[p1, p2, :]
        s_21 = s_full_i[p2, p1, :]
        s_22 = s_full_i[p2, p2, :]
        snp_freq = snp_freq_new

        n = snp_freq.size
        s = np.zeros((2, 2, n), dtype=np.complex128)
        z = np.zeros((2, 2, n), dtype=np.complex128)
        denom = (1 - s_11) * (1 - s_22) - s_12 * s_21
        s[0, 0, :] = s_11
        s[0, 1, :] = s_12
        s[1, 0, :] = s_21
        s[1, 1, :] = s_22
        z[0, 0, :] = self.Z0 * (((1 + s_11) * (1 - s_22) + s_12 * s_21) / denom)
        z[0, 1, :] = self.Z0 * ((2 * s_12) / denom)
        z[1, 0, :] = self.Z0 * ((2 * s_21) / denom)
        z[1, 1, :] = self.Z0 * (((1 - s_11) * (1 + s_22) + s_12 * s_21) / denom)
        return ChanData(freq=snp_freq, S=s, Z=z, S_full=s_full_i)

    def load_tline_chan_data(self, tline_len: float, alpha: float) -> ChanData:
        c = 3e8
        vf = 0.66
        freq = np.arange(0.0, self.SAMP_FREQ_HZ / 2 + self.CHAN_RES_FREQ_HZ, self.CHAN_RES_FREQ_HZ)
        beta = 2 * np.pi * freq / (c * vf)
        gamma = alpha + 1j * beta
        theta = gamma * tline_len

        s = np.zeros((2, 2, freq.size), dtype=np.complex128)
        s21 = np.exp(-theta)
        s[0, 0, :] = 0.0
        s[0, 1, :] = s21
        s[1, 0, :] = s21
        s[1, 1, :] = 0.0
        z = np.zeros_like(s)
        self.chan_data = ChanData(freq=freq, S=s, Z=z, S_full=s)
        return self.chan_data

    def _calc_loaded_transfer_from_source(
        self,
        s_full_f: npt.NDArray[np.complex128],
        gamma_load: npt.NDArray[np.complex128],
        src_idx: int,
        dst_idx: int,
    ) -> complex:
        n_ports = s_full_f.shape[0]
        e_src = np.zeros(n_ports, dtype=np.complex128)
        e_src[src_idx] = 1.0

        g_vec = gamma_load.copy()
        # Excited source is assumed matched at the driving port.
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
        den = v[src_idx]
        if abs(den) < 1e-18:
            return 0.0 + 0.0j
        return complex(v[dst_idx] / den)

    def _update_xtalk_impulses(
        self,
        z_load_mas: npt.NDArray[np.complex128],
        z_load_slv: npt.NDArray[np.complex128],
    ) -> None:
        n_freq = self.tf_freq.size
        self.tf_xtalk_next_total = np.zeros(n_freq, dtype=np.complex128)
        self.tf_xtalk_fext_total = np.zeros(n_freq, dtype=np.complex128)
        self.tf_xtalk_next_by_port = {}
        self.tf_xtalk_fext_by_port = {}
        self.imp_xtalk_next_by_port = {}
        self.imp_xtalk_fext_by_port = {}
        self._xtalk_next_filters = {}
        self._xtalk_fext_filters = {}
        self._xtalk_next_dst_port_by_port = {}
        self._xtalk_fext_dst_port_by_port = {}

        if len(self.aggressor_ports) == 0:
            self.imp_xtalk_next_total = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
            self.imp_xtalk_fext_total = np.zeros(AFE.FILTER_LEN, dtype=np.float64)
            return

        n_ports = self.chan_data.S_full.shape[0]
        vic_pair = self.victim_pair
        vic_leg0 = vic_pair[0] - 1
        vic_leg1 = vic_pair[1] - 1

        for port in self.aggressor_ports:
            src_idx = port - 1
            aggr_pair = self.get_channel_pair_for_port(port)
            if aggr_pair is None:
                raise ValueError(f"Aggressor port {port} does not belong to a declared channel pair.")
            aggr_leg = self._get_leg_in_pair(port, aggr_pair)
            if aggr_leg is None:
                raise ValueError(f"Cannot determine leg index for aggressor port {port} in pair {aggr_pair}.")
            # Mapping rule for single-ended pair network:
            # aggressor leg0 -> NEXT to victim leg0, FEXT to victim leg1
            # aggressor leg1 -> NEXT to victim leg1, FEXT to victim leg0
            if aggr_leg == 0:
                next_dst = vic_leg0
                fext_dst = vic_leg1
            else:
                next_dst = vic_leg1
                fext_dst = vic_leg0

            tf_next = np.zeros(n_freq, dtype=np.complex128)
            tf_fext = np.zeros(n_freq, dtype=np.complex128)
            for i in range(n_freq):
                z_load = np.full(n_ports, self.Z0, dtype=np.complex128)
                z_load[vic_leg0] = z_load_mas[i] if self.chan_port_one_sel == vic_pair[0] else z_load_slv[i]
                z_load[vic_leg1] = z_load_mas[i] if self.chan_port_one_sel == vic_pair[1] else z_load_slv[i]
                gamma_load = (z_load - self.Z0) / (z_load + self.Z0)
                s_full_f = self.chan_data.S_full[:, :, i]
                tf_next[i] = self._calc_loaded_transfer_from_source(s_full_f, gamma_load, src_idx, next_dst)
                tf_fext[i] = self._calc_loaded_transfer_from_source(s_full_f, gamma_load, src_idx, fext_dst)

            self.tf_xtalk_next_by_port[port] = tf_next
            self.tf_xtalk_fext_by_port[port] = tf_fext
            self._xtalk_next_dst_port_by_port[port] = next_dst + 1
            self._xtalk_fext_dst_port_by_port[port] = fext_dst + 1
            self.tf_xtalk_next_total += tf_next
            self.tf_xtalk_fext_total += tf_fext

            _, imp_next = Tools.convert_tf_to_imp(
                self.tf_freq, tf_next, self.SAMP_FREQ_HZ, self.AFE_IMP_DELAY_PS / 1e12
            )
            _, imp_fext = Tools.convert_tf_to_imp(
                self.tf_freq, tf_fext, self.SAMP_FREQ_HZ, self.AFE_IMP_DELAY_PS / 1e12
            )
            imp_next = imp_next[: AFE.FILTER_LEN]
            imp_fext = imp_fext[: AFE.FILTER_LEN]
            self.imp_xtalk_next_by_port[port] = imp_next
            self.imp_xtalk_fext_by_port[port] = imp_fext
            self._xtalk_next_filters[port] = FIR(imp_next)
            self._xtalk_fext_filters[port] = FIR(imp_fext)

        _, imp_next_total = Tools.convert_tf_to_imp(
            self.tf_freq, self.tf_xtalk_next_total, self.SAMP_FREQ_HZ, self.AFE_IMP_DELAY_PS / 1e12
        )
        _, imp_fext_total = Tools.convert_tf_to_imp(
            self.tf_freq, self.tf_xtalk_fext_total, self.SAMP_FREQ_HZ, self.AFE_IMP_DELAY_PS / 1e12
        )
        self.imp_xtalk_next_total = imp_next_total[: AFE.FILTER_LEN]
        self.imp_xtalk_fext_total = imp_fext_total[: AFE.FILTER_LEN]

    def update_chan_afe_impulses(self) -> None:
        freq = self.chan_data.freq
        self.tf_freq = freq

        mas_state = self.io_termination.build_side_state(
            freq_hz=freq,
            main_drv_codes=self.master_ctrl.tx_main_drv_codes,
            echo_drv_codes=self.master_ctrl.tx_echo_drv_codes,
            side="master",
        )
        slv_state = self.io_termination.build_side_state(
            freq_hz=freq,
            main_drv_codes=self.slave_ctrl.tx_main_drv_codes,
            echo_drv_codes=self.slave_ctrl.tx_echo_drv_codes,
            side="slave",
        )
        z_load_mas = mas_state.z_load
        z_load_slv = slv_state.z_load

        gamma_load_mas = (z_load_mas - self.Z0) / (z_load_mas + self.Z0)
        gamma_load_slv = (z_load_slv - self.Z0) / (z_load_slv + self.Z0)

        s11 = self.chan_data.S[0, 0, :]
        s12 = self.chan_data.S[0, 1, :]
        s21 = self.chan_data.S[1, 0, :]
        s22 = self.chan_data.S[1, 1, :]

        z_in_chan_from_mas = self.Z0 * (
            ((1 - gamma_load_slv * s22) * (1 + s11) + gamma_load_slv * s12 * s21)
            / ((1 - gamma_load_slv * s22) * (1 - s11) - gamma_load_slv * s12 * s21)
        )
        z_in_chan_from_slv = self.Z0 * (
            ((1 - gamma_load_mas * s11) * (1 + s22) + gamma_load_mas * s21 * s12)
            / ((1 - gamma_load_mas * s11) * (1 - s22) - gamma_load_mas * s21 * s12)
        )
        gamma_in_chan_from_mas = (z_in_chan_from_mas - self.Z0) / (z_in_chan_from_mas + self.Z0)
        gamma_in_chan_from_slv = (z_in_chan_from_slv - self.Z0) / (z_in_chan_from_slv + self.Z0)

        self.tf_chan_mas_to_slv = (
            s21 / (1 - gamma_load_slv * s22) * (1 + gamma_load_slv) / (1 + gamma_in_chan_from_mas)
        )
        self.tf_chan_slv_to_mas = (
            s12 / (1 - gamma_load_mas * s11) * (1 + gamma_load_mas) / (1 + gamma_in_chan_from_slv)
        )

        mas_tf = self.io_termination.compute_side_transfer(
            side="master",
            side_state=mas_state,
            z_in_chan=z_in_chan_from_mas,
        )
        self.tf_chan_to_rx_mas = mas_tf["tf_chan_to_rx"]
        self.tf_main_drv_to_chan_mas = mas_tf["tf_main_drv_to_chan"]
        self.tf_main_drv_to_rx_mas = mas_tf["tf_main_drv_to_rx"]
        self.tf_echo_drv_to_rx_mas = mas_tf["tf_echo_drv_to_rx"]
        self.tf_echo_drv_to_chan_mas = mas_tf["tf_echo_drv_to_chan"]

        slv_tf = self.io_termination.compute_side_transfer(
            side="slave",
            side_state=slv_state,
            z_in_chan=z_in_chan_from_slv,
        )
        self.tf_chan_to_rx_slv = slv_tf["tf_chan_to_rx"]
        self.tf_main_drv_to_chan_slv = slv_tf["tf_main_drv_to_chan"]
        self.tf_main_drv_to_rx_slv = slv_tf["tf_main_drv_to_rx"]
        self.tf_echo_drv_to_rx_slv = slv_tf["tf_echo_drv_to_rx"]
        self.tf_echo_drv_to_chan_slv = slv_tf["tf_echo_drv_to_chan"]

        # Crosstalk from selected aggressor ports to victim ports, with proper termination at all ports.
        self._update_xtalk_impulses(z_load_mas=z_load_mas, z_load_slv=z_load_slv)

        _, self.imp_chan_mas_to_slv = Tools.convert_tf_to_imp(freq, self.tf_chan_mas_to_slv, self.SAMP_FREQ_HZ)
        _, self.imp_chan_slv_to_mas = Tools.convert_tf_to_imp(freq, self.tf_chan_slv_to_mas, self.SAMP_FREQ_HZ)

        afe_delay = self.AFE_IMP_DELAY_PS / 1e12
        _, self.imp_chan_to_rx_mas = Tools.convert_tf_to_imp(freq, self.tf_chan_to_rx_mas, self.SAMP_FREQ_HZ, afe_delay)
        _, self.imp_main_drv_to_chan_mas = Tools.convert_tf_to_imp(
            freq, self.tf_main_drv_to_chan_mas, self.SAMP_FREQ_HZ, afe_delay
        )
        _, self.imp_main_drv_to_rx_mas = Tools.convert_tf_to_imp(
            freq, self.tf_main_drv_to_rx_mas, self.SAMP_FREQ_HZ, afe_delay
        )
        _, self.imp_echo_drv_to_chan_mas = Tools.convert_tf_to_imp(
            freq, self.tf_echo_drv_to_chan_mas, self.SAMP_FREQ_HZ, afe_delay
        )
        _, self.imp_echo_drv_to_rx_mas = Tools.convert_tf_to_imp(
            freq, self.tf_echo_drv_to_rx_mas, self.SAMP_FREQ_HZ, afe_delay
        )

        _, self.imp_chan_to_rx_slv = Tools.convert_tf_to_imp(freq, self.tf_chan_to_rx_slv, self.SAMP_FREQ_HZ, 50e-12)
        _, self.imp_main_drv_to_chan_slv = Tools.convert_tf_to_imp(
            freq, self.tf_main_drv_to_chan_slv, self.SAMP_FREQ_HZ, afe_delay
        )
        _, self.imp_main_drv_to_rx_slv = Tools.convert_tf_to_imp(
            freq, self.tf_main_drv_to_rx_slv, self.SAMP_FREQ_HZ, afe_delay
        )
        _, self.imp_echo_drv_to_chan_slv = Tools.convert_tf_to_imp(
            freq, self.tf_echo_drv_to_chan_slv, self.SAMP_FREQ_HZ, afe_delay
        )
        _, self.imp_echo_drv_to_rx_slv = Tools.convert_tf_to_imp(
            freq, self.tf_echo_drv_to_rx_slv, self.SAMP_FREQ_HZ, afe_delay
        )

        self.imp_chan_mas_to_slv = self.imp_chan_mas_to_slv[: Channel.FILTER_LEN]
        self.imp_chan_slv_to_mas = self.imp_chan_slv_to_mas[: Channel.FILTER_LEN]
        self.imp_main_drv_to_chan_mas = self.imp_main_drv_to_chan_mas[: AFE.FILTER_LEN]
        self.imp_echo_drv_to_chan_mas = self.imp_echo_drv_to_chan_mas[: AFE.FILTER_LEN]
        self.imp_main_drv_to_rx_mas = self.imp_main_drv_to_rx_mas[: AFE.FILTER_LEN]
        self.imp_echo_drv_to_rx_mas = self.imp_echo_drv_to_rx_mas[: AFE.FILTER_LEN]
        self.imp_chan_to_rx_mas = self.imp_chan_to_rx_mas[: AFE.FILTER_LEN]
        self.imp_main_drv_to_chan_slv = self.imp_main_drv_to_chan_slv[: AFE.FILTER_LEN]
        self.imp_echo_drv_to_chan_slv = self.imp_echo_drv_to_chan_slv[: AFE.FILTER_LEN]
        self.imp_main_drv_to_rx_slv = self.imp_main_drv_to_rx_slv[: AFE.FILTER_LEN]
        self.imp_echo_drv_to_rx_slv = self.imp_echo_drv_to_rx_slv[: AFE.FILTER_LEN]
        self.imp_chan_to_rx_slv = self.imp_chan_to_rx_slv[: AFE.FILTER_LEN]

        self.chan.update_filter(self.imp_chan_mas_to_slv, self.imp_chan_slv_to_mas)
        self.master_afe.update_filter(
            self.imp_main_drv_to_chan_mas,
            self.imp_echo_drv_to_chan_mas,
            self.imp_main_drv_to_rx_mas,
            self.imp_echo_drv_to_rx_mas,
            self.imp_chan_to_rx_mas,
        )
        self.slave_afe.update_filter(
            self.imp_main_drv_to_chan_slv,
            self.imp_echo_drv_to_chan_slv,
            self.imp_main_drv_to_rx_slv,
            self.imp_echo_drv_to_rx_slv,
            self.imp_chan_to_rx_slv,
        )

    def _norm_codes(self, codes: npt.ArrayLike) -> npt.NDArray[np.float64]:
        arr = np.asarray(codes, dtype=np.float64).reshape(-1)
        total = np.sum(arr)
        if total == 0:
            return np.zeros_like(arr)
        return arr / total

    def update_tx_drv_weights(self) -> None:
        self.master_tx.main_drivers_weights = self._norm_codes(self.master_ctrl.tx_main_drv_codes)
        self.master_tx.echo_drivers_weights = self._norm_codes(self.master_ctrl.tx_echo_drv_codes)
        self.slave_tx.main_drivers_weights = self._norm_codes(self.slave_ctrl.tx_main_drv_codes)
        self.slave_tx.echo_drivers_weights = self._norm_codes(self.slave_ctrl.tx_echo_drv_codes)

    def apply_rx_controls(self) -> None:
        self.master_rx.clk_ofst = self.master_ctrl.rx_clk_ofset
        self.master_rx.ref = self.master_ctrl.rx_slc_ref
        self.master_rx.pd_out_gain = self.master_ctrl.rx_pd_out_gain
        self.master_rx.ctle_en = self.master_ctrl.rx_ctle_en
        self.master_rx.ctle_dc_gain_db = self.master_ctrl.rx_ctle_dc_gain_db
        self.master_rx.ctle_peaking_gain_db = self.master_ctrl.rx_ctle_peaking_gain_db
        self.master_rx.ctle_peaking_freq_hz = self.master_ctrl.rx_ctle_peaking_freq_hz
        self.master_rx.ctle_zero_freq_hz = np.asarray(self.master_ctrl.rx_ctle_zero_freq_hz, dtype=np.float64)
        self.master_rx.ctle_pole_freq_hz = np.asarray(self.master_ctrl.rx_ctle_pole_freq_hz, dtype=np.float64)
        self.master_rx.dfe_en = self.master_ctrl.rx_dfe_en
        self.master_rx.dfe_taps = np.asarray(self.master_ctrl.rx_dfe_taps, dtype=np.float64)
        self.master_rx.slicer_sensitivity = self.master_ctrl.rx_slicer_sensitivity
        self.master_rx.slicer_aperture_ui = self.master_ctrl.rx_slicer_aperture_ui

        self.slave_rx.clk_ofst = self.slave_ctrl.rx_clk_ofset
        self.slave_rx.ref = self.slave_ctrl.rx_slc_ref
        self.slave_rx.pd_out_gain = self.slave_ctrl.rx_pd_out_gain
        self.slave_rx.ctle_en = self.slave_ctrl.rx_ctle_en
        self.slave_rx.ctle_dc_gain_db = self.slave_ctrl.rx_ctle_dc_gain_db
        self.slave_rx.ctle_peaking_gain_db = self.slave_ctrl.rx_ctle_peaking_gain_db
        self.slave_rx.ctle_peaking_freq_hz = self.slave_ctrl.rx_ctle_peaking_freq_hz
        self.slave_rx.ctle_zero_freq_hz = np.asarray(self.slave_ctrl.rx_ctle_zero_freq_hz, dtype=np.float64)
        self.slave_rx.ctle_pole_freq_hz = np.asarray(self.slave_ctrl.rx_ctle_pole_freq_hz, dtype=np.float64)
        self.slave_rx.dfe_en = self.slave_ctrl.rx_dfe_en
        self.slave_rx.dfe_taps = np.asarray(self.slave_ctrl.rx_dfe_taps, dtype=np.float64)
        self.slave_rx.slicer_sensitivity = self.slave_ctrl.rx_slicer_sensitivity
        self.slave_rx.slicer_aperture_ui = self.slave_ctrl.rx_slicer_aperture_ui

    def run(self) -> None:
        if self.master_ctrl.is_drv_codes_changed or self.slave_ctrl.is_drv_codes_changed:
            self.update_chan_afe_impulses()
            self.update_tx_drv_weights()
            self.master_ctrl.reset_is_drv_codes_changed()
            self.slave_ctrl.reset_is_drv_codes_changed()

        self.master_clk_src.run()

        self.master_tx_pi.clk_in_i = self.master_clk_src.clk_i
        self.master_tx_pi.clk_in_q = self.master_clk_src.clk_q
        self.master_tx_pi.phase_code = self.master_ctrl.tx_pi_code
        self.master_tx_pi.run()

        self.master_rx_pi.clk_in_i = self.master_clk_src.clk_i
        self.master_rx_pi.clk_in_q = self.master_clk_src.clk_q
        self.master_rx_pi.phase_code = self.master_rx.pi_code
        self.master_rx_pi.run()

        self.chan_clk_delay_i.clk_in = self.master_clk_src.clk_i
        self.chan_clk_delay_i.run()
        self.chan_clk_delay_q.clk_in = self.master_clk_src.clk_q
        self.chan_clk_delay_q.run()

        self.slave_tx_pi.clk_in_i = self.chan_clk_delay_i.clk_out
        self.slave_tx_pi.clk_in_q = self.chan_clk_delay_q.clk_out
        self.slave_tx_pi.phase_code = self.slave_ctrl.tx_pi_code
        self.slave_tx_pi.run()

        self.slave_rx_pi.clk_in_i = self.chan_clk_delay_i.clk_out
        self.slave_rx_pi.clk_in_q = self.chan_clk_delay_q.clk_out
        self.slave_rx_pi.phase_code = self.slave_rx.pi_code
        self.slave_rx_pi.run()

        self.master_tx.clk = self.master_tx_pi.clk_out
        self.master_tx.data_gen_pattern = self.master_ctrl.tx_data_gen_pattern
        self.master_tx.main_drivers_inv_pol = np.asarray(self.master_ctrl.tx_main_drv_inv_pol, dtype=bool)
        self.master_tx.main_drivers_en = self.master_ctrl.tx_main_drv_en
        self.master_tx.echo_drivers_inv_pol = np.asarray(self.master_ctrl.tx_echo_drv_inv_pol, dtype=bool)
        self.master_tx.echo_drivers_en = self.master_ctrl.tx_echo_drv_en

        self.slave_tx.clk = self.slave_tx_pi.clk_out
        self.slave_tx.data_gen_pattern = self.slave_ctrl.tx_data_gen_pattern
        self.slave_tx.main_drivers_inv_pol = np.asarray(self.slave_ctrl.tx_main_drv_inv_pol, dtype=bool)
        self.slave_tx.main_drivers_en = self.slave_ctrl.tx_main_drv_en
        self.slave_tx.echo_drivers_inv_pol = np.asarray(self.slave_ctrl.tx_echo_drv_inv_pol, dtype=bool)
        self.slave_tx.echo_drivers_en = self.slave_ctrl.tx_echo_drv_en

        self.master_tx.run()
        self.slave_tx.run()

        self.master_afe.in_from_main_drv = self.master_tx.main_drivers_out
        self.master_afe.in_from_echo_drv = self.master_tx.echo_drivers_out
        self.master_afe.run_outbound()

        self.slave_afe.in_from_main_drv = self.slave_tx.main_drivers_out
        self.slave_afe.in_from_echo_drv = self.slave_tx.echo_drivers_out
        self.slave_afe.run_outbound()

        self.chan.in_from_port_one = self.master_afe.out_to_bump
        self.chan.in_from_port_two = self.slave_afe.out_to_bump
        self.chan.run()

        xtalk_to_mas = 0.0
        xtalk_to_slv = 0.0
        for port in self.aggressor_ports:
            src = float(self.aggressor_port_src.get(port, 0.0))
            if port in self._xtalk_next_filters:
                v_next = self._xtalk_next_filters[port].run(src)
                if self._xtalk_next_dst_port_by_port.get(port) == self.chan_port_one_sel:
                    xtalk_to_mas += v_next
                elif self._xtalk_next_dst_port_by_port.get(port) == self.chan_port_two_sel:
                    xtalk_to_slv += v_next
            if port in self._xtalk_fext_filters:
                v_fext = self._xtalk_fext_filters[port].run(src)
                if self._xtalk_fext_dst_port_by_port.get(port) == self.chan_port_one_sel:
                    xtalk_to_mas += v_fext
                elif self._xtalk_fext_dst_port_by_port.get(port) == self.chan_port_two_sel:
                    xtalk_to_slv += v_fext

        self.master_afe.in_from_main_drv = self.master_tx.main_drivers_out
        self.master_afe.in_from_echo_drv = self.master_tx.echo_drivers_out
        self.master_afe.in_from_bump = self.chan.out_to_port_one + xtalk_to_mas
        self.master_afe.run_inboud()

        self.slave_afe.in_from_main_drv = self.slave_tx.main_drivers_out
        self.slave_afe.in_from_echo_drv = self.slave_tx.echo_drivers_out
        self.slave_afe.in_from_bump = self.chan.out_to_port_two + xtalk_to_slv
        self.slave_afe.run_inboud()

        self.apply_rx_controls()

        self.master_rx.clk = self.master_rx_pi.clk_out
        self.master_rx.din = self.master_afe.out_to_rx
        self.master_rx.run()

        self.slave_rx.clk = self.slave_rx_pi.clk_out
        self.slave_rx.din = self.slave_afe.out_to_rx
        self.slave_rx.run()

    def get_aggressor_victim_pulse_response(
        self,
        aggressor_port: int,
        include_total: bool = False,
    ) -> dict[str, Any]:
        port = int(aggressor_port)
        if port not in self.imp_xtalk_next_by_port or port not in self.imp_xtalk_fext_by_port:
            raise ValueError(
                f"Aggressor port {port} is unavailable. Enabled aggressor ports: {self.aggressor_ports}"
            )

        t = np.arange(AFE.FILTER_LEN, dtype=np.float64) / self.SAMP_FREQ_HZ
        out: dict[str, Any] = {
            "time_sec": t,
            "next_impulse": self.imp_xtalk_next_by_port[port].copy(),
            "fext_impulse": self.imp_xtalk_fext_by_port[port].copy(),
        }
        out["next_dst_port"] = int(self._xtalk_next_dst_port_by_port.get(port, self.chan_port_one_sel))
        out["fext_dst_port"] = int(self._xtalk_fext_dst_port_by_port.get(port, self.chan_port_two_sel))
        if include_total:
            out["next_total_impulse"] = self.imp_xtalk_next_total.copy()
            out["fext_total_impulse"] = self.imp_xtalk_fext_total.copy()
        return out

    def plot_aggressor_victim_pulse_response(
        self,
        aggressor_port: int,
        include_total: bool = True,
        time_unit: str = "ns",
    ) -> None:
        data = self.get_aggressor_victim_pulse_response(aggressor_port, include_total=include_total)
        pair = self.get_channel_pair_for_port(aggressor_port)
        pair_txt = f" (Channel {pair[0]}-{pair[1]})" if pair is not None else ""
        next_dst = int(data["next_dst_port"])
        fext_dst = int(data["fext_dst_port"])
        unit_scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}
        if time_unit not in unit_scale:
            raise ValueError(f"Unsupported time_unit '{time_unit}'. Use one of {list(unit_scale.keys())}.")

        scale = unit_scale[time_unit]
        t = data["time_sec"] * scale

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].plot(
            t,
            data["next_impulse"],
            label=f"Port {aggressor_port}{pair_txt} -> Victim Port {next_dst} (NEXT)",
        )
        if include_total and "next_total_impulse" in data:
            axes[0].plot(t, data["next_total_impulse"], "--", label="NEXT total (all aggressors)")
        axes[0].set_title("Near-End Crosstalk Pulse Response")
        axes[0].set_xlabel(f"Time ({time_unit})")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend()

        axes[1].plot(
            t,
            data["fext_impulse"],
            label=f"Port {aggressor_port}{pair_txt} -> Victim Port {fext_dst} (FEXT)",
        )
        if include_total and "fext_total_impulse" in data:
            axes[1].plot(t, data["fext_total_impulse"], "--", label="FEXT total (all aggressors)")
        axes[1].set_title("Far-End Crosstalk Pulse Response")
        axes[1].set_xlabel(f"Time ({time_unit})")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        plt.show()

    def diagnostic(self, show: bool = True) -> dict[str, Any]:
        master_stats = [s.__dict__ for s in self.master_afe.get_stats()]
        slave_stats = [s.__dict__ for s in self.slave_afe.get_stats()]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        _, master_eye_metrics = self.master_rx.plot_eye(
            axes[0],
            mask_type="diamond",
            mask_sigma=1,
            x_unit="ui",
            return_metrics=True)
        master_xc = float(master_eye_metrics.get("x_center_in_unit", master_eye_metrics.get("x_center", float("nan"))))
        axes[0].set_title(f"Master Eye (x_center={master_xc:.3f} UI)")
        _, slave_eye_metrics = self.slave_rx.plot_eye(
            axes[1],
            mask_type="diamond",
            mask_sigma=1,
            x_unit="ui",
            return_metrics=True)
        slave_xc = float(slave_eye_metrics.get("x_center_in_unit", slave_eye_metrics.get("x_center", float("nan"))))
        axes[1].set_title(f"Slave Eye (x_center={slave_xc:.3f} UI)")
        fig.tight_layout()
        if show:
            plt.pause(0.001)
        return {
            "master_stats": master_stats,
            "slave_stats": slave_stats,
            "master_eye_metrics": master_eye_metrics,
            "slave_eye_metrics": slave_eye_metrics,
        }

    def plot_chan_afe_impulses(self, time_unit: str = "ns") -> None:
        unit_scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}
        if time_unit not in unit_scale:
            raise ValueError(f"Unsupported time_unit '{time_unit}'. Use one of {list(unit_scale.keys())}.")
        scale = unit_scale[time_unit]

        t_afe = np.arange(len(self.imp_main_drv_to_rx_mas), dtype=np.float64) / self.SAMP_FREQ_HZ * scale
        t_chan = np.arange(len(self.imp_chan_mas_to_slv), dtype=np.float64) / self.SAMP_FREQ_HZ * scale

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes[0, 0].plot(t_afe, self.imp_main_drv_to_rx_mas, label="main->rx (master)")
        axes[0, 0].plot(t_afe, self.imp_echo_drv_to_rx_mas, label="echo->rx (master)")
        axes[0, 0].set_xlabel(f"Time ({time_unit})")
        axes[0, 0].legend()
        axes[0, 0].grid(True, which="both", alpha=0.3)

        axes[0, 1].plot(t_afe, self.imp_main_drv_to_rx_slv, label="main->rx (slave)")
        axes[0, 1].plot(t_afe, self.imp_echo_drv_to_rx_slv, label="echo->rx (slave)")
        axes[0, 1].set_xlabel(f"Time ({time_unit})")
        axes[0, 1].legend()
        axes[0, 1].grid(True, which="both", alpha=0.3)

        axes[1, 0].plot(t_afe, self.imp_main_drv_to_chan_mas, label="main->chan (master)")
        axes[1, 0].plot(t_afe, self.imp_echo_drv_to_chan_mas, label="echo->chan (master)")
        axes[1, 0].set_xlabel(f"Time ({time_unit})")
        axes[1, 0].legend()
        axes[1, 0].grid(True, which="both", alpha=0.3)

        axes[1, 1].plot(t_afe, self.imp_main_drv_to_chan_slv, label="main->chan (slave)")
        axes[1, 1].plot(t_afe, self.imp_echo_drv_to_chan_slv, label="echo->chan (slave)")
        axes[1, 1].set_xlabel(f"Time ({time_unit})")
        axes[1, 1].legend()
        axes[1, 1].grid(True, which="both", alpha=0.3)

        axes[2, 0].plot(t_chan, self.imp_chan_mas_to_slv, label="chan mas->slv")
        axes[2, 0].plot(t_chan, self.imp_chan_slv_to_mas, label="chan slv->mas")
        axes[2, 0].set_xlabel(f"Time ({time_unit})")
        axes[2, 0].legend()
        axes[2, 0].grid(True, which="both", alpha=0.3)

        axes[2, 1].plot(t_afe, self.imp_chan_to_rx_mas, label="chan->rx (master)")
        axes[2, 1].plot(t_afe, self.imp_chan_to_rx_slv, label="chan->rx (slave)")
        axes[2, 1].set_xlabel(f"Time ({time_unit})")
        axes[2, 1].legend()
        axes[2, 1].grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        plt.show()
