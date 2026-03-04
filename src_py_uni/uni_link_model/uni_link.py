from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import interpolate

from src_py.link_model.channel import Channel
from src_py.link_model.clock_gen import ClockGen
from src_py.link_model.data_gen import Pattern
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


class UniDirLink:
    """
    Unidirectional link model:
      TX FFE -> TX IO/termination -> channel -> RX IO/termination -> RX
    """

    SAMP_FREQ_HZ = 64e9 * 16
    CLK_FREQ_HZ = 64e9
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
    ) -> None:
        self.chan_file = str(chan_file)
        self.chan_port_tx_sel = int(chan_port_tx_sel)
        self.chan_port_rx_sel = int(chan_port_rx_sel)

        self.tx = TxFFE()
        self.rx = Rx()
        self.chan = Channel()
        self.clk_src = ClockGen(self.CLK_FREQ_HZ, self.SAMP_FREQ_HZ)
        self.tx_pi = PI()
        self.rx_pi = PI()
        self.io_term = UniIOTerminationModel(z0_ohm=self.Z0)

        self.tx_pattern = Pattern(tx_pattern)
        self.tx_pi_code = int(tx_pi_code)
        self.rx_pi_code = int(rx_pi_code)
        self.rx_term_code = float(rx_term_code)
        self.rx.pi_code = int(self.rx_pi_code)
        if tx_ffe_taps is not None:
            self.tx.set_ffe_taps(tx_ffe_taps)

        self.rx.samples_per_ui = int(round(self.SAMP_FREQ_HZ / self.CLK_FREQ_HZ))
        self.rx.sample_rate_hz = self.SAMP_FREQ_HZ

        self.tf_freq = np.array([], dtype=np.float64)
        self.tf_tx_drv_to_chan = np.array([], dtype=np.complex128)
        self.tf_chan_tx_to_rx = np.array([], dtype=np.complex128)
        self.tf_chan_to_rx = np.array([], dtype=np.complex128)
        self.tf_tx_drv_to_rx = np.array([], dtype=np.complex128)

        self.imp_tx_drv_to_chan = np.zeros(Channel.FILTER_LEN, dtype=np.float64)
        self.imp_chan_tx_to_rx = np.zeros(Channel.FILTER_LEN, dtype=np.float64)
        self.imp_chan_to_rx = np.zeros(Channel.FILTER_LEN, dtype=np.float64)
        self.imp_tx_drv_to_rx = np.zeros(Channel.FILTER_LEN, dtype=np.float64)

        self.tx_drv_out = 0.0
        self.tx_to_chan = 0.0
        self.rx_bump = 0.0
        self.rx_in = 0.0

        self._filt_tx_drv_to_chan = FIR(np.zeros(Channel.FILTER_LEN, dtype=np.float64))
        self._filt_chan_to_rx = FIR(np.zeros(Channel.FILTER_LEN, dtype=np.float64))
        self._last_impulse_signature: tuple[float, ...] | None = None

        self.chan_data = self.load_chan_data()
        self.update_impulses()

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
        return ChanData(freq=snp_freq_new, S=s)

    def _impulse_signature(self) -> tuple[float, ...]:
        taps = tuple(float(x) for x in np.asarray(self.tx.ffe_taps, dtype=np.float64).reshape(-1))
        return taps + (float(self.rx_term_code),)

    def update_impulses(self) -> None:
        freq = self.chan_data.freq
        self.tf_freq = freq

        term_state = self.io_term.build_state(
            freq_hz=freq,
            tx_ffe_taps=self.tx.ffe_taps,
            rx_term_code=self.rx_term_code,
        )
        s11 = self.chan_data.S[0, 0, :]
        s12 = self.chan_data.S[0, 1, :]
        s21 = self.chan_data.S[1, 0, :]
        s22 = self.chan_data.S[1, 1, :]
        tf = self.io_term.compute_transfer(
            s11=s11,
            s12=s12,
            s21=s21,
            s22=s22,
            state=term_state,
        )

        self.tf_tx_drv_to_chan = tf["tf_tx_drv_to_chan"]
        self.tf_chan_tx_to_rx = tf["tf_chan_tx_to_rx"]
        self.tf_chan_to_rx = tf["tf_chan_to_rx"]
        self.tf_tx_drv_to_rx = tf["tf_tx_drv_to_rx"]

        delay_s = self.IMP_DELAY_PS / 1e12
        _, self.imp_tx_drv_to_chan = Tools.convert_tf_to_imp(freq, self.tf_tx_drv_to_chan, self.SAMP_FREQ_HZ, delay_s)
        _, self.imp_chan_tx_to_rx = Tools.convert_tf_to_imp(freq, self.tf_chan_tx_to_rx, self.SAMP_FREQ_HZ)
        _, self.imp_chan_to_rx = Tools.convert_tf_to_imp(freq, self.tf_chan_to_rx, self.SAMP_FREQ_HZ, delay_s)
        _, self.imp_tx_drv_to_rx = Tools.convert_tf_to_imp(freq, self.tf_tx_drv_to_rx, self.SAMP_FREQ_HZ, delay_s)

        self.imp_tx_drv_to_chan = self.imp_tx_drv_to_chan[: Channel.FILTER_LEN]
        self.imp_chan_tx_to_rx = self.imp_chan_tx_to_rx[: Channel.FILTER_LEN]
        self.imp_chan_to_rx = self.imp_chan_to_rx[: Channel.FILTER_LEN]
        self.imp_tx_drv_to_rx = self.imp_tx_drv_to_rx[: Channel.FILTER_LEN]

        self._filt_tx_drv_to_chan.set_coeff(self.imp_tx_drv_to_chan)
        self._filt_chan_to_rx.set_coeff(self.imp_chan_to_rx)
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

        self.rx_bump = float(self.chan.out_to_port_two)
        self.rx_in = float(self._filt_chan_to_rx.run(self.rx_bump))

        self.rx.clk = self.rx_pi.clk_out
        self.rx.din = self.rx_in
        self.rx.run()

    def plot_path_impulses(self, time_unit: str = "ns") -> None:
        unit_scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}
        if time_unit not in unit_scale:
            raise ValueError(f"Unsupported time_unit '{time_unit}'. Use one of {list(unit_scale.keys())}.")
        scale = unit_scale[time_unit]

        t = np.arange(Channel.FILTER_LEN, dtype=np.float64) / self.SAMP_FREQ_HZ * scale
        fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        axs[0, 0].plot(t, self.imp_tx_drv_to_chan)
        axs[0, 0].set_title("TX drv -> channel")
        axs[0, 1].plot(t, self.imp_chan_tx_to_rx)
        axs[0, 1].set_title("Channel TX -> RX bump")
        axs[1, 0].plot(t, self.imp_chan_to_rx)
        axs[1, 0].set_title("RX bump -> RX input")
        axs[1, 1].plot(t, self.imp_tx_drv_to_rx)
        axs[1, 1].set_title("Total TX drv -> RX input")
        for ax in axs.reshape(-1):
            ax.set_xlabel(f"Time ({time_unit})")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
