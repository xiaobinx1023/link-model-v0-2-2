from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class DriverResistanceConfig:
    main_drv_scale_ohm: float = 2700.0
    echo_drv_scale_ohm: float = 5400.0
    echo_drv_series_ohm: float = 230.0


@dataclass
class SideTerminationConfig:
    # Common analog front-end resistance seen between RX and TX legs.
    sense_res_ohm: float = 160.0
    # Package/IO capacitances.
    tx_cap_f: float = 50e-15
    rx_cap_f: float = 35e-15
    # Optional package inductances (series with IO capacitances).
    tx_ind_h: float = 0.0
    rx_ind_h: float = 0.0


@dataclass
class SideTerminationState:
    r_main_drv_ohm: float
    r_echo_drv_ohm: float
    z_tx_io: npt.NDArray[np.complex128]
    z_rx_io: npt.NDArray[np.complex128]
    z_load: npt.NDArray[np.complex128]


class IOTerminationModel:
    """
    IO/termination abstraction used by Link for AFE/channel loading.
    """

    def __init__(
        self,
        z0_ohm: float = 50.0,
        driver_cfg: DriverResistanceConfig | None = None,
        master_cfg: SideTerminationConfig | None = None,
        slave_cfg: SideTerminationConfig | None = None,
    ) -> None:
        self.z0_ohm = float(z0_ohm)
        self.driver_cfg = driver_cfg if driver_cfg is not None else DriverResistanceConfig()
        self.master_cfg = master_cfg if master_cfg is not None else SideTerminationConfig()
        self.slave_cfg = slave_cfg if slave_cfg is not None else SideTerminationConfig()

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

    def _main_drv_res_ohm(self, codes: npt.ArrayLike) -> float:
        code_sum = np.sum(np.asarray(codes, dtype=np.float64))
        return float(np.float64(self.driver_cfg.main_drv_scale_ohm) / code_sum)

    def _echo_drv_res_ohm(self, codes: npt.ArrayLike) -> float:
        code_sum = np.sum(np.asarray(codes, dtype=np.float64))
        return float(
            np.float64(self.driver_cfg.echo_drv_scale_ohm) / code_sum + np.float64(self.driver_cfg.echo_drv_series_ohm)
        )

    def _cfg_for_side(self, side: str) -> SideTerminationConfig:
        s = side.strip().lower()
        if s == "master":
            return self.master_cfg
        if s == "slave":
            return self.slave_cfg
        raise ValueError(f"Unsupported side '{side}'. Use 'master' or 'slave'.")

    def build_side_state(
        self,
        freq_hz: npt.ArrayLike,
        main_drv_codes: npt.ArrayLike,
        echo_drv_codes: npt.ArrayLike,
        side: str,
    ) -> SideTerminationState:
        cfg = self._cfg_for_side(side)

        r_main = self._main_drv_res_ohm(main_drv_codes)
        r_echo = self._echo_drv_res_ohm(echo_drv_codes)
        z_tx = self._series_cap_ind_impedance(freq_hz, cap_f=cfg.tx_cap_f, ind_h=cfg.tx_ind_h)
        z_rx = self._series_cap_ind_impedance(freq_hz, cap_f=cfg.rx_cap_f, ind_h=cfg.rx_ind_h)

        z_load = self._par(
            self._par(self._par(r_echo, z_rx) + float(cfg.sense_res_ohm), r_main),
            z_tx,
        )
        return SideTerminationState(
            r_main_drv_ohm=r_main,
            r_echo_drv_ohm=r_echo,
            z_tx_io=z_tx,
            z_rx_io=z_rx,
            z_load=z_load.astype(np.complex128, copy=False),
        )

    def compute_side_transfer(
        self,
        side: str,
        side_state: SideTerminationState,
        z_in_chan: npt.ArrayLike,
    ) -> dict[str, npt.NDArray[np.complex128]]:
        cfg = self._cfg_for_side(side)
        r_sens = float(cfg.sense_res_ohm)
        r_main = float(side_state.r_main_drv_ohm)
        r_echo = float(side_state.r_echo_drv_ohm)
        z_tx = side_state.z_tx_io
        z_rx = side_state.z_rx_io
        z_in = np.asarray(z_in_chan, dtype=np.complex128)

        z_rx_echo_par = self._par(z_rx, r_echo)
        tf_chan_to_rx = z_rx_echo_par / (r_sens + z_rx_echo_par)

        tf_main_to_chan = self._par(
            r_main,
            self._par(
                (r_sens + z_rx_echo_par),
                self._par(z_tx, z_in),
            ),
        ) / r_main
        tf_main_to_rx = tf_main_to_chan * z_rx_echo_par / (z_rx_echo_par + r_sens)

        z_main_chan_par = self._par(r_main, self._par(z_tx, z_in))
        tf_echo_to_rx = self._par(
            r_echo,
            self._par(
                z_rx,
                r_sens + z_main_chan_par,
            ),
        ) / r_echo
        tf_echo_to_chan = tf_echo_to_rx * z_main_chan_par / (z_main_chan_par + r_sens)

        return {
            "tf_chan_to_rx": np.asarray(tf_chan_to_rx, dtype=np.complex128),
            "tf_main_drv_to_chan": np.asarray(tf_main_to_chan, dtype=np.complex128),
            "tf_main_drv_to_rx": np.asarray(tf_main_to_rx, dtype=np.complex128),
            "tf_echo_drv_to_rx": np.asarray(tf_echo_to_rx, dtype=np.complex128),
            "tf_echo_drv_to_chan": np.asarray(tf_echo_to_chan, dtype=np.complex128),
        }
