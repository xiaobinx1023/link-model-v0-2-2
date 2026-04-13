from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .data_gen import Pattern


class Controller:
    def __init__(self) -> None:
        self.tx_data_gen_pattern = Pattern.ALL_ZEROS

        self._tx_main_drv_codes = np.array([63.0, 5], dtype=np.float64)
        self.tx_main_drv_inv_pol = np.array([False, True], dtype=bool)
        self.tx_main_drv_en = False

        self._tx_echo_drv_codes = np.array([57.0, 7, 2], dtype=np.float64)
        self.tx_echo_drv_inv_pol = np.array([True, False, False], dtype=bool)
        self.tx_echo_drv_en = False

        self.tx_pi_code = 0
        self.rx_clk_ofset = 4
        self.rx_slc_ref = 0.5 * 0.75
        self.rx_pd_out_gain = 0.0

        # RX equalization / decision path controls (digital cascade).
        self.rx_ctle_en = False
        self.rx_ctle_dc_gain_db = 0.0
        self.rx_ctle_peaking_gain_db = 0.0
        self.rx_ctle_peaking_freq_hz: float | None = None
        self._rx_ctle_zero_freq_hz = np.array([], dtype=np.float64)
        self._rx_ctle_pole_freq_hz = np.array([], dtype=np.float64)

        self.rx_dfe_en = False
        self._rx_dfe_taps = np.array([], dtype=np.float64)

        self.rx_slicer_sensitivity = 0.0
        self.rx_slicer_aperture_ui = 0.0

        self.is_drv_codes_changed = False

    @property
    def tx_main_drv_codes(self) -> npt.NDArray[np.float64]:
        return self._tx_main_drv_codes

    @tx_main_drv_codes.setter
    def tx_main_drv_codes(self, value: npt.ArrayLike) -> None:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.array_equal(arr, self._tx_main_drv_codes):
            self._tx_main_drv_codes = arr
            self.is_drv_codes_changed = True

    @property
    def tx_echo_drv_codes(self) -> npt.NDArray[np.float64]:
        return self._tx_echo_drv_codes

    @tx_echo_drv_codes.setter
    def tx_echo_drv_codes(self, value: npt.ArrayLike) -> None:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.array_equal(arr, self._tx_echo_drv_codes):
            self._tx_echo_drv_codes = arr
            self.is_drv_codes_changed = True

    @property
    def rx_dfe_taps(self) -> npt.NDArray[np.float64]:
        return self._rx_dfe_taps

    @rx_dfe_taps.setter
    def rx_dfe_taps(self, value: npt.ArrayLike) -> None:
        self._rx_dfe_taps = np.asarray(value, dtype=np.float64).reshape(-1)

    @property
    def rx_ctle_zero_freq_hz(self) -> npt.NDArray[np.float64]:
        return self._rx_ctle_zero_freq_hz

    @rx_ctle_zero_freq_hz.setter
    def rx_ctle_zero_freq_hz(self, value: npt.ArrayLike) -> None:
        self._rx_ctle_zero_freq_hz = np.asarray(value, dtype=np.float64).reshape(-1)

    @property
    def rx_ctle_pole_freq_hz(self) -> npt.NDArray[np.float64]:
        return self._rx_ctle_pole_freq_hz

    @rx_ctle_pole_freq_hz.setter
    def rx_ctle_pole_freq_hz(self, value: npt.ArrayLike) -> None:
        self._rx_ctle_pole_freq_hz = np.asarray(value, dtype=np.float64).reshape(-1)

    def reset_is_drv_codes_changed(self) -> None:
        self.is_drv_codes_changed = False
