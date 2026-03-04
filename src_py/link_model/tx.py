from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .clock import Clock
from .data_gen import DataGen, Pattern
from .driver import Driver


class Tx:
    NUM_MAIN_DRIVERS = 2
    NUM_ECHO_DRIVERS = 3

    def __init__(self) -> None:
        self.clk = Clock()
        self.data_gen_pattern = Pattern.ALL_ZEROS

        self.main_drivers_weights = np.zeros(self.NUM_MAIN_DRIVERS, dtype=np.float64)
        self.main_drivers_inv_pol = np.zeros(self.NUM_MAIN_DRIVERS, dtype=bool)
        self.main_drivers_en = False

        self.echo_drivers_weights = np.zeros(self.NUM_ECHO_DRIVERS, dtype=np.float64)
        self.echo_drivers_inv_pol = np.zeros(self.NUM_ECHO_DRIVERS, dtype=bool)
        self.echo_drivers_en = False

        self.main_drivers_out = 0.0
        self.echo_drivers_out = 0.0
        self.data = 0

        self._data_gen = DataGen()
        self._main_drivers = [Driver() for _ in range(self.NUM_MAIN_DRIVERS)]
        self._main_drivers_delay_line = np.zeros(self.NUM_MAIN_DRIVERS, dtype=np.uint8)
        self._echo_drivers = [Driver() for _ in range(self.NUM_ECHO_DRIVERS)]
        self._echo_drivers_delay_line = np.zeros(self.NUM_ECHO_DRIVERS, dtype=np.uint8)

    def run(self) -> None:
        self._data_gen.clk = self.clk
        self._data_gen.pattern = self.data_gen_pattern
        self._data_gen.run()

        if self.clk.is_edge:
            self.data = int(self._data_gen.out)
            self._main_drivers_delay_line[1:] = self._main_drivers_delay_line[:-1]
            self._main_drivers_delay_line[0] = self.data
            self._echo_drivers_delay_line[1:] = self._echo_drivers_delay_line[:-1]
            self._echo_drivers_delay_line[0] = self.data

        self.main_drivers_out = 0.0
        if self.main_drivers_en:
            for i in range(self.NUM_MAIN_DRIVERS):
                driver_in = int(self._main_drivers_delay_line[i])
                if bool(self.main_drivers_inv_pol[i]):
                    driver_in = int(not bool(driver_in))
                drv = self._main_drivers[i]
                drv.clk = self.clk
                drv.in_ = driver_in
                drv.run()
                self.main_drivers_out += drv.out * float(self.main_drivers_weights[i])

        self.echo_drivers_out = 0.0
        if self.echo_drivers_en:
            for i in range(self.NUM_ECHO_DRIVERS):
                driver_in = int(self._echo_drivers_delay_line[i])
                if bool(self.echo_drivers_inv_pol[i]):
                    driver_in = int(not bool(driver_in))
                drv = self._echo_drivers[i]
                drv.clk = self.clk
                drv.in_ = driver_in
                drv.run()
                self.echo_drivers_out += drv.out * float(self.echo_drivers_weights[i])

    def set_main_drivers_weights(self, weights: npt.ArrayLike) -> None:
        arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if arr.size != self.NUM_MAIN_DRIVERS:
            raise ValueError("The input weights do not match the number of main drivers.")
        self.main_drivers_weights = arr

    def set_echo_drivers_weights(self, weights: npt.ArrayLike) -> None:
        arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if arr.size != self.NUM_ECHO_DRIVERS:
            raise ValueError("The input weights do not match the number of echo drivers.")
        self.echo_drivers_weights = arr

