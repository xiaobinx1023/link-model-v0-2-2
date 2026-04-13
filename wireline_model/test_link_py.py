from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import signal

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:
    tk = None
    messagebox = None
    ttk = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_py.link_model import Link
from src_py.link_model import Pattern
from src_py.link_model.controller import Controller


@dataclass
class ControllerConfig:
    tx_data_gen_pattern: Pattern
    tx_main_drv_codes: list[float]
    tx_main_drv_inv_pol: list[bool]
    tx_main_drv_en: bool
    tx_echo_drv_codes: list[float]
    tx_echo_drv_inv_pol: list[bool]
    tx_echo_drv_en: bool
    tx_pi_code: int
    rx_clk_ofset: int
    rx_slc_ref: float
    rx_pd_out_gain: float
    rx_ctle_en: bool
    rx_ctle_dc_gain_db: float
    rx_ctle_peaking_gain_db: float
    rx_ctle_peaking_freq_hz: float | None
    rx_ctle_zero_freq_hz: list[float]
    rx_ctle_pole_freq_hz: list[float]
    rx_dfe_en: bool
    rx_dfe_taps: list[float]
    rx_slicer_sensitivity: float
    rx_slicer_aperture_ui: float


@dataclass
class GuiRunConfig:
    master: ControllerConfig
    slave: ControllerConfig
    num_cycles: int
    aggressor_amplitude: float
    run_pulse_response: bool


def _controller_to_config(ctrl: Controller) -> ControllerConfig:
    return ControllerConfig(
        tx_data_gen_pattern=Pattern(ctrl.tx_data_gen_pattern),
        tx_main_drv_codes=[float(x) for x in np.asarray(ctrl.tx_main_drv_codes, dtype=np.float64)],
        tx_main_drv_inv_pol=[bool(x) for x in np.asarray(ctrl.tx_main_drv_inv_pol, dtype=bool)],
        tx_main_drv_en=bool(ctrl.tx_main_drv_en),
        tx_echo_drv_codes=[float(x) for x in np.asarray(ctrl.tx_echo_drv_codes, dtype=np.float64)],
        tx_echo_drv_inv_pol=[bool(x) for x in np.asarray(ctrl.tx_echo_drv_inv_pol, dtype=bool)],
        tx_echo_drv_en=bool(ctrl.tx_echo_drv_en),
        tx_pi_code=int(ctrl.tx_pi_code),
        rx_clk_ofset=int(ctrl.rx_clk_ofset),
        rx_slc_ref=float(ctrl.rx_slc_ref),
        rx_pd_out_gain=float(ctrl.rx_pd_out_gain),
        rx_ctle_en=bool(ctrl.rx_ctle_en),
        rx_ctle_dc_gain_db=float(ctrl.rx_ctle_dc_gain_db),
        rx_ctle_peaking_gain_db=float(ctrl.rx_ctle_peaking_gain_db),
        rx_ctle_peaking_freq_hz=(
            None if ctrl.rx_ctle_peaking_freq_hz is None else float(ctrl.rx_ctle_peaking_freq_hz)
        ),
        rx_ctle_zero_freq_hz=[float(x) for x in np.asarray(ctrl.rx_ctle_zero_freq_hz, dtype=np.float64)],
        rx_ctle_pole_freq_hz=[float(x) for x in np.asarray(ctrl.rx_ctle_pole_freq_hz, dtype=np.float64)],
        rx_dfe_en=bool(ctrl.rx_dfe_en),
        rx_dfe_taps=[float(x) for x in np.asarray(ctrl.rx_dfe_taps, dtype=np.float64)],
        rx_slicer_sensitivity=float(ctrl.rx_slicer_sensitivity),
        rx_slicer_aperture_ui=float(ctrl.rx_slicer_aperture_ui),
    )


def _apply_controller_config(ctrl: Controller, cfg: ControllerConfig) -> None:
    if len(cfg.tx_main_drv_codes) != len(ctrl.tx_main_drv_codes):
        raise ValueError(
            f"tx_main_drv_codes length mismatch: expected {len(ctrl.tx_main_drv_codes)}, got {len(cfg.tx_main_drv_codes)}"
        )
    if len(cfg.tx_main_drv_inv_pol) != len(ctrl.tx_main_drv_inv_pol):
        raise ValueError(
            f"tx_main_drv_inv_pol length mismatch: expected {len(ctrl.tx_main_drv_inv_pol)}, got {len(cfg.tx_main_drv_inv_pol)}"
        )
    if len(cfg.tx_echo_drv_codes) != len(ctrl.tx_echo_drv_codes):
        raise ValueError(
            f"tx_echo_drv_codes length mismatch: expected {len(ctrl.tx_echo_drv_codes)}, got {len(cfg.tx_echo_drv_codes)}"
        )
    if len(cfg.tx_echo_drv_inv_pol) != len(ctrl.tx_echo_drv_inv_pol):
        raise ValueError(
            f"tx_echo_drv_inv_pol length mismatch: expected {len(ctrl.tx_echo_drv_inv_pol)}, got {len(cfg.tx_echo_drv_inv_pol)}"
        )

    ctrl.tx_data_gen_pattern = cfg.tx_data_gen_pattern
    ctrl.tx_main_drv_codes = np.asarray(cfg.tx_main_drv_codes, dtype=np.float64)
    ctrl.tx_main_drv_inv_pol = np.asarray(cfg.tx_main_drv_inv_pol, dtype=bool)
    ctrl.tx_main_drv_en = bool(cfg.tx_main_drv_en)
    ctrl.tx_echo_drv_codes = np.asarray(cfg.tx_echo_drv_codes, dtype=np.float64)
    ctrl.tx_echo_drv_inv_pol = np.asarray(cfg.tx_echo_drv_inv_pol, dtype=bool)
    ctrl.tx_echo_drv_en = bool(cfg.tx_echo_drv_en)
    ctrl.tx_pi_code = int(cfg.tx_pi_code)
    ctrl.rx_clk_ofset = int(cfg.rx_clk_ofset)
    ctrl.rx_slc_ref = float(cfg.rx_slc_ref)
    ctrl.rx_pd_out_gain = float(cfg.rx_pd_out_gain)
    ctrl.rx_ctle_en = bool(cfg.rx_ctle_en)
    ctrl.rx_ctle_dc_gain_db = float(cfg.rx_ctle_dc_gain_db)
    ctrl.rx_ctle_peaking_gain_db = float(cfg.rx_ctle_peaking_gain_db)
    ctrl.rx_ctle_peaking_freq_hz = (
        None if cfg.rx_ctle_peaking_freq_hz is None else float(cfg.rx_ctle_peaking_freq_hz)
    )
    ctrl.rx_ctle_zero_freq_hz = np.asarray(cfg.rx_ctle_zero_freq_hz, dtype=np.float64)
    ctrl.rx_ctle_pole_freq_hz = np.asarray(cfg.rx_ctle_pole_freq_hz, dtype=np.float64)
    ctrl.rx_dfe_en = bool(cfg.rx_dfe_en)
    ctrl.rx_dfe_taps = np.asarray(cfg.rx_dfe_taps, dtype=np.float64)
    ctrl.rx_slicer_sensitivity = float(cfg.rx_slicer_sensitivity)
    ctrl.rx_slicer_aperture_ui = float(cfg.rx_slicer_aperture_ui)


def _parse_csv_floats(raw: str, expected_len: int, field_name: str) -> list[float]:
    items = _tokenize_numeric_list(raw)
    if len(items) != expected_len:
        raise ValueError(f"{field_name} expects {expected_len} values, got {len(items)}: '{raw}'")
    try:
        return [float(x) for x in items]
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain numeric values: '{raw}'") from exc


def _parse_csv_floats_var(raw: str, field_name: str) -> list[float]:
    items = _tokenize_numeric_list(raw)
    if len(items) == 0:
        return []
    try:
        return [float(x) for x in items]
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain numeric values: '{raw}'") from exc


def _tokenize_numeric_list(raw: str) -> list[str]:
    txt = str(raw).strip()
    if txt == "":
        return []
    for ch in "[](){}":
        txt = txt.replace(ch, " ")
    txt = txt.replace(";", ",")
    return [tok for tok in re.split(r"[,\s]+", txt) if tok]


def _parse_float_required(raw: str, field_name: str) -> float:
    txt = str(raw).strip()
    if txt == "":
        raise ValueError(f"{field_name} must be numeric and cannot be empty")
    try:
        v = float(txt)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric: '{raw}'") from exc
    if not np.isfinite(v):
        raise ValueError(f"{field_name} must be finite: '{raw}'")
    return float(v)


def _parse_int_required(raw: str, field_name: str) -> int:
    v = _parse_float_required(raw, field_name)
    v_i = int(round(v))
    if abs(v - float(v_i)) > 1e-9:
        raise ValueError(f"{field_name} must be an integer value: '{raw}'")
    return int(v_i)


def _parse_optional_float(raw: str, field_name: str) -> float | None:
    txt = str(raw).strip()
    if txt == "":
        return None
    return _parse_float_required(txt, field_name)


class LinkConfigGUI:
    def __init__(self, master_default: ControllerConfig, slave_default: ControllerConfig) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is unavailable. Please install/enable Tkinter in your Python environment.")

        self._root = tk.Tk()
        self._root.title("Link Controller Configuration")
        self._root.geometry("980x760")
        self._result: GuiRunConfig | None = None

        top = ttk.Frame(self._root, padding=10)
        top.pack(fill="both", expand=True)

        notebook = ttk.Notebook(top)
        notebook.pack(fill="both", expand=True)

        master_tab = ttk.Frame(notebook, padding=10)
        slave_tab = ttk.Frame(notebook, padding=10)
        sim_tab = ttk.Frame(notebook, padding=10)
        notebook.add(master_tab, text="Master Controller")
        notebook.add(slave_tab, text="Slave Controller")
        notebook.add(sim_tab, text="Simulation")

        self._master_vars = self._build_controller_tab(master_tab, master_default)
        self._slave_vars = self._build_controller_tab(slave_tab, slave_default)
        self._sim_vars = self._build_sim_tab(sim_tab)

        btn_row = ttk.Frame(top)
        btn_row.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_row, text="Run", command=self._on_run).pack(side="right")
        ttk.Button(btn_row, text="Cancel", command=self._on_cancel).pack(side="right", padx=(0, 8))

        self._root.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def run(self) -> GuiRunConfig | None:
        self._root.mainloop()
        return self._result

    def _on_cancel(self) -> None:
        self._result = None
        self._root.destroy()

    def _show_error(self, msg: str) -> None:
        if messagebox is not None:
            messagebox.showerror("Invalid Configuration", msg)
        else:
            print(msg)

    def _on_run(self) -> None:
        try:
            master_cfg = self._parse_controller_tab(self._master_vars, "Master")
            slave_cfg = self._parse_controller_tab(self._slave_vars, "Slave")

            num_cycles = _parse_int_required(str(self._sim_vars["num_cycles"].get()), "num_cycles")
            if num_cycles <= 0:
                raise ValueError("num_cycles must be > 0")

            aggressor_amplitude = _parse_float_required(
                str(self._sim_vars["aggressor_amplitude"].get()),
                "aggressor_amplitude",
            )
            run_pulse_response = bool(self._sim_vars["run_pulse_response"].get())

            self._result = GuiRunConfig(
                master=master_cfg,
                slave=slave_cfg,
                num_cycles=num_cycles,
                aggressor_amplitude=aggressor_amplitude,
                run_pulse_response=run_pulse_response,
            )
            self._root.destroy()
        except Exception as exc:
            self._show_error(str(exc))

    def _build_controller_tab(self, parent: "ttk.Frame", default: ControllerConfig) -> dict[str, object]:
        vars_out: dict[str, object] = {}
        pattern_names = [p.name for p in Pattern]

        row = 0
        ttk.Label(parent, text="tx_data_gen_pattern").grid(row=row, column=0, sticky="w", pady=4)
        pattern_var = tk.StringVar(value=default.tx_data_gen_pattern.name)
        pattern_box = ttk.Combobox(parent, textvariable=pattern_var, values=pattern_names, state="readonly", width=24)
        pattern_box.grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["tx_data_gen_pattern"] = pattern_var
        row += 1

        ttk.Label(parent, text="tx_main_drv_codes (csv)").grid(row=row, column=0, sticky="w", pady=4)
        main_codes_var = tk.StringVar(value=", ".join(f"{x:g}" for x in default.tx_main_drv_codes))
        ttk.Entry(parent, textvariable=main_codes_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["tx_main_drv_codes"] = main_codes_var
        row += 1

        ttk.Label(parent, text="tx_main_drv_inv_pol").grid(row=row, column=0, sticky="w", pady=4)
        main_inv_frame = ttk.Frame(parent)
        main_inv_frame.grid(row=row, column=1, sticky="w", pady=4)
        main_inv_vars: list[tk.BooleanVar] = []
        for i, val in enumerate(default.tx_main_drv_inv_pol):
            v = tk.BooleanVar(value=bool(val))
            ttk.Checkbutton(main_inv_frame, text=f"[{i}]", variable=v).pack(side="left", padx=(0, 8))
            main_inv_vars.append(v)
        vars_out["tx_main_drv_inv_pol"] = main_inv_vars
        row += 1

        main_en_var = tk.BooleanVar(value=default.tx_main_drv_en)
        ttk.Checkbutton(parent, text="tx_main_drv_en", variable=main_en_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=4
        )
        vars_out["tx_main_drv_en"] = main_en_var
        row += 1

        ttk.Label(parent, text="tx_echo_drv_codes (csv)").grid(row=row, column=0, sticky="w", pady=4)
        echo_codes_var = tk.StringVar(value=", ".join(f"{x:g}" for x in default.tx_echo_drv_codes))
        ttk.Entry(parent, textvariable=echo_codes_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["tx_echo_drv_codes"] = echo_codes_var
        row += 1

        ttk.Label(parent, text="tx_echo_drv_inv_pol").grid(row=row, column=0, sticky="w", pady=4)
        echo_inv_frame = ttk.Frame(parent)
        echo_inv_frame.grid(row=row, column=1, sticky="w", pady=4)
        echo_inv_vars: list[tk.BooleanVar] = []
        for i, val in enumerate(default.tx_echo_drv_inv_pol):
            v = tk.BooleanVar(value=bool(val))
            ttk.Checkbutton(echo_inv_frame, text=f"[{i}]", variable=v).pack(side="left", padx=(0, 8))
            echo_inv_vars.append(v)
        vars_out["tx_echo_drv_inv_pol"] = echo_inv_vars
        row += 1

        echo_en_var = tk.BooleanVar(value=default.tx_echo_drv_en)
        ttk.Checkbutton(parent, text="tx_echo_drv_en", variable=echo_en_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=4
        )
        vars_out["tx_echo_drv_en"] = echo_en_var
        row += 1

        ttk.Label(parent, text="tx_pi_code").grid(row=row, column=0, sticky="w", pady=4)
        tx_pi_code_var = tk.StringVar(value=str(default.tx_pi_code))
        ttk.Entry(parent, textvariable=tx_pi_code_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["tx_pi_code"] = tx_pi_code_var
        row += 1

        ttk.Label(parent, text="rx_clk_ofset").grid(row=row, column=0, sticky="w", pady=4)
        rx_clk_ofset_var = tk.StringVar(value=str(default.rx_clk_ofset))
        ttk.Entry(parent, textvariable=rx_clk_ofset_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["rx_clk_ofset"] = rx_clk_ofset_var
        row += 1

        ttk.Label(parent, text="rx_slc_ref").grid(row=row, column=0, sticky="w", pady=4)
        rx_slc_ref_var = tk.StringVar(value=f"{default.rx_slc_ref:g}")
        ttk.Entry(parent, textvariable=rx_slc_ref_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["rx_slc_ref"] = rx_slc_ref_var
        row += 1

        ttk.Label(parent, text="rx_pd_out_gain").grid(row=row, column=0, sticky="w", pady=4)
        rx_pd_out_gain_var = tk.StringVar(value=f"{default.rx_pd_out_gain:g}")
        ttk.Entry(parent, textvariable=rx_pd_out_gain_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["rx_pd_out_gain"] = rx_pd_out_gain_var
        row += 1

        rx_ctle_en_var = tk.BooleanVar(value=default.rx_ctle_en)
        ttk.Checkbutton(parent, text="rx_ctle_en", variable=rx_ctle_en_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=4
        )
        vars_out["rx_ctle_en"] = rx_ctle_en_var
        row += 1

        ttk.Label(parent, text="rx_ctle_dc_gain_db").grid(row=row, column=0, sticky="w", pady=4)
        rx_ctle_dc_gain_db_var = tk.StringVar(value=f"{default.rx_ctle_dc_gain_db:g}")
        ttk.Entry(parent, textvariable=rx_ctle_dc_gain_db_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["rx_ctle_dc_gain_db"] = rx_ctle_dc_gain_db_var
        row += 1

        ttk.Label(parent, text="rx_ctle_peaking_gain_db (gain-peak mode)").grid(row=row, column=0, sticky="w", pady=4)
        rx_ctle_peaking_gain_db_var = tk.StringVar(value=f"{default.rx_ctle_peaking_gain_db:g}")
        ttk.Entry(parent, textvariable=rx_ctle_peaking_gain_db_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["rx_ctle_peaking_gain_db"] = rx_ctle_peaking_gain_db_var
        row += 1

        ttk.Label(parent, text="rx_ctle_peaking_freq_hz (gain-peak mode)").grid(row=row, column=0, sticky="w", pady=4)
        peak_freq_text = "" if default.rx_ctle_peaking_freq_hz is None else f"{default.rx_ctle_peaking_freq_hz:g}"
        rx_ctle_peaking_freq_hz_var = tk.StringVar(value=peak_freq_text)
        ttk.Entry(parent, textvariable=rx_ctle_peaking_freq_hz_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["rx_ctle_peaking_freq_hz"] = rx_ctle_peaking_freq_hz_var
        row += 1

        ttk.Label(parent, text="rx_ctle_zero_freq_hz (csv)").grid(row=row, column=0, sticky="w", pady=4)
        rx_ctle_zero_freq_hz_var = tk.StringVar(value=", ".join(f"{x:g}" for x in default.rx_ctle_zero_freq_hz))
        ttk.Entry(parent, textvariable=rx_ctle_zero_freq_hz_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["rx_ctle_zero_freq_hz"] = rx_ctle_zero_freq_hz_var
        row += 1

        ttk.Label(parent, text="rx_ctle_pole_freq_hz (csv)").grid(row=row, column=0, sticky="w", pady=4)
        rx_ctle_pole_freq_hz_var = tk.StringVar(value=", ".join(f"{x:g}" for x in default.rx_ctle_pole_freq_hz))
        ttk.Entry(parent, textvariable=rx_ctle_pole_freq_hz_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["rx_ctle_pole_freq_hz"] = rx_ctle_pole_freq_hz_var
        row += 1

        rx_dfe_en_var = tk.BooleanVar(value=default.rx_dfe_en)
        ttk.Checkbutton(parent, text="rx_dfe_en", variable=rx_dfe_en_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=4
        )
        vars_out["rx_dfe_en"] = rx_dfe_en_var
        row += 1

        ttk.Label(parent, text="rx_dfe_taps (csv)").grid(row=row, column=0, sticky="w", pady=4)
        rx_dfe_taps_var = tk.StringVar(value=", ".join(f"{x:g}" for x in default.rx_dfe_taps))
        ttk.Entry(parent, textvariable=rx_dfe_taps_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["rx_dfe_taps"] = rx_dfe_taps_var
        row += 1

        ttk.Label(parent, text="rx_slicer_sensitivity").grid(row=row, column=0, sticky="w", pady=4)
        rx_slicer_sensitivity_var = tk.StringVar(value=f"{default.rx_slicer_sensitivity:g}")
        ttk.Entry(parent, textvariable=rx_slicer_sensitivity_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["rx_slicer_sensitivity"] = rx_slicer_sensitivity_var
        row += 1

        ttk.Label(parent, text="rx_slicer_aperture_ui").grid(row=row, column=0, sticky="w", pady=4)
        rx_slicer_aperture_ui_var = tk.StringVar(value=f"{default.rx_slicer_aperture_ui:g}")
        ttk.Entry(parent, textvariable=rx_slicer_aperture_ui_var).grid(row=row, column=1, sticky="ew", pady=4)
        vars_out["rx_slicer_aperture_ui"] = rx_slicer_aperture_ui_var

        parent.columnconfigure(1, weight=1)
        return vars_out

    def _build_sim_tab(self, parent: "ttk.Frame") -> dict[str, object]:
        vars_out: dict[str, object] = {}

        ttk.Label(parent, text="num_cycles").grid(row=0, column=0, sticky="w", pady=4)
        num_cycles_var = tk.StringVar(value="10000")
        ttk.Entry(parent, textvariable=num_cycles_var).grid(row=0, column=1, sticky="ew", pady=4)
        vars_out["num_cycles"] = num_cycles_var

        ttk.Label(parent, text="aggressor_amplitude").grid(row=1, column=0, sticky="w", pady=4)
        aggressor_amp_var = tk.StringVar(value="0.2")
        ttk.Entry(parent, textvariable=aggressor_amp_var).grid(row=1, column=1, sticky="ew", pady=4)
        vars_out["aggressor_amplitude"] = aggressor_amp_var

        run_pulse_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Run 1 UI pulse-response setup first", variable=run_pulse_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=4
        )
        vars_out["run_pulse_response"] = run_pulse_var

        parent.columnconfigure(1, weight=1)
        return vars_out

    def _parse_controller_tab(self, vars_map: dict[str, object], name: str) -> ControllerConfig:
        pattern_name = str(vars_map["tx_data_gen_pattern"].get()).strip()
        if pattern_name not in Pattern.__members__:
            raise ValueError(f"{name}: invalid tx_data_gen_pattern '{pattern_name}'")
        tx_data_gen_pattern = Pattern[pattern_name]

        tx_main_drv_codes = _parse_csv_floats(
            str(vars_map["tx_main_drv_codes"].get()),
            len(vars_map["tx_main_drv_inv_pol"]),
            f"{name}.tx_main_drv_codes",
        )
        tx_main_drv_inv_pol = [bool(v.get()) for v in vars_map["tx_main_drv_inv_pol"]]
        tx_main_drv_en = bool(vars_map["tx_main_drv_en"].get())

        tx_echo_drv_codes = _parse_csv_floats(
            str(vars_map["tx_echo_drv_codes"].get()),
            len(vars_map["tx_echo_drv_inv_pol"]),
            f"{name}.tx_echo_drv_codes",
        )
        tx_echo_drv_inv_pol = [bool(v.get()) for v in vars_map["tx_echo_drv_inv_pol"]]
        tx_echo_drv_en = bool(vars_map["tx_echo_drv_en"].get())

        tx_pi_code = _parse_int_required(str(vars_map["tx_pi_code"].get()), f"{name}.tx_pi_code")
        if tx_pi_code < 0 or tx_pi_code > 127:
            raise ValueError(f"{name}.tx_pi_code must be in [0, 127]")

        rx_clk_ofset = _parse_int_required(str(vars_map["rx_clk_ofset"].get()), f"{name}.rx_clk_ofset")
        rx_slc_ref = _parse_float_required(str(vars_map["rx_slc_ref"].get()), f"{name}.rx_slc_ref")
        rx_pd_out_gain = _parse_float_required(str(vars_map["rx_pd_out_gain"].get()), f"{name}.rx_pd_out_gain")
        rx_ctle_en = bool(vars_map["rx_ctle_en"].get())
        rx_ctle_dc_gain_db = _parse_float_required(
            str(vars_map["rx_ctle_dc_gain_db"].get()),
            f"{name}.rx_ctle_dc_gain_db",
        )
        rx_ctle_peaking_gain_db = _parse_float_required(
            str(vars_map["rx_ctle_peaking_gain_db"].get()),
            f"{name}.rx_ctle_peaking_gain_db",
        )
        rx_ctle_peaking_freq_hz = _parse_optional_float(
            str(vars_map["rx_ctle_peaking_freq_hz"].get()),
            f"{name}.rx_ctle_peaking_freq_hz",
        )
        rx_ctle_zero_freq_hz = _parse_csv_floats_var(
            str(vars_map["rx_ctle_zero_freq_hz"].get()),
            f"{name}.rx_ctle_zero_freq_hz",
        )
        rx_ctle_pole_freq_hz = _parse_csv_floats_var(
            str(vars_map["rx_ctle_pole_freq_hz"].get()),
            f"{name}.rx_ctle_pole_freq_hz",
        )
        rx_dfe_en = bool(vars_map["rx_dfe_en"].get())
        rx_dfe_taps = _parse_csv_floats_var(str(vars_map["rx_dfe_taps"].get()), f"{name}.rx_dfe_taps")
        rx_slicer_sensitivity = _parse_float_required(
            str(vars_map["rx_slicer_sensitivity"].get()),
            f"{name}.rx_slicer_sensitivity",
        )
        rx_slicer_aperture_ui = _parse_float_required(
            str(vars_map["rx_slicer_aperture_ui"].get()),
            f"{name}.rx_slicer_aperture_ui",
        )

        return ControllerConfig(
            tx_data_gen_pattern=tx_data_gen_pattern,
            tx_main_drv_codes=tx_main_drv_codes,
            tx_main_drv_inv_pol=tx_main_drv_inv_pol,
            tx_main_drv_en=tx_main_drv_en,
            tx_echo_drv_codes=tx_echo_drv_codes,
            tx_echo_drv_inv_pol=tx_echo_drv_inv_pol,
            tx_echo_drv_en=tx_echo_drv_en,
            tx_pi_code=tx_pi_code,
            rx_clk_ofset=rx_clk_ofset,
            rx_slc_ref=rx_slc_ref,
            rx_pd_out_gain=rx_pd_out_gain,
            rx_ctle_en=rx_ctle_en,
            rx_ctle_dc_gain_db=rx_ctle_dc_gain_db,
            rx_ctle_peaking_gain_db=rx_ctle_peaking_gain_db,
            rx_ctle_peaking_freq_hz=rx_ctle_peaking_freq_hz,
            rx_ctle_zero_freq_hz=rx_ctle_zero_freq_hz,
            rx_ctle_pole_freq_hz=rx_ctle_pole_freq_hz,
            rx_dfe_en=rx_dfe_en,
            rx_dfe_taps=rx_dfe_taps,
            rx_slicer_sensitivity=rx_slicer_sensitivity,
            rx_slicer_aperture_ui=rx_slicer_aperture_ui,
        )


def get_gui_run_config(master_default: ControllerConfig, slave_default: ControllerConfig) -> GuiRunConfig | None:
    gui = LinkConfigGUI(master_default, slave_default)
    return gui.run()


def _get_pulse_monitor_definitions() -> dict[str, tuple[str, Callable[[Link], float]]]:
    return {
        "master_tx_main_out": ("Master TX main out", lambda lk: float(lk.master_tx.main_drivers_out)),
        "master_tx_echo_out": ("Master TX echo out", lambda lk: float(lk.master_tx.echo_drivers_out)),
        "master_afe_out_to_bump": ("Master AFE out->bump", lambda lk: float(lk.master_afe.out_to_bump)),
        "slave_afe_out_to_bump": ("Slave AFE out->bump", lambda lk: float(lk.slave_afe.out_to_bump)),
        "channel_out_to_master": ("Channel out->master", lambda lk: float(lk.chan.out_to_port_one)),
        "channel_out_to_slave": ("Channel out->slave", lambda lk: float(lk.chan.out_to_port_two)),
        "master_afe_rx_in": ("Master RX in (pre-CTLE)", lambda lk: float(lk.master_afe.out_to_rx)),
        "slave_afe_rx_in": ("Slave RX in (pre-CTLE)", lambda lk: float(lk.slave_afe.out_to_rx)),
        "master_rx_post_ctle": ("Master RX post-CTLE", lambda lk: float(lk.master_rx.din_ctle)),
        "slave_rx_post_ctle": ("Slave RX post-CTLE", lambda lk: float(lk.slave_rx.din_ctle)),
        "master_rx_post_dfe": ("Master RX post-DFE", lambda lk: float(lk.master_rx.din_eq)),
        "slave_rx_post_dfe": ("Slave RX post-DFE", lambda lk: float(lk.slave_rx.din_eq)),
        "master_rx_post_aperture": ("Master RX post-aperture", lambda lk: float(lk.master_rx.din_apertured)),
        "slave_rx_post_aperture": ("Slave RX post-aperture", lambda lk: float(lk.slave_rx.din_apertured)),
    }


def _resolve_pulse_monitors(
    monitor_points: list[str] | None,
) -> list[tuple[str, str, Callable[[Link], float]]]:
    defs = _get_pulse_monitor_definitions()
    if monitor_points is None:
        keys = list(defs.keys())
    else:
        keys = []
        seen: set[str] = set()
        for raw in monitor_points:
            key = str(raw).strip()
            if key == "" or key in seen:
                continue
            seen.add(key)
            keys.append(key)
        unknown = [k for k in keys if k not in defs]
        if unknown:
            raise ValueError(
                f"Unknown pulse monitor point(s): {unknown}. Available: {list(defs.keys())}"
            )
    return [(k, defs[k][0], defs[k][1]) for k in keys]


def _plot_pulse_monitor_traces(
    t_axis_ns: npt.NDArray[np.float64],
    traces: dict[str, npt.NDArray[np.float64]],
    monitors: list[tuple[str, str, Callable[[Link], float]]],
    title: str,
    pulse_start_ns: float,
    pulse_end_ns: float,
) -> None:
    if len(monitors) == 0:
        return
    n = len(monitors)
    n_cols = 2 if n > 1 else 1
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, max(3.0, 2.6 * n_rows)), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)

    for i, (key, label, _) in enumerate(monitors):
        ax = axes_arr[i]
        y = traces[key]
        ax.plot(t_axis_ns, y, linewidth=1.2)
        ax.axvspan(pulse_start_ns, pulse_end_ns, color="orange", alpha=0.15)
        ax.set_title(label)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    for i in range(n, axes_arr.size):
        fig.delaxes(axes_arr[i])
    for i in range(min(n, axes_arr.size)):
        axes_arr[i].set_xlabel("Time (ns)")

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))


def run_pulse_response_setup(
    link: Link,
    monitor_points: list[str] | None = None,
) -> dict[str, object]:
    """
    Pulse response test setup:
    1) Victim main path: send 1 UI pulse from victim TX main driver.
    2) Aggressor path: send 1 UI pulse from one aggressor source.
    Capture selected monitor points for both cases.
    """
    monitors = _resolve_pulse_monitors(monitor_points)
    ui_samples = int(round(link.SAMP_FREQ_HZ / link.CLK_FREQ_HZ))
    warmup = 128
    pulse_len = ui_samples  # 1 UI pulse in this oversampled model.
    tail = 256
    total_cycles = warmup + pulse_len + tail
    pulse_start = warmup
    pulse_end = warmup + pulse_len
    t_axis_ns = np.arange(total_cycles, dtype=np.float64) / link.SAMP_FREQ_HZ * 1e9
    pulse_start_ns = pulse_start / link.SAMP_FREQ_HZ * 1e9
    pulse_end_ns = pulse_end / link.SAMP_FREQ_HZ * 1e9

    # Save existing setup so the caller can continue running normal tests.
    saved = {
        "master_tx_echo_drv_en": bool(link.master_ctrl.tx_echo_drv_en),
        "slave_tx_echo_drv_en": bool(link.slave_ctrl.tx_echo_drv_en),
        "master_tx_main_drv_en": bool(link.master_ctrl.tx_main_drv_en),
        "slave_tx_main_drv_en": bool(link.slave_ctrl.tx_main_drv_en),
        "master_tx_data_gen_pattern": Pattern(link.master_ctrl.tx_data_gen_pattern),
        "slave_tx_data_gen_pattern": Pattern(link.slave_ctrl.tx_data_gen_pattern),
        "master_rx_pd_out_gain": float(link.master_ctrl.rx_pd_out_gain),
        "slave_rx_pd_out_gain": float(link.slave_ctrl.rx_pd_out_gain),
        "aggressor_src": dict(link.aggressor_port_src),
    }

    main_monitor_traces: dict[str, list[float]] = {key: [] for key, _, _ in monitors}
    aggr_monitor_traces: dict[str, list[float]] = {key: [] for key, _, _ in monitors}
    aggr_port: int | None = None

    def capture_into(dst: dict[str, list[float]]) -> None:
        for key, _, getter in monitors:
            dst[key].append(float(getter(link)))

    try:
        # Common baseline setup: disable extra contributors.
        link.master_ctrl.tx_echo_drv_en = False
        link.slave_ctrl.tx_echo_drv_en = False
        link.slave_ctrl.tx_main_drv_en = False
        link.master_ctrl.rx_pd_out_gain = 0.0
        link.slave_ctrl.rx_pd_out_gain = 0.0
        link.slave_ctrl.tx_data_gen_pattern = Pattern.ALL_ZEROS
        if link.aggressor_ports:
            link.set_aggressor_sources({p: 0.0 for p in link.aggressor_ports})

        # -----------------------------
        # Case A: Victim main 1-UI pulse
        # -----------------------------
        link.master_ctrl.tx_main_drv_en = True
        link.master_ctrl.tx_data_gen_pattern = Pattern.ALL_ZEROS
        for t in range(total_cycles):
            if pulse_start <= t < pulse_end:
                link.master_ctrl.tx_data_gen_pattern = Pattern.ALL_ONES
            else:
                link.master_ctrl.tx_data_gen_pattern = Pattern.ALL_ZEROS
            link.run()
            capture_into(main_monitor_traces)

        # -----------------------------
        # Case B: Aggressor 1-UI pulse
        # -----------------------------
        link.master_ctrl.tx_main_drv_en = False
        link.master_ctrl.tx_data_gen_pattern = Pattern.ALL_ZEROS

        if link.aggressor_ports:
            aggr_rank = []
            for p in link.aggressor_ports:
                pr = link.get_aggressor_victim_pulse_response(p, include_total=False)
                peak = max(
                    float(np.max(np.abs(pr["next_impulse"]))),
                    float(np.max(np.abs(pr["fext_impulse"]))),
                )
                aggr_rank.append((p, peak))
            aggr_rank.sort(key=lambda x: x[1], reverse=True)
            aggr_port = int(aggr_rank[0][0])
            # Keep all other aggressors at zero for isolated pulse response.
            link.set_aggressor_sources({p: 0.0 for p in link.aggressor_ports})

        for t in range(total_cycles):
            if aggr_port is not None and pulse_start <= t < pulse_end:
                link.set_aggressor_sources({aggr_port: 1.0})
            elif aggr_port is not None:
                link.set_aggressor_sources({aggr_port: 0.0})
            link.run()
            capture_into(aggr_monitor_traces)

        main_traces_np = {k: np.asarray(v, dtype=np.float64) for k, v in main_monitor_traces.items()}
        aggr_traces_np = {k: np.asarray(v, dtype=np.float64) for k, v in aggr_monitor_traces.items()}

        _plot_pulse_monitor_traces(
            t_axis_ns=t_axis_ns,
            traces=main_traces_np,
            monitors=monitors,
            title="Victim Main-Path 1 UI Pulse Response (Monitor Points)",
            pulse_start_ns=pulse_start_ns,
            pulse_end_ns=pulse_end_ns,
        )
        aggr_title = "Aggressor 1 UI Pulse Response (Monitor Points)"
        if aggr_port is not None:
            aggr_title += f" - Port {aggr_port}"
        _plot_pulse_monitor_traces(
            t_axis_ns=t_axis_ns,
            traces=aggr_traces_np,
            monitors=monitors,
            title=aggr_title,
            pulse_start_ns=pulse_start_ns,
            pulse_end_ns=pulse_end_ns,
        )

        # Keep legacy summary plot of victim RX-in waveforms for quick comparison.
        if "master_afe_rx_in" in main_traces_np and "slave_afe_rx_in" in main_traces_np:
            fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            axs[0].plot(t_axis_ns, main_traces_np["master_afe_rx_in"], label="Main pulse -> Master RX in")
            axs[0].plot(t_axis_ns, main_traces_np["slave_afe_rx_in"], label="Main pulse -> Slave RX in")
            axs[0].axvspan(pulse_start_ns, pulse_end_ns, color="orange", alpha=0.15, label="1 UI pulse window")
            axs[0].set_title("Victim Main-Path 1 UI Pulse Response (RX In)")
            axs[0].set_ylabel("Amplitude")
            axs[0].grid(True, alpha=0.3)
            axs[0].legend()

            axs[1].plot(t_axis_ns, aggr_traces_np["master_afe_rx_in"], label="Aggressor pulse -> Master RX in")
            axs[1].plot(t_axis_ns, aggr_traces_np["slave_afe_rx_in"], label="Aggressor pulse -> Slave RX in")
            axs[1].axvspan(pulse_start_ns, pulse_end_ns, color="orange", alpha=0.15, label="1 UI pulse window")
            title = "Aggressor 1 UI Pulse Response (RX In)"
            if aggr_port is not None:
                title += f" - Port {aggr_port}"
            axs[1].set_title(title)
            axs[1].set_xlabel("Time (ns)")
            axs[1].set_ylabel("Amplitude")
            axs[1].grid(True, alpha=0.3)
            axs[1].legend()
            fig.tight_layout()
    finally:
        # Restore caller setup.
        link.master_ctrl.tx_echo_drv_en = saved["master_tx_echo_drv_en"]
        link.slave_ctrl.tx_echo_drv_en = saved["slave_tx_echo_drv_en"]
        link.master_ctrl.tx_main_drv_en = saved["master_tx_main_drv_en"]
        link.slave_ctrl.tx_main_drv_en = saved["slave_tx_main_drv_en"]
        link.master_ctrl.tx_data_gen_pattern = saved["master_tx_data_gen_pattern"]
        link.slave_ctrl.tx_data_gen_pattern = saved["slave_tx_data_gen_pattern"]
        link.master_ctrl.rx_pd_out_gain = saved["master_rx_pd_out_gain"]
        link.slave_ctrl.rx_pd_out_gain = saved["slave_rx_pd_out_gain"]
        if link.aggressor_ports:
            restored_sources = {
                p: float(saved["aggressor_src"].get(p, 0.0))
                for p in link.aggressor_ports
            }
            link.set_aggressor_sources(restored_sources)

    main_traces_np = {k: np.asarray(v, dtype=np.float64) for k, v in main_monitor_traces.items()}
    aggr_traces_np = {k: np.asarray(v, dtype=np.float64) for k, v in aggr_monitor_traces.items()}

    result = {
        "time_ns": t_axis_ns,
        "main_master_rx_in": main_traces_np.get("master_afe_rx_in", np.array([], dtype=np.float64)),
        "main_slave_rx_in": main_traces_np.get("slave_afe_rx_in", np.array([], dtype=np.float64)),
        "aggr_master_rx_in": aggr_traces_np.get("master_afe_rx_in", np.array([], dtype=np.float64)),
        "aggr_slave_rx_in": aggr_traces_np.get("slave_afe_rx_in", np.array([], dtype=np.float64)),
        "monitor_keys": [key for key, _, _ in monitors],
        "monitor_labels": {key: label for key, label, _ in monitors},
        "main_monitor_traces": main_traces_np,
        "aggr_monitor_traces": aggr_traces_np,
        "ui_samples": ui_samples,
        "pulse_start_sample": pulse_start,
        "pulse_end_sample": pulse_end,
        "selected_aggressor_port": aggr_port,
    }
    print(
        "Pulse-response setup complete:",
        result["ui_samples"],
        result["selected_aggressor_port"],
        "monitor_keys=",
        result["monitor_keys"],
    )
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    chan_file = ROOT / "data" / "A1_Combined_models_Slice_C_clocks_Slice_D_Data.s10p"

    link = Link(
        chan_file=chan_file,
        chan_port_one_sel=5,
        chan_port_two_sel=6,
        channel_pairs=[(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)],
        aggressor_ports=None,
    )
    print("Channel pairs:", link.channel_pairs, "Victim pair:", link.victim_pair)
    print("Default aggressor ports (all non-victim):", link.aggressor_ports)

    default_master = _controller_to_config(link.master_ctrl)
    default_slave = _controller_to_config(link.slave_ctrl)
    gui_cfg = get_gui_run_config(default_master, default_slave)
    if gui_cfg is None:
        print("GUI cancelled. Exiting.")
        return

    _apply_controller_config(link.master_ctrl, gui_cfg.master)
    _apply_controller_config(link.slave_ctrl, gui_cfg.slave)

    print("Master controller config:", gui_cfg.master)
    print("Slave controller config:", gui_cfg.slave)

    if link.master_ctrl.rx_ctle_en or link.slave_ctrl.rx_ctle_en:
        link.master_rx.set_ctle_logging(enabled=True, level=logging.INFO, log_response_queries=True)
        link.slave_rx.set_ctle_logging(enabled=True, level=logging.INFO, log_response_queries=True)
        link.apply_rx_controls()
        fig_ctle, axs_ctle = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        t_m, h_m = link.master_rx.get_ctle_impulse_response(n_samples=256)
        t_s, h_s = link.slave_rx.get_ctle_impulse_response(n_samples=256)
        axs_ctle[0].plot(t_m * 1e9, h_m)
        axs_ctle[0].set_title("Master RX CTLE Impulse Response")
        axs_ctle[0].set_ylabel("Amplitude")
        axs_ctle[0].grid(True, alpha=0.3)
        axs_ctle[1].plot(t_s * 1e9, h_s)
        axs_ctle[1].set_title("Slave RX CTLE Impulse Response")
        axs_ctle[1].set_xlabel("Time (ns)")
        axs_ctle[1].set_ylabel("Amplitude")
        axs_ctle[1].grid(True, alpha=0.3)
        fig_ctle.tight_layout()

        fig_ctle_fr, axs_ctle_fr = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
        f_m_hz, h_m_f = link.master_rx.get_ctle_frequency_response(n_points=2048)
        f_s_hz, h_s_f = link.slave_rx.get_ctle_frequency_response(n_points=2048)
        m_info = link.master_rx.get_ctle_design_info()
        s_info = link.slave_rx.get_ctle_design_info()

        m_mask = f_m_hz > 0.0
        s_mask = f_s_hz > 0.0
        m_mag_db = 20.0 * np.log10(np.maximum(np.abs(h_m_f), 1e-15))
        s_mag_db = 20.0 * np.log10(np.maximum(np.abs(h_s_f), 1e-15))
        m_phase_deg = np.unwrap(np.angle(h_m_f)) * 180.0 / np.pi
        s_phase_deg = np.unwrap(np.angle(h_s_f)) * 180.0 / np.pi

        ax_mm = axs_ctle_fr[0, 0]
        ax_ms = axs_ctle_fr[1, 0]
        ax_sm = axs_ctle_fr[0, 1]
        ax_ss = axs_ctle_fr[1, 1]

        ax_mm.semilogx(f_m_hz[m_mask] * 1e-9, m_mag_db[m_mask], color="tab:blue")
        ax_ms.semilogx(f_m_hz[m_mask] * 1e-9, m_phase_deg[m_mask], color="tab:blue")
        ax_sm.semilogx(f_s_hz[s_mask] * 1e-9, s_mag_db[s_mask], color="tab:orange")
        ax_ss.semilogx(f_s_hz[s_mask] * 1e-9, s_phase_deg[s_mask], color="tab:orange")

        def _add_ref_lines(ax, freqs_hz: list[float], color: str, style: str, label: str) -> None:
            for i, f in enumerate([float(x) for x in freqs_hz if float(x) > 0.0]):
                ax.axvline(
                    f * 1e-9,
                    color=color,
                    linestyle=style,
                    linewidth=1.0,
                    alpha=0.35,
                    label=label if i == 0 else None,
                )

        _add_ref_lines(
            ax_mm,
            [float(x) for x in np.asarray(m_info.get("zero_freq_hz_effective", []), dtype=np.float64)],
            "tab:blue",
            ":",
            "Master zero(s)",
        )
        _add_ref_lines(
            ax_mm,
            [float(x) for x in np.asarray(m_info.get("pole_freq_hz_effective", []), dtype=np.float64)],
            "tab:blue",
            "--",
            "Master pole(s)",
        )
        _add_ref_lines(
            ax_sm,
            [float(x) for x in np.asarray(s_info.get("zero_freq_hz_effective", []), dtype=np.float64)],
            "tab:orange",
            ":",
            "Slave zero(s)",
        )
        _add_ref_lines(
            ax_sm,
            [float(x) for x in np.asarray(s_info.get("pole_freq_hz_effective", []), dtype=np.float64)],
            "tab:orange",
            "--",
            "Slave pole(s)",
        )

        ax_mm.set_title("Master CTLE Magnitude")
        ax_mm.set_ylabel("Magnitude (dB)")
        ax_mm.grid(True, which="both", alpha=0.3)
        ax_mm.legend()

        ax_ms.set_title("Master CTLE Phase")
        ax_ms.set_xlabel("Frequency (GHz)")
        ax_ms.set_ylabel("Phase (deg)")
        ax_ms.grid(True, which="both", alpha=0.3)

        ax_sm.set_title("Slave CTLE Magnitude")
        ax_sm.set_ylabel("Magnitude (dB)")
        ax_sm.grid(True, which="both", alpha=0.3)
        ax_sm.legend()

        ax_ss.set_title("Slave CTLE Phase")
        ax_ss.set_xlabel("Frequency (GHz)")
        ax_ss.set_ylabel("Phase (deg)")
        ax_ss.grid(True, which="both", alpha=0.3)
        fig_ctle_fr.tight_layout()

        m_metrics = link.master_rx.get_ctle_response_metrics(n_points=4096)
        s_metrics = link.slave_rx.get_ctle_response_metrics(n_points=4096)
        print("Master CTLE mode:", m_info.get("mode"))
        print("Slave CTLE mode:", s_info.get("mode"))
        print("Master CTLE coeff b_z:", m_info.get("b_z"))
        print("Master CTLE coeff a_z:", m_info.get("a_z"))
        print("Slave CTLE coeff b_z:", s_info.get("b_z"))
        print("Slave CTLE coeff a_z:", s_info.get("a_z"))

        # Sanity check: plotted responses should match response from current coeffs.
        imp = np.zeros_like(h_m)
        imp[0] = 1.0
        h_m_from_coeff = signal.lfilter(np.asarray(m_info["b_z"]), np.asarray(m_info["a_z"]), imp)
        h_s_from_coeff = signal.lfilter(np.asarray(s_info["b_z"]), np.asarray(s_info["a_z"]), imp)
        imp_err_m = float(np.max(np.abs(h_m_from_coeff - h_m)))
        imp_err_s = float(np.max(np.abs(h_s_from_coeff - h_s)))

        f_m_chk, h_m_chk = signal.freqz(
            np.asarray(m_info["b_z"]),
            np.asarray(m_info["a_z"]),
            worN=2048,
            fs=float(link.SAMP_FREQ_HZ),
        )
        f_s_chk, h_s_chk = signal.freqz(
            np.asarray(s_info["b_z"]),
            np.asarray(s_info["a_z"]),
            worN=2048,
            fs=float(link.SAMP_FREQ_HZ),
        )
        fr_err_m = float(np.max(np.abs(h_m_chk - h_m_f)))
        fr_err_s = float(np.max(np.abs(h_s_chk - h_s_f)))
        print("Master CTLE response consistency (impulse_max_err, freq_max_err):", imp_err_m, fr_err_m)
        print("Slave CTLE response consistency (impulse_max_err, freq_max_err):", imp_err_s, fr_err_s)

        print("Master CTLE measured response:", m_metrics)
        print("Slave CTLE measured response:", s_metrics)

    if link.aggressor_ports:
        link.set_aggressor_sources({p: gui_cfg.aggressor_amplitude for p in link.aggressor_ports})

    # Set to None to include all available pulse monitors. To use a subset, pass
    # a list such as ["master_tx_main_out", "slave_afe_rx_in", "slave_rx_post_ctle"].
    pulse_monitor_points: list[str] | None = None
    if gui_cfg.run_pulse_response:
        run_pulse_response_setup(link, monitor_points=pulse_monitor_points)

    num_cycles = gui_cfg.num_cycles
    master_tx_data: list[int] = []
    slave_rx_data: list[int] = []
    master_rx_input_data: list[float] = []
    slave_rx_input_data: list[float] = []
    master_tx_clk_edges: list[float] = []
    slave_rx_clk_edges: list[float] = []
    slave_pi_code: list[int] = []

    for i in range(num_cycles):
        link.run()

        if link.master_tx_pi.clk_out.is_edge:
            master_tx_clk_edges.append(i + link.master_tx_pi.clk_out.frac_dly)
            master_tx_data.append(link.master_tx.data)

        master_rx_input_data.append(link.master_afe.out_to_rx)

        if link.slave_rx_pi.clk_out.is_edge:
            slave_rx_clk_edges.append(i + link.slave_rx_pi.clk_out.frac_dly)
            slave_rx_data.append(link.slave_rx.data)
            slave_pi_code.append(link.slave_rx.pi_code)

        slave_rx_input_data.append(link.slave_afe.out_to_rx)

    print("Master TX samples:", master_tx_data[:10])
    print("Slave RX samples:", slave_rx_data[:10])
    print("Master RX input samples:", master_rx_input_data[:10])
    print("Slave RX input samples:", slave_rx_input_data[:10])

    if len(master_tx_clk_edges) > 1 and len(slave_rx_clk_edges) > 1:
        master_periods = np.diff(master_tx_clk_edges) / link.SAMP_FREQ_HZ
        slave_periods = np.diff(slave_rx_clk_edges) / link.SAMP_FREQ_HZ

        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        axs[0].plot(master_periods, "-o")
        axs[0].set_title("Master TX Clock Periods")
        axs[0].set_xlabel("Edge Index")
        axs[0].set_ylabel("Period (s)")
        axs[0].grid(True)

        axs[1].plot(slave_periods, "-o")
        axs[1].set_title("Slave RX Clock Periods")
        axs[1].set_xlabel("Edge Index")
        axs[1].set_ylabel("Period (s)")
        axs[1].grid(True)
        fig.tight_layout()

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(master_rx_input_data, "-")
    axs[0].set_title("Master RX Input")
    axs[0].grid(True)
    axs[1].plot(slave_rx_input_data, "-")
    axs[1].set_title("Slave RX Input")
    axs[1].grid(True)
    fig.tight_layout()

    plt.figure(figsize=(10, 3))
    plt.plot(slave_pi_code, "-")
    plt.title("Slave RX PI Code")
    plt.xlabel("Cycle")
    plt.grid(True)
    plt.tight_layout()

    diag = link.diagnostic(show=True)
    print("Master eye metrics:", diag["master_eye_metrics"])
    print("Slave eye metrics:", diag["slave_eye_metrics"])
    if link.aggressor_ports:
        aggr_rank = link.find_dominant_aggressors()
        print("Aggressor ranking (port, peak, NEXT_peak, FEXT_peak):", aggr_rank)
        aggr = aggr_rank[0][0]
        xtalk = link.get_aggressor_victim_pulse_response(aggr, include_total=True)
        print(
            f"Aggressor {aggr} pulse response samples (NEXT/FEXT):",
            xtalk["next_impulse"][:5],
            xtalk["fext_impulse"][:5],
        )
        link.plot_aggressor_victim_pulse_response(aggr, include_total=True, time_unit="ns")

    link.plot_chan_afe_impulses()
    plt.show()

    # Interactive Plotly eye (master + slave)
    _, metrics_m = link.master_rx.plot_eye_plotly(
        mask_type="diamond",
        mask_sigma=0.5,
        show=True,
    )
    _, metrics_s = link.slave_rx.plot_eye_plotly(
        mask_type="diamond",
        mask_sigma=0.5,
        show=True,
    )

    # Metrics only (master + slave)
    metrics_m = link.master_rx.get_eye_metrics()
    metrics_s = link.slave_rx.get_eye_metrics()
    print("Master eye metrics (direct):", metrics_m)
    print("Slave eye metrics (direct):", metrics_s)


if __name__ == "__main__":
    main()
