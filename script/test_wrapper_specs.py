from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import random
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required for specs wrapper. Install with: pip install pyyaml"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_py.link_model import Link, Pattern
from src_py.link_model.driver import Driver
from src_py.link_model.io_termination import (
    DriverResistanceConfig,
    IOTerminationModel,
    SideTerminationConfig,
)
from script import test_uni_feature as uni_feature_suite
from script import test_uni_link_py as uni_test_suite
from src_py_uni.uni_link_model import UniDirLink
from src_py_uni.uni_link_model.io_termination_uni import (
    RxTerminationConfig,
    TxTerminationConfig,
    UniIOTerminationModel,
)


def _as_dict(value: Any) -> dict[str, Any]:
    """Return value as a dictionary or an empty dictionary."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """Return value as a list or an empty list."""
    return value if isinstance(value, list) else []


def _as_bool(value: Any, default: bool = False) -> bool:
    """Convert value to boolean with optional default for None."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _resolve_path(path_value: Any) -> Path:
    """Resolve a possibly relative path against the project root."""
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _to_pattern(value: Any, default: Pattern = Pattern.PRBS7) -> Pattern:
    """Convert a YAML pattern value into the Pattern enum."""
    if value is None:
        return Pattern(default)
    if isinstance(value, Pattern):
        return Pattern(value)
    if isinstance(value, str):
        key = value.strip().upper()
        if key in Pattern.__members__:
            return Pattern.__members__[key]
    return Pattern(int(value))


def _to_pairs(value: Any) -> list[tuple[int, int]] | None:
    """Convert a list-like input into integer channel-port pairs."""
    if value is None:
        return None
    out: list[tuple[int, int]] = []
    for item in _as_list(value):
        pair = _as_list(item)
        if len(pair) != 2:
            raise ValueError(f"Invalid channel pair entry: {item}")
        out.append((int(pair[0]), int(pair[1])))
    return out


def _to_int_port_list(value: Any) -> list[int] | None:
    """Convert a list-like input into integer port IDs."""
    if value is None:
        return None
    return [int(x) for x in _as_list(value)]


def _dataclass_from_dict(dc_type, value: Any):
    """Build a dataclass instance from a dictionary, filtering unknown keys."""
    d = _as_dict(value)
    allowed = {f.name for f in fields(dc_type)}
    kwargs = {k: d[k] for k in d if k in allowed}
    return dc_type(**kwargs)


def _set_deterministic_clock(clock_gen: Any) -> None:
    # Keep deterministic, jitter-free behavior for repeatable runs.
    """Force deterministic jitter-free behavior on a clock generator."""
    if hasattr(clock_gen, "abs_jitter_std_sec"):
        clock_gen.abs_jitter_std_sec = 0.0
    if hasattr(clock_gen, "_abs_jitter_prev"):
        clock_gen._abs_jitter_prev = 0.0
    if hasattr(clock_gen, "_abs_jitter"):
        clock_gen._abs_jitter = 0.0
    if hasattr(clock_gen, "_period_jitter"):
        clock_gen._period_jitter = 0.0
    if hasattr(clock_gen, "nominal_period"):
        clock_gen._period = clock_gen.nominal_period
        clock_gen._timer = 0.0
        if hasattr(clock_gen, "_reset_cycle_timers"):
            clock_gen._reset_cycle_timers(carry_frac=0.0)
        else:
            clock_gen._clk_q_pos_edge_timer_val = clock_gen._period / 4.0
            clock_gen._clk_i_neg_edge_timer_val = clock_gen._period / 2.0
            clock_gen._clk_q_neg_edge_timer_val = clock_gen._period * 3.0 / 4.0
            clock_gen._clk_i_pos_edge_timer_val = clock_gen._period


def _load_spec(path: Path) -> dict[str, Any]:
    """Load and validate the YAML spec as a dictionary."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Spec file must parse to a mapping. Got: {type(raw)}")
    return raw


def _iter_cache_source_files() -> list[Path]:
    """Return source files that participate in the wrapper cache key."""
    roots = [
        ROOT / "src_py",
        ROOT / "src_py_uni",
        ROOT / "script",
    ]
    files: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            files.append(p.resolve())
    uniq = {str(p): p for p in files}
    return [uniq[k] for k in sorted(uniq.keys())]


def _build_run_cache_key(
    spec_path: Path,
    *,
    model_override: str | None,
    cycles_override: int | None,
    show_plots: bool,
) -> tuple[str, dict[str, Any]]:
    """Build deterministic cache key from spec content, run args, and source hashes."""
    spec_bytes = spec_path.read_bytes()
    spec_sha = hashlib.sha256(spec_bytes).hexdigest()

    src_hashes: dict[str, str] = {}
    h = hashlib.sha256()
    h.update(f"spec_sha256:{spec_sha}\n".encode("utf-8"))
    h.update(f"model_override:{model_override}\n".encode("utf-8"))
    h.update(f"cycles_override:{cycles_override}\n".encode("utf-8"))
    h.update(f"show_plots:{bool(show_plots)}\n".encode("utf-8"))
    for p in _iter_cache_source_files():
        rel = p.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        src_hashes[rel] = digest
        h.update(f"{rel}:{digest}\n".encode("utf-8"))

    key = h.hexdigest()[:24]
    fingerprint = {
        "spec_path": str(spec_path),
        "spec_sha256": spec_sha,
        "model_override": model_override,
        "cycles_override": cycles_override,
        "show_plots": bool(show_plots),
        "source_hashes": src_hashes,
    }
    return key, fingerprint


def _save_open_figures_to_cache(fig_dir: Path) -> list[str]:
    """Save currently open matplotlib figures to cache and return file names."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for i, fig_num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(fig_num)
        name = f"figure_{i:03d}.png"
        fig.savefig(fig_dir / name, dpi=150, bbox_inches="tight")
        files.append(name)
    return files


def _replay_cached_figures(fig_dir: Path) -> int:
    """Replay cached figure images as matplotlib windows."""
    if not fig_dir.exists():
        return 0
    files = sorted(fig_dir.glob("figure_*.png"))
    count = 0
    for p in files:
        try:
            img = plt.imread(str(p))
        except Exception:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(p.stem)
        fig.tight_layout()
        count += 1
    return count


def _apply_common_rx_to_uni(link: UniDirLink, common: dict[str, Any], uni_rx: dict[str, Any]) -> None:
    """Apply common RX settings from spec to a uni-directional link."""
    rx_common = _as_dict(common.get("rx_common"))
    rx_eq = _as_dict(common.get("rx_eq"))
    ctle = _as_dict(rx_eq.get("ctle"))
    dfe = _as_dict(rx_eq.get("dfe"))

    if "slicer_ref_v" in rx_common:
        link.rx.ref = float(rx_common["slicer_ref_v"])
    if "pd_out_gain" in rx_common:
        link.rx.pd_out_gain = float(rx_common["pd_out_gain"])
    if "slicer_sensitivity_v" in rx_common:
        link.rx.slicer_sensitivity = float(rx_common["slicer_sensitivity_v"])
    if "slicer_aperture_ui" in rx_common:
        link.rx.slicer_aperture_ui = float(rx_common["slicer_aperture_ui"])
    if "samples_per_ui" in rx_common:
        link.rx.samples_per_ui = int(rx_common["samples_per_ui"])
    if "eye_trace_span_ui" in rx_common:
        link.rx.eye_trace_span_ui = float(rx_common["eye_trace_span_ui"])
    if "sample_rate_hz" in rx_common:
        link.rx.sample_rate_hz = float(rx_common["sample_rate_hz"])

    if "enable" in ctle:
        link.rx.ctle_en = _as_bool(ctle.get("enable"), default=False)
    if "dc_gain_db" in ctle:
        link.rx.ctle_dc_gain_db = float(ctle["dc_gain_db"])
    if "peaking_gain_db" in ctle:
        link.rx.ctle_peaking_gain_db = float(ctle["peaking_gain_db"])
    if "peaking_freq_hz" in ctle:
        link.rx.ctle_peaking_freq_hz = None if ctle["peaking_freq_hz"] is None else float(ctle["peaking_freq_hz"])
    if "zero_freq_hz" in ctle:
        link.rx.ctle_zero_freq_hz = np.asarray(_as_list(ctle["zero_freq_hz"]), dtype=np.float64)
    if "pole_freq_hz" in ctle:
        link.rx.ctle_pole_freq_hz = np.asarray(_as_list(ctle["pole_freq_hz"]), dtype=np.float64)

    if "enable" in dfe:
        link.rx.dfe_en = _as_bool(dfe.get("enable"), default=False)
    if "taps" in dfe:
        link.rx.dfe_taps = np.asarray(_as_list(dfe["taps"]), dtype=np.float64)

    # Model-specific RX overrides take precedence.
    for key in (
        "ctle_en",
        "ctle_dc_gain_db",
        "ctle_peaking_gain_db",
        "ctle_peaking_freq_hz",
        "dfe_en",
        "slicer_sensitivity",
        "slicer_aperture_ui",
        "samples_per_ui",
        "eye_trace_span_ui",
        "sample_rate_hz",
    ):
        if key in uni_rx:
            setattr(link.rx, key, uni_rx[key])
    if "ctle_zero_freq_hz" in uni_rx:
        link.rx.ctle_zero_freq_hz = np.asarray(_as_list(uni_rx["ctle_zero_freq_hz"]), dtype=np.float64)
    if "ctle_pole_freq_hz" in uni_rx:
        link.rx.ctle_pole_freq_hz = np.asarray(_as_list(uni_rx["ctle_pole_freq_hz"]), dtype=np.float64)
    if "dfe_taps" in uni_rx:
        link.rx.dfe_taps = np.asarray(_as_list(uni_rx["dfe_taps"]), dtype=np.float64)


def _sync_uni_rate_timing(link: UniDirLink) -> None:
    # Keep UI timing tied to model clock/rate mode.
    """Align RX sample timing to the link rate-mode derived UI."""
    link.rx.sample_rate_hz = float(link.SAMP_FREQ_HZ)
    link.rx.samples_per_ui = int(getattr(link, "data_ui_samples", link.rx.samples_per_ui))


def _apply_uni_io_termination(link: UniDirLink, uni_cfg: dict[str, Any]) -> None:
    """Apply uni-directional IO termination settings from spec."""
    io_cfg = _as_dict(uni_cfg.get("io_termination"))
    if not io_cfg:
        return
    z0 = float(io_cfg.get("z0_ohm", link.Z0))
    tx_cfg = _dataclass_from_dict(TxTerminationConfig, io_cfg.get("tx_cfg"))
    rx_cfg = _dataclass_from_dict(RxTerminationConfig, io_cfg.get("rx_cfg"))
    link.io_term = UniIOTerminationModel(z0_ohm=z0, tx_cfg=tx_cfg, rx_cfg=rx_cfg)
    link.update_impulses()


def _apply_uni_aggressors(link: UniDirLink, common: dict[str, Any]) -> None:
    """Configure uni-directional aggressors from common spec settings."""
    aggr = _as_dict(common.get("aggressors"))
    if "ports" in aggr:
        link.set_aggressor_ports(_to_int_port_list(aggr.get("ports")))
    if "enable" in aggr:
        link.set_aggressor_enable(_as_bool(aggr.get("enable"), default=True))

    mode = str(aggr.get("source_mode", "manual")).strip().lower()
    link.set_aggressor_source_mode(mode)

    if mode == "manual":
        manual = _as_dict(aggr.get("manual_sources_v"))
        if manual:
            link.set_aggressor_sources({int(p): float(v) for p, v in manual.items()})
    else:
        pb = _as_dict(aggr.get("pattern_broadcast"))
        base_pattern = _to_pattern(pb.get("pattern"), default=Pattern.PRBS7)
        base_amp = float(pb.get("amplitude_v", Driver.AVDD))
        link.broadcast_aggressor_pattern(base_pattern, amplitude=base_amp)
        overrides = _as_dict(aggr.get("pattern_overrides"))
        for p_str, ov in overrides.items():
            ov_d = _as_dict(ov)
            link.set_aggressor_pattern(
                aggressor_port=int(p_str),
                pattern=_to_pattern(ov_d.get("pattern"), default=base_pattern),
                amplitude=float(ov_d.get("amplitude_v", base_amp)),
            )

    # Optional per-aggressor PI / phase config block.
    # This block is not required in specs.yaml, but supported by the wrapper.
    pi_cfg = _as_dict(aggr.get("pi_codes"))
    if pi_cfg:
        tx_b = pi_cfg.get("tx_broadcast")
        rx_b = pi_cfg.get("rx_broadcast")
        if tx_b is not None or rx_b is not None:
            link.broadcast_aggressor_pi_codes(
                tx_pi_code=None if tx_b is None else int(tx_b),
                rx_pi_code=None if rx_b is None else int(rx_b),
            )
        for p_str, d in _as_dict(pi_cfg.get("by_port")).items():
            dd = _as_dict(d)
            link.set_aggressor_pi_codes(
                aggressor_port=int(p_str),
                tx_pi_code=None if dd.get("tx_pi_code") is None else int(dd["tx_pi_code"]),
                rx_pi_code=None if dd.get("rx_pi_code") is None else int(dd["rx_pi_code"]),
            )

    phase_cfg = _as_dict(aggr.get("phase_offsets"))
    if phase_cfg:
        tx_b = phase_cfg.get("tx_broadcast")
        rx_b = phase_cfg.get("rx_broadcast")
        if tx_b is not None or rx_b is not None:
            link.broadcast_aggressor_phase_offsets(
                tx_phase_offset_code=0 if tx_b is None else int(tx_b),
                rx_phase_offset_code=0 if rx_b is None else int(rx_b),
            )
        for p_str, d in _as_dict(phase_cfg.get("by_port")).items():
            dd = _as_dict(d)
            link.set_aggressor_phase_offsets(
                aggressor_port=int(p_str),
                tx_phase_offset_code=int(dd.get("tx_phase_offset_code", 0)),
                rx_phase_offset_code=int(dd.get("rx_phase_offset_code", 0)),
            )


def _build_uni_link_from_spec(spec: dict[str, Any]) -> UniDirLink:
    """Construct a UniDirLink instance from the YAML spec."""
    common = _as_dict(spec.get("common"))
    uni_cfg = _as_dict(spec.get("uni_dir"))
    link_init = _as_dict(uni_cfg.get("link_init"))

    chan = _as_dict(common.get("channel"))
    victim = _as_dict(common.get("victim_ports"))
    clocks = _as_dict(common.get("clocks"))
    rx_common = _as_dict(common.get("rx_common"))

    chan_file = _resolve_path(chan.get("chan_file", "./data/A1_Combined_models_Slice_C_clocks_Slice_D_Data.s10p"))
    channel_pairs = _to_pairs(chan.get("channel_pairs"))

    tx_port = int(link_init.get("chan_port_tx_sel", victim.get("tx_port", 7)))
    rx_port = int(link_init.get("chan_port_rx_sel", victim.get("rx_port", 8)))
    tx_pi_code = int(link_init.get("tx_pi_code", clocks.get("tx_pi_code", 0)))
    rx_pi_code = int(link_init.get("rx_pi_code", clocks.get("rx_pi_code", 0)))
    txrx_rate_mode = str(link_init.get("txrx_rate_mode", clocks.get("txrx_rate_mode", "full")))
    txrx_clock_freq_hz = float(link_init.get("txrx_clock_freq_hz", clocks.get("txrx_clock_freq_hz", UniDirLink.CLK_FREQ_HZ)))
    clk_dcd_ui = float(link_init.get("clk_dcd_ui", clocks.get("clk_dcd_ui", 0.0)))
    clk_iq_mismatch_ui = float(link_init.get("clk_iq_mismatch_ui", clocks.get("clk_iq_mismatch_ui", 0.0)))
    rx_clk_ofst = float(link_init.get("rx_clk_ofst", clocks.get("rx_clk_offset_samples", 4.0)))
    rx_slicer_ref = float(link_init.get("rx_slicer_ref", rx_common.get("slicer_ref_v", 0.5 * Driver.AVDD)))
    rx_pd_out_gain = float(link_init.get("rx_pd_out_gain", rx_common.get("pd_out_gain", 0.0)))
    tx_ffe_raw = link_init.get("tx_ffe_taps", [0.0, 62.0, 0.0, 0.0])
    tx_ffe_taps = None if tx_ffe_raw is None else _as_list(tx_ffe_raw)

    link = UniDirLink(
        chan_file=chan_file,
        chan_port_tx_sel=tx_port,
        chan_port_rx_sel=rx_port,
        tx_pattern=_to_pattern(link_init.get("tx_pattern"), default=Pattern.PRBS7),
        tx_ffe_taps=tx_ffe_taps,
        tx_pi_code=tx_pi_code,
        rx_pi_code=rx_pi_code,
        txrx_rate_mode=txrx_rate_mode,
        txrx_clock_freq_hz=txrx_clock_freq_hz,
        clk_dcd_ui=clk_dcd_ui,
        clk_iq_mismatch_ui=clk_iq_mismatch_ui,
        rx_term_code=float(link_init.get("rx_term_code", 63.0)),
        rx_clk_ofst=rx_clk_ofst,
        rx_slicer_ref=rx_slicer_ref,
        rx_pd_out_gain=rx_pd_out_gain,
        channel_pairs=channel_pairs,
        aggressor_ports=_to_int_port_list(link_init.get("aggressor_ports")),
        aggressor_enable=_as_bool(link_init.get("aggressor_enable"), default=True),
    )

    _apply_common_rx_to_uni(link, common=common, uni_rx=_as_dict(uni_cfg.get("rx")))
    _sync_uni_rate_timing(link)
    _apply_uni_io_termination(link, uni_cfg=uni_cfg)
    _apply_uni_aggressors(link, common=common)
    return link


def _apply_controller_overrides(ctrl: Any, cfg: dict[str, Any], common: dict[str, Any]) -> None:
    """Apply controller settings and shared common overrides."""
    clocks = _as_dict(common.get("clocks"))
    rx_common = _as_dict(common.get("rx_common"))
    rx_eq = _as_dict(common.get("rx_eq"))
    ctle = _as_dict(rx_eq.get("ctle"))
    dfe = _as_dict(rx_eq.get("dfe"))

    if "tx_pi_code" in clocks:
        ctrl.tx_pi_code = int(clocks["tx_pi_code"])
    if "rx_clk_offset_samples" in clocks:
        ctrl.rx_clk_ofset = int(clocks["rx_clk_offset_samples"])
    if "slicer_ref_v" in rx_common:
        ctrl.rx_slc_ref = float(rx_common["slicer_ref_v"])
    if "pd_out_gain" in rx_common:
        ctrl.rx_pd_out_gain = float(rx_common["pd_out_gain"])
    if "slicer_sensitivity_v" in rx_common:
        ctrl.rx_slicer_sensitivity = float(rx_common["slicer_sensitivity_v"])
    if "slicer_aperture_ui" in rx_common:
        ctrl.rx_slicer_aperture_ui = float(rx_common["slicer_aperture_ui"])
    if "enable" in ctle:
        ctrl.rx_ctle_en = _as_bool(ctle["enable"], default=False)
    if "dc_gain_db" in ctle:
        ctrl.rx_ctle_dc_gain_db = float(ctle["dc_gain_db"])
    if "peaking_gain_db" in ctle:
        ctrl.rx_ctle_peaking_gain_db = float(ctle["peaking_gain_db"])
    if "peaking_freq_hz" in ctle:
        ctrl.rx_ctle_peaking_freq_hz = None if ctle["peaking_freq_hz"] is None else float(ctle["peaking_freq_hz"])
    if "zero_freq_hz" in ctle:
        ctrl.rx_ctle_zero_freq_hz = np.asarray(_as_list(ctle["zero_freq_hz"]), dtype=np.float64)
    if "pole_freq_hz" in ctle:
        ctrl.rx_ctle_pole_freq_hz = np.asarray(_as_list(ctle["pole_freq_hz"]), dtype=np.float64)
    if "enable" in dfe:
        ctrl.rx_dfe_en = _as_bool(dfe["enable"], default=False)
    if "taps" in dfe:
        ctrl.rx_dfe_taps = np.asarray(_as_list(dfe["taps"]), dtype=np.float64)

    # Side-specific overrides.
    if "tx_data_gen_pattern" in cfg:
        ctrl.tx_data_gen_pattern = _to_pattern(cfg["tx_data_gen_pattern"], default=ctrl.tx_data_gen_pattern)
    if "tx_main_drv_codes" in cfg:
        ctrl.tx_main_drv_codes = np.asarray(_as_list(cfg["tx_main_drv_codes"]), dtype=np.float64)
    if "tx_main_drv_inv_pol" in cfg:
        ctrl.tx_main_drv_inv_pol = np.asarray(_as_list(cfg["tx_main_drv_inv_pol"]), dtype=bool)
    if "tx_main_drv_en" in cfg:
        ctrl.tx_main_drv_en = _as_bool(cfg["tx_main_drv_en"], default=ctrl.tx_main_drv_en)
    if "tx_echo_drv_codes" in cfg:
        ctrl.tx_echo_drv_codes = np.asarray(_as_list(cfg["tx_echo_drv_codes"]), dtype=np.float64)
    if "tx_echo_drv_inv_pol" in cfg:
        ctrl.tx_echo_drv_inv_pol = np.asarray(_as_list(cfg["tx_echo_drv_inv_pol"]), dtype=bool)
    if "tx_echo_drv_en" in cfg:
        ctrl.tx_echo_drv_en = _as_bool(cfg["tx_echo_drv_en"], default=ctrl.tx_echo_drv_en)
    if "tx_pi_code" in cfg:
        ctrl.tx_pi_code = int(cfg["tx_pi_code"])
    if "rx_clk_ofset" in cfg:
        ctrl.rx_clk_ofset = int(cfg["rx_clk_ofset"])
    if "rx_slc_ref" in cfg:
        ctrl.rx_slc_ref = float(cfg["rx_slc_ref"])
    if "rx_pd_out_gain" in cfg:
        ctrl.rx_pd_out_gain = float(cfg["rx_pd_out_gain"])
    if "rx_ctle_en" in cfg:
        ctrl.rx_ctle_en = _as_bool(cfg["rx_ctle_en"], default=ctrl.rx_ctle_en)
    if "rx_ctle_dc_gain_db" in cfg:
        ctrl.rx_ctle_dc_gain_db = float(cfg["rx_ctle_dc_gain_db"])
    if "rx_ctle_peaking_gain_db" in cfg:
        ctrl.rx_ctle_peaking_gain_db = float(cfg["rx_ctle_peaking_gain_db"])
    if "rx_ctle_peaking_freq_hz" in cfg:
        ctrl.rx_ctle_peaking_freq_hz = None if cfg["rx_ctle_peaking_freq_hz"] is None else float(cfg["rx_ctle_peaking_freq_hz"])
    if "rx_ctle_zero_freq_hz" in cfg:
        ctrl.rx_ctle_zero_freq_hz = np.asarray(_as_list(cfg["rx_ctle_zero_freq_hz"]), dtype=np.float64)
    if "rx_ctle_pole_freq_hz" in cfg:
        ctrl.rx_ctle_pole_freq_hz = np.asarray(_as_list(cfg["rx_ctle_pole_freq_hz"]), dtype=np.float64)
    if "rx_dfe_en" in cfg:
        ctrl.rx_dfe_en = _as_bool(cfg["rx_dfe_en"], default=ctrl.rx_dfe_en)
    if "rx_dfe_taps" in cfg:
        ctrl.rx_dfe_taps = np.asarray(_as_list(cfg["rx_dfe_taps"]), dtype=np.float64)
    if "rx_slicer_sensitivity" in cfg:
        ctrl.rx_slicer_sensitivity = float(cfg["rx_slicer_sensitivity"])
    if "rx_slicer_aperture_ui" in cfg:
        ctrl.rx_slicer_aperture_ui = float(cfg["rx_slicer_aperture_ui"])


def _apply_dual_io_termination(link: Link, dual_cfg: dict[str, Any]) -> None:
    """Apply dual-direction IO termination settings from spec."""
    io_cfg = _as_dict(dual_cfg.get("io_termination"))
    if not io_cfg:
        return
    z0 = float(io_cfg.get("z0_ohm", link.Z0))
    driver_cfg = _dataclass_from_dict(DriverResistanceConfig, io_cfg.get("driver_cfg"))
    master_cfg = _dataclass_from_dict(SideTerminationConfig, io_cfg.get("master_cfg"))
    slave_cfg = _dataclass_from_dict(SideTerminationConfig, io_cfg.get("slave_cfg"))
    link.io_termination = IOTerminationModel(
        z0_ohm=z0,
        driver_cfg=driver_cfg,
        master_cfg=master_cfg,
        slave_cfg=slave_cfg,
    )
    link.update_chan_afe_impulses()


def _build_dual_link_from_spec(spec: dict[str, Any]) -> Link:
    """Construct a dual-direction Link instance from spec."""
    common = _as_dict(spec.get("common"))
    dual_cfg = _as_dict(spec.get("dual_dir"))
    link_init = _as_dict(dual_cfg.get("link_init"))

    chan = _as_dict(common.get("channel"))
    victim = _as_dict(common.get("victim_ports"))

    chan_file = _resolve_path(chan.get("chan_file", "./data/A1_Combined_models_Slice_C_clocks_Slice_D_Data.s10p"))
    channel_pairs = _to_pairs(chan.get("channel_pairs"))

    link = Link(
        chan_file=chan_file,
        chan_port_one_sel=int(link_init.get("chan_port_one_sel", victim.get("tx_port", 7))),
        chan_port_two_sel=int(link_init.get("chan_port_two_sel", victim.get("rx_port", 8))),
        channel_pairs=channel_pairs,
        aggressor_ports=_to_int_port_list(link_init.get("aggressor_ports")),
    )

    _apply_dual_io_termination(link, dual_cfg=dual_cfg)
    _apply_controller_overrides(link.master_ctrl, _as_dict(dual_cfg.get("master_controller")), common=common)
    _apply_controller_overrides(link.slave_ctrl, _as_dict(dual_cfg.get("slave_controller")), common=common)

    # Optional shared RX PI initialization from common clock section.
    clocks = _as_dict(common.get("clocks"))
    if "rx_pi_code" in clocks:
        pi_code = int(clocks["rx_pi_code"])
        link.master_rx.pi_code = pi_code
        link.slave_rx.pi_code = pi_code
    return link


def _pulse_metric_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    """Serialize pulse metric objects into plain dictionaries."""
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if hasattr(v, "__dict__"):
            out[str(k)] = dict(vars(v))
        else:
            out[str(k)] = v
    return out


def _compute_pi_tracking_stats(
    pi_codes: np.ndarray,
    tail_edges: int = 1024,
) -> dict[str, Any]:
    """Compute PI tracking spread metrics, emphasizing a tail window."""
    pi = np.asarray(pi_codes, dtype=np.int16).reshape(-1)
    if pi.size == 0:
        return {
            "n_edges": 0,
            "lock_code": None,
            "tail_edges": 0,
            "delta_std_tail": None,
            "delta_pp_tail": None,
            "n_unique_tail": 0,
            "top_codes_tail": [],
        }

    hist_all = np.bincount(pi, minlength=128)
    lock_code = int(np.argmax(hist_all))
    pi_delta = ((pi - lock_code + 64) % 128) - 64

    tail_n = int(max(1, min(int(tail_edges), pi_delta.size)))
    tail_delta = np.asarray(pi_delta[-tail_n:], dtype=np.int16)
    tail_codes = np.asarray(pi[-tail_n:], dtype=np.int16)
    hist_tail = np.bincount(tail_codes, minlength=128)
    top_codes = np.argsort(hist_tail)[-5:][::-1]
    return {
        "n_edges": int(pi.size),
        "lock_code": lock_code,
        "tail_edges": int(tail_n),
        "delta_std_tail": float(np.std(tail_delta)),
        "delta_pp_tail": float(np.max(tail_delta) - np.min(tail_delta)),
        "n_unique_tail": int(np.count_nonzero(hist_tail)),
        "top_codes_tail": [(int(c), int(hist_tail[c])) for c in top_codes if hist_tail[c] > 0],
    }


def _is_pi_tracking_stable(
    stats: dict[str, Any],
    *,
    max_delta_pp: float,
    max_delta_std: float,
    max_unique_codes: int,
) -> bool:
    """Return True when PI spread is within configured lock thresholds."""
    if int(stats.get("n_edges", 0)) <= 0:
        return False
    dpp = stats.get("delta_pp_tail")
    dstd = stats.get("delta_std_tail")
    nu = int(stats.get("n_unique_tail", 0))
    if dpp is None or dstd is None:
        return False
    return bool(float(dpp) <= float(max_delta_pp) and float(dstd) <= float(max_delta_std) and nu <= int(max_unique_codes))


def _run_eye_case_with_stable_cdr(
    eye_link: UniDirLink,
    *,
    eye_cycles: int,
    with_aggressors: bool,
    drive_mode: str,
    aggr_amp: float,
    aggr_pattern_broadcast: Pattern,
    aggr_pattern_overrides: dict[int, Pattern],
    aggr_seed: int,
    settle_cycles: int,
    settle_chunk_cycles: int,
    settle_max_extra_cycles: int,
    tail_edges: int,
    max_delta_pp: float,
    max_delta_std: float,
    max_unique_codes: int,
    require_stable: bool,
) -> dict[str, Any]:
    """Run one eye case, settling CDR first so eye traces are captured after lock."""
    cdr_active = float(getattr(eye_link.rx, "pd_out_gain", 0.0)) != 0.0
    settle_total = 0
    settle_attempts = 0
    stable_before_capture = True
    settle_stats: dict[str, Any] = {
        "cdr_active": bool(cdr_active),
        "settle_cycles_total": 0,
        "settle_attempts": 0,
        "stable_before_capture": True,
    }

    if cdr_active:
        stable_before_capture = False
        settle_budget = int(max(1, settle_cycles))
        max_total = int(max(settle_budget, settle_budget + max(0, int(settle_max_extra_cycles))))
        chunk = int(max(1, settle_chunk_cycles))
        latest_stats: dict[str, Any] = {}
        while settle_total < max_total:
            n = int(min(chunk if settle_attempts > 0 else settle_budget, max_total - settle_total))
            if n <= 0:
                break
            pr_settle = uni_test_suite.run_prbs(
                eye_link,
                num_cycles=n,
                with_aggressors=with_aggressors,
                aggressor_drive_mode=drive_mode,
                aggressor_amplitude=aggr_amp,
                aggressor_pattern_broadcast=aggr_pattern_broadcast,
                aggressor_pattern_overrides=aggr_pattern_overrides,
                aggressor_seed=aggr_seed + settle_attempts + 1,
            )
            settle_total += n
            settle_attempts += 1
            latest_stats = _compute_pi_tracking_stats(pr_settle["rx_pi_code"], tail_edges=tail_edges)
            stable_before_capture = _is_pi_tracking_stable(
                latest_stats,
                max_delta_pp=max_delta_pp,
                max_delta_std=max_delta_std,
                max_unique_codes=max_unique_codes,
            )
            if stable_before_capture:
                break
        settle_stats.update(latest_stats)
        settle_stats["settle_cycles_total"] = int(settle_total)
        settle_stats["settle_attempts"] = int(settle_attempts)
        settle_stats["stable_before_capture"] = bool(stable_before_capture)
        if require_stable and not stable_before_capture:
            raise AssertionError(
                "CDR PI did not stabilize before eye capture: "
                f"delta_pp_tail={settle_stats.get('delta_pp_tail')}, "
                f"delta_std_tail={settle_stats.get('delta_std_tail')}, "
                f"n_unique_tail={settle_stats.get('n_unique_tail')}"
            )

    # Ensure plotted eye uses only post-settling traces.
    if hasattr(eye_link.rx, "reset_eye_monitor"):
        eye_link.rx.reset_eye_monitor()

    pr_eye = uni_test_suite.run_prbs(
        eye_link,
        num_cycles=int(max(1, eye_cycles)),
        with_aggressors=with_aggressors,
        aggressor_drive_mode=drive_mode,
        aggressor_amplitude=aggr_amp,
        aggressor_pattern_broadcast=aggr_pattern_broadcast,
        aggressor_pattern_overrides=aggr_pattern_overrides,
        aggressor_seed=aggr_seed + 1000003,
    )
    capture_stats = _compute_pi_tracking_stats(pr_eye["rx_pi_code"], tail_edges=tail_edges)
    capture_stable = _is_pi_tracking_stable(
        capture_stats,
        max_delta_pp=max_delta_pp,
        max_delta_std=max_delta_std,
        max_unique_codes=max_unique_codes,
    ) if cdr_active else True
    if require_stable and cdr_active and (not capture_stable):
        raise AssertionError(
            "CDR PI was not stable during eye capture: "
            f"delta_pp_tail={capture_stats.get('delta_pp_tail')}, "
            f"delta_std_tail={capture_stats.get('delta_std_tail')}, "
            f"n_unique_tail={capture_stats.get('n_unique_tail')}"
        )
    return {
        "settle": settle_stats,
        "capture": {
            **capture_stats,
            "stable_during_capture": bool(capture_stable),
            "eye_cycles": int(max(1, eye_cycles)),
        },
    }


def _run_uni_basic_test_case_plots(spec: dict[str, Any], show_plots: bool) -> dict[str, Any]:
    """Run wrapper-native uni-directional DC/step and CDR plot cases."""
    sim = _as_dict(spec.get("simulation"))
    num_cfg = _as_dict(sim.get("num_cycles"))
    deterministic_clock = _as_bool(sim.get("deterministic_clock"), default=False)
    pre = int(num_cfg.get("step_pre", 512))
    post = int(num_cfg.get("step_post", 1024))
    cdr_cycles = int(num_cfg.get("cdr_pi_run", 50000))

    def _build_case_link() -> UniDirLink:
        """Build a dedicated uni-directional link for one wrapper test case."""
        case_link = _build_uni_link_from_spec(spec)
        case_link.set_aggressor_enable(False)
        case_link.set_aggressor_source_mode("manual")
        if case_link.aggressor_ports:
            case_link.set_aggressor_sources({int(p): 0.0 for p in case_link.aggressor_ports})
        if deterministic_clock:
            _set_deterministic_clock(case_link.clk_src)
        return case_link

    def _run_step_case(p0: Pattern, p1: Pattern) -> tuple[np.ndarray, np.ndarray]:
        """Run one step-pattern transition and capture TX/RX traces."""
        case_link = _build_case_link()
        n = int(pre + post)
        tx = np.zeros(n, dtype=np.float64)
        rx = np.zeros(n, dtype=np.float64)
        for i in range(n):
            case_link.tx_pattern = Pattern(p0) if i < pre else Pattern(p1)
            case_link.run()
            tx[i] = float(case_link.tx_drv_out)
            rx[i] = float(case_link.rx_in)
        return tx, rx

    tx_up, rx_up = _run_step_case(Pattern.ALL_ZEROS, Pattern.ALL_ONES)
    tx_dn, rx_dn = _run_step_case(Pattern.ALL_ONES, Pattern.ALL_ZEROS)
    w = max(1, min(128, pre, post))
    dc_step_metrics = {
        "up_tx_delta": float(np.mean(tx_up[-w:]) - np.mean(tx_up[:w])),
        "up_rx_delta": float(np.mean(rx_up[-w:]) - np.mean(rx_up[:w])),
        "dn_tx_delta": float(np.mean(tx_dn[-w:]) - np.mean(tx_dn[:w])),
        "dn_rx_delta": float(np.mean(rx_dn[-w:]) - np.mean(rx_dn[:w])),
        "pre_samples": int(pre),
        "post_samples": int(post),
    }

    if show_plots:
        x = np.arange(int(pre + post), dtype=np.float64)
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(x, tx_up, label="TX 0->1")
        axs[0].plot(x, tx_dn, label="TX 1->0")
        axs[0].set_title("DC/STEP Check at TX")
        axs[0].set_ylabel("Amplitude")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        axs[1].plot(x, rx_up, label="RX 0->1")
        axs[1].plot(x, rx_dn, label="RX 1->0")
        axs[1].set_title("DC/STEP Check at RX")
        axs[1].set_xlabel("Sample")
        axs[1].set_ylabel("Amplitude")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        fig.tight_layout()

    cdr_link = _build_case_link()
    cdr_link.tx_pattern = Pattern.PRBS13
    if float(cdr_link.rx.pd_out_gain) == 0.0:
        cdr_link.rx.pd_out_gain = 0.125
    cdr_link.rx.ref = float(0.2 * Driver.AVDD)

    pi_codes: list[int] = []
    pd_out: list[int] = []
    for _ in range(max(1, cdr_cycles)):
        cdr_link.run()
        if cdr_link.rx.clk.is_edge:
            pi_codes.append(int(cdr_link.rx.pi_code))
            pd_out.append(int(cdr_link.rx._pd_out))
    pi = np.asarray(pi_codes, dtype=np.int16)
    pd = np.asarray(pd_out, dtype=np.int8)

    cdr_metrics = {
        "n_edges": int(pi.size),
        "pi_min": int(np.min(pi)) if pi.size else None,
        "pi_max": int(np.max(pi)) if pi.size else None,
        "pi_std": float(np.std(pi)) if pi.size else None,
        "pd_counts": {
            "-1": int(np.sum(pd == -1)),
            "0": int(np.sum(pd == 0)),
            "+1": int(np.sum(pd == 1)),
        },
        "pd_out_gain": float(cdr_link.rx.pd_out_gain),
    }

    if show_plots and pi.size > 0:
        x = np.arange(pi.size, dtype=np.float64)
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(x, pi, linewidth=1.0)
        axs[0].set_title("CDR PI Code Tracking")
        axs[0].set_ylabel("PI Code")
        axs[0].grid(True, alpha=0.3)
        axs[1].plot(x, np.cumsum(pd), linewidth=1.0)
        axs[1].set_title("CDR PD Output Cumulative Sum")
        axs[1].set_xlabel("RX Sampling Edge Index")
        axs[1].set_ylabel("Cum PD")
        axs[1].grid(True, alpha=0.3)
        fig.tight_layout()

    return {
        "dc_step": dc_step_metrics,
        "cdr_pi_tracking": cdr_metrics,
    }


def _run_uni_feature_suite(show_plots: bool) -> dict[str, Any]:
    """Execute the test_uni_feature suite and collect structured results."""
    results: dict[str, Any] = {}
    failures: list[str] = []
    sanity_errors: list[str] = []

    def _run_case(name: str, fn) -> Any:
        """Run one feature-case function and capture pass/fail metadata."""
        try:
            val = fn()
            results[name] = {"status": "pass", "result": val}
            return val
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            results[name] = {"status": "fail", "error": str(exc)}
            return None

    _run_case("test_1_dc_levels", uni_feature_suite.test_1_dc_levels)
    _run_case("test_2_step_patterns", uni_feature_suite.test_2_step_patterns)
    _run_case("test_3_cdr_loop_pi_tracking", uni_feature_suite.test_3_cdr_loop_pi_tracking)
    _run_case("test_4_eye_prbs13_1024", uni_feature_suite.test_4_eye_prbs13_1024)
    _run_case("test_5_eye_with_without_xtalk", uni_feature_suite.test_5_eye_with_without_xtalk)

    out6 = _run_case("test_6_1ui_pulse_chain", uni_feature_suite.test_6_1ui_pulse_chain)
    if out6 is not None:
        metrics6, errs6 = out6
        results["test_6_1ui_pulse_chain"]["result"] = {
            "metrics": _pulse_metric_dict(metrics6),
            "errors": list(errs6),
        }
        sanity_errors.extend([str(e) for e in errs6])

    out7 = _run_case("test_7_aggressor_1ui_pulse_to_victim_rx", uni_feature_suite.test_7_aggressor_1ui_pulse_to_victim_rx)
    if out7 is not None:
        metrics7, errs7 = out7
        results["test_7_aggressor_1ui_pulse_to_victim_rx"]["result"] = {
            "metrics": _pulse_metric_dict(metrics7),
            "errors": list(errs7),
        }
        sanity_errors.extend([str(e) for e in errs7])

    if not show_plots:
        plt.close("all")

    return {
        "results": results,
        "failures": failures,
        "sanity_errors": sanity_errors,
        "passed": (len(failures) == 0 and len(sanity_errors) == 0),
    }


def _run_uni_sim(
    link: UniDirLink,
    spec: dict[str, Any],
    cycles_override: int | None = None,
    show_plots: bool = True,
) -> dict[str, Any]:
    """Run the uni-directional model flow from spec and summarize diagnostics."""
    common = _as_dict(spec.get("common"))
    sim = _as_dict(spec.get("simulation"))
    num_cfg = _as_dict(sim.get("num_cycles"))
    diag = _as_dict(common.get("diagnostics"))
    eye_plot = _as_dict(common.get("eye_plot"))
    aggr = _as_dict(common.get("aggressors"))
    run_feature_suite = _as_bool(diag.get("run_uni_feature_suite"), default=show_plots)
    run_basic_case_plots = _as_bool(
        diag.get("run_basic_test_case_plots"),
        default=(show_plots and not run_feature_suite),
    )

    seed = sim.get("random_seed")
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))
    if _as_bool(sim.get("deterministic_clock"), default=False):
        _set_deterministic_clock(link.clk_src)

    if show_plots and _as_bool(diag.get("plot_path_impulses"), default=True):
        link.plot_path_impulses(time_unit=str(diag.get("path_impulse_time_unit", "ns")))

    if show_plots and _as_bool(diag.get("plot_specific_aggressor_to_victim"), default=False):
        aggr_port = int(diag.get("specific_aggressor_port", 1))
        vic = int(diag.get("specific_victim_port", int(link.chan_port_rx_sel)))
        req = diag.get("required_coupling")
        req_c = None if req is None else str(req)
        link.plot_aggressor_to_victim_port_pulse_response(
            aggressor_port=aggr_port,
            victim_port=vic,
            require_coupling=req_c,
            time_unit=str(diag.get("pulse_time_unit", "ns")),
        )

    # Pull aggressor drive setup from spec for PRBS runs (same behavior as test_uni_link_py).
    drive_mode_raw = str(aggr.get("source_mode", "pattern")).strip().lower()
    drive_mode = "pattern" if drive_mode_raw == "pattern" else "manual_random"
    aggr_amp = float(_as_dict(aggr.get("pattern_broadcast")).get("amplitude_v", Driver.AVDD))
    aggr_pattern_broadcast = _to_pattern(_as_dict(aggr.get("pattern_broadcast")).get("pattern"), default=Pattern.PRBS7)
    aggr_pattern_overrides: dict[int, Pattern] = {
        int(p): _to_pattern(_as_dict(v).get("pattern"), default=aggr_pattern_broadcast)
        for p, v in _as_dict(aggr.get("pattern_overrides")).items()
    }
    aggr_seed = int(seed if seed is not None else 20260304)

    prbs_cycles_default = int(num_cfg.get("prbs_run", 20000))
    prbs_cycles = int(cycles_override) if cycles_override is not None else prbs_cycles_default
    eye_cycles = int(num_cfg.get("eye_run", prbs_cycles))
    main_trace_with_aggressors = _as_bool(aggr.get("enable"), default=True)

    pulse_res: dict[str, Any] | None = None
    if show_plots:
        pulse_res = uni_test_suite.run_1ui_pulse_response(
            link,
            monitor_keys=[
                "tx_drv_out",
                "tx_to_chan",
                "rx_xtalk_bump",
                "rx_bump",
                "rx_xtalk_in",
                "rx_in_pre_ctle",
                "rx_post_ctle",
                "rx_post_aperture",
            ],
        )
    basic_case_plots: dict[str, Any] | None = None
    if run_basic_case_plots:
        basic_case_plots = _run_uni_basic_test_case_plots(spec=spec, show_plots=show_plots)

    prbs = uni_test_suite.run_prbs(
        link,
        num_cycles=prbs_cycles,
        with_aggressors=bool(main_trace_with_aggressors),
        aggressor_drive_mode=drive_mode,
        aggressor_amplitude=aggr_amp,
        aggressor_pattern_broadcast=aggr_pattern_broadcast,
        aggressor_pattern_overrides=aggr_pattern_overrides,
        aggressor_seed=aggr_seed,
    )

    pi_stats: dict[str, Any] = {}
    if prbs["rx_pi_code"].size > 0:
        pi_codes = np.asarray(prbs["rx_pi_code"], dtype=np.int16)
        pi_hist = np.bincount(pi_codes, minlength=128)
        lock_code = int(np.argmax(pi_hist))
        pi_delta = ((pi_codes - lock_code + 64) % 128) - 64
        top_codes = np.argsort(pi_hist)[-5:][::-1]
        pi_stats = {
            "lock_code": lock_code,
            "delta_std": float(np.std(pi_delta)),
            "delta_pp": float(np.max(pi_delta) - np.min(pi_delta)),
            "n_unique_codes": int(np.count_nonzero(pi_hist)),
            "top_codes": [(int(c), int(pi_hist[c])) for c in top_codes if pi_hist[c] > 0],
        }
    pd_counts = {
        "-1": int(np.sum(prbs["rx_pd_out"] == -1)),
        "0": int(np.sum(prbs["rx_pd_out"] == 0)),
        "+1": int(np.sum(prbs["rx_pd_out"] == 1)),
    }

    if show_plots:
        fig, axs = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
        axs[0].plot(prbs["rx_in"], linewidth=0.8)
        axs[0].set_title("RX Input (pre-CTLE)")
        axs[0].set_ylabel("Amp")
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(prbs["rx_post_ctle"], linewidth=0.8)
        axs[1].set_title("RX Post-CTLE")
        axs[1].set_ylabel("Amp")
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(prbs["rx_post_aperture"], linewidth=0.8)
        axs[2].set_title("RX Post-Aperture")
        axs[2].set_ylabel("Amp")
        axs[2].grid(True, alpha=0.3)

        axs[3].plot(prbs["rx_xtalk_in"], linewidth=0.8)
        axs[3].set_title("RX Crosstalk Component (post RX termination)")
        axs[3].set_xlabel("Sample")
        axs[3].set_ylabel("Amp")
        axs[3].grid(True, alpha=0.3)
        fig.tight_layout()

    # Eye comparison (with and without aggressors), gated by CDR PI stability.
    eye_cdr_cfg = _as_dict(diag.get("eye_cdr_stability"))
    eye_tail_edges = int(eye_cdr_cfg.get("tail_edges", 1024))
    eye_settle_cycles = int(eye_cdr_cfg.get("settle_cycles", max(2000, int(eye_cycles // 4))))
    eye_settle_chunk_cycles = int(eye_cdr_cfg.get("settle_chunk_cycles", max(512, int(eye_settle_cycles // 2))))
    eye_settle_max_extra = int(eye_cdr_cfg.get("settle_max_extra_cycles", max(0, eye_settle_cycles)))
    eye_max_delta_pp = float(eye_cdr_cfg.get("max_delta_pp", 4.0))
    eye_max_delta_std = float(eye_cdr_cfg.get("max_delta_std", 1.0))
    eye_max_unique = int(eye_cdr_cfg.get("max_unique_codes", 8))
    eye_require_stable = _as_bool(eye_cdr_cfg.get("require_stable"), default=True)

    eye_cases: list[tuple[str, UniDirLink]] = []
    eye_metrics_by_case: dict[str, dict[str, Any]] = {}
    eye_cdr_by_case: dict[str, dict[str, Any]] = {}
    if _as_bool(diag.get("plot_eye_without_aggressors"), default=True):
        link_wo = _build_uni_link_from_spec(spec)
        if _as_bool(sim.get("deterministic_clock"), default=False):
            _set_deterministic_clock(link_wo.clk_src)
        eye_cdr_by_case["without_aggressors"] = _run_eye_case_with_stable_cdr(
            link_wo,
            eye_cycles=eye_cycles,
            with_aggressors=False,
            drive_mode=drive_mode,
            aggr_amp=aggr_amp,
            aggr_pattern_broadcast=aggr_pattern_broadcast,
            aggr_pattern_overrides=aggr_pattern_overrides,
            aggr_seed=aggr_seed,
            settle_cycles=eye_settle_cycles,
            settle_chunk_cycles=eye_settle_chunk_cycles,
            settle_max_extra_cycles=eye_settle_max_extra,
            tail_edges=eye_tail_edges,
            max_delta_pp=eye_max_delta_pp,
            max_delta_std=eye_max_delta_std,
            max_unique_codes=eye_max_unique,
            require_stable=eye_require_stable,
        )
        eye_cases.append(("Without Aggressors", link_wo))
        eye_metrics_by_case["without_aggressors"] = link_wo.rx.get_eye_metrics()

    if _as_bool(diag.get("plot_eye_with_aggressors"), default=True):
        link_w = _build_uni_link_from_spec(spec)
        if _as_bool(sim.get("deterministic_clock"), default=False):
            _set_deterministic_clock(link_w.clk_src)
        eye_cdr_by_case["with_aggressors"] = _run_eye_case_with_stable_cdr(
            link_w,
            eye_cycles=eye_cycles,
            with_aggressors=True,
            drive_mode=drive_mode,
            aggr_amp=aggr_amp,
            aggr_pattern_broadcast=aggr_pattern_broadcast,
            aggr_pattern_overrides=aggr_pattern_overrides,
            aggr_seed=aggr_seed,
            settle_cycles=eye_settle_cycles,
            settle_chunk_cycles=eye_settle_chunk_cycles,
            settle_max_extra_cycles=eye_settle_max_extra,
            tail_edges=eye_tail_edges,
            max_delta_pp=eye_max_delta_pp,
            max_delta_std=eye_max_delta_std,
            max_unique_codes=eye_max_unique,
            require_stable=eye_require_stable,
        )
        eye_cases.append(("With Aggressors", link_w))
        eye_metrics_by_case["with_aggressors"] = link_w.rx.get_eye_metrics()

    x_unit = str(eye_plot.get("x_unit", "ui"))
    mask_sigma = float(eye_plot.get("mask_sigma", 1.0))
    mask_type = eye_plot.get("mask_type")
    if isinstance(mask_type, str):
        m = mask_type.strip().lower()
        if m in {"", "none", "null"}:
            mask_type = None

    if show_plots and len(eye_cases) > 0:
        fig_eye, ax_eye = plt.subplots(1, len(eye_cases), figsize=(7.2 * len(eye_cases), 4), squeeze=False)
        for i, (label, eye_link) in enumerate(eye_cases):
            _, eye_metrics = eye_link.rx.plot_eye(
                ax=ax_eye[0, i],
                mask_type=mask_type,
                mask_sigma=mask_sigma,
                x_unit=x_unit,
                return_metrics=True,
            )
            xc = float(eye_metrics.get("x_center_in_unit", eye_metrics.get("x_center", float("nan"))))
            ax_eye[0, i].set_title(f"{label} (x_center={xc:.3f} {x_unit})")
        fig_eye.tight_layout()

    summary = {
        "active_model": "uni_dir",
        "cycles_run": prbs_cycles,
        "clocking": {
            "txrx_rate_mode": str(getattr(link, "txrx_rate_mode", "full")),
            "clock_period_ui": float(getattr(link, "txrx_period_ui_scale", 1.0)),
            "txrx_clock_freq_hz": float(getattr(link, "txrx_clock_freq_hz", 0.0)),
            "data_rate_hz": float(getattr(link, "data_rate_hz", 0.0)),
            "data_ui_samples": int(getattr(link, "data_ui_samples", 1)),
            "clk_dcd_ui": float(getattr(link, "clk_dcd_ui", 0.0)),
            "clk_iq_mismatch_ui": float(getattr(link, "clk_iq_mismatch_ui", 0.0)),
        },
        "victim_ports": {
            "tx": int(link.chan_port_tx_sel),
            "rx": int(link.chan_port_rx_sel),
        },
        "aggressor_ports": [int(p) for p in link.aggressor_ports],
        "aggressor_drive": {
            "mode": drive_mode,
            "amplitude_v": float(aggr_amp),
            "pattern_broadcast": aggr_pattern_broadcast.name,
            "pattern_overrides": {int(k): v.name for k, v in aggr_pattern_overrides.items()},
        },
        "pulse_response": None if pulse_res is None else {
            "monitor_keys": pulse_res["monitor_keys"],
            "ui_samples": int(pulse_res["ui_samples"]),
            "pulse_start_sample": int(pulse_res["pulse_start_sample"]),
            "pulse_end_sample": int(pulse_res["pulse_end_sample"]),
            "baseline_by_monitor": {
                str(k): float(v) for k, v in _as_dict(pulse_res.get("baseline_by_monitor")).items()
            },
            "cursor_magnitudes_by_monitor": {
                str(k): {str(cn): float(cv) for cn, cv in _as_dict(cdict).items()}
                for k, cdict in _as_dict(pulse_res.get("cursor_magnitudes_by_monitor")).items()
            },
        },
        "basic_test_case_plots": basic_case_plots,
        "prbs": {
            "rx_in_mean": float(np.mean(prbs["rx_in"])) if prbs["rx_in"].size else 0.0,
            "rx_in_std": float(np.std(prbs["rx_in"])) if prbs["rx_in"].size else 0.0,
            "rx_xtalk_in_mean": float(np.mean(prbs["rx_xtalk_in"])) if prbs["rx_xtalk_in"].size else 0.0,
            "rx_xtalk_in_std": float(np.std(prbs["rx_xtalk_in"])) if prbs["rx_xtalk_in"].size else 0.0,
            "pi_code_stats": pi_stats,
            "pd_counts": pd_counts,
        },
        "eye_metrics": eye_metrics_by_case,
        "eye_cdr_stability": eye_cdr_by_case,
        "rx_pi_code": int(link.rx.pi_code),
    }
    if run_feature_suite:
        feature_suite = _run_uni_feature_suite(show_plots=show_plots)
        summary["uni_feature_suite"] = feature_suite
        if not bool(feature_suite.get("passed")):
            fail_n = len(_as_list(feature_suite.get("failures")))
            sanity_n = len(_as_list(feature_suite.get("sanity_errors")))
            raise AssertionError(
                f"test_uni_feature suite failed (failures={fail_n}, sanity_errors={sanity_n})."
            )
    return summary


def _apply_dual_aggressors_once(link: Link, common: dict[str, Any]) -> tuple[str, float, dict[int, float]]:
    """Apply dual-direction aggressor settings for the simulation run."""
    aggr = _as_dict(common.get("aggressors"))
    if "ports" in aggr:
        link.set_aggressor_ports(_to_int_port_list(aggr.get("ports")))

    enabled = _as_bool(aggr.get("enable"), default=True)
    mode = str(aggr.get("source_mode", "manual")).strip().lower()
    amp = float(_as_dict(aggr.get("pattern_broadcast")).get("amplitude_v", Driver.AVDD))

    manual_sources = {int(k): float(v) for k, v in _as_dict(aggr.get("manual_sources_v")).items()}
    if enabled:
        if mode == "manual" and manual_sources:
            link.set_aggressor_sources(manual_sources)
        elif len(link.aggressor_ports) > 0:
            # Pattern aggressor modeling is not native in dual-dir Link;
            # apply static amplitude here and optional per-cycle random update in run loop.
            link.set_aggressor_sources({int(p): amp for p in link.aggressor_ports})
    elif len(link.aggressor_ports) > 0:
        link.set_aggressor_sources({int(p): 0.0 for p in link.aggressor_ports})

    return mode, amp, manual_sources


def _run_dual_sim(
    link: Link,
    spec: dict[str, Any],
    cycles_override: int | None = None,
    show_plots: bool = True,
) -> dict[str, Any]:
    """Run the dual-direction model flow from spec and collect metrics."""
    common = _as_dict(spec.get("common"))
    sim = _as_dict(spec.get("simulation"))
    num_cfg = _as_dict(sim.get("num_cycles"))

    seed = sim.get("random_seed")
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))

    mode, amp, _ = _apply_dual_aggressors_once(link, common=common)
    aggr_enabled = _as_bool(_as_dict(common.get("aggressors")).get("enable"), default=True)
    rng = np.random.default_rng(None if seed is None else int(seed))

    warmup = int(num_cfg.get("warmup", 0))
    run_cycles = int(num_cfg.get("prbs_run", 20000))
    total = int(cycles_override) if cycles_override is not None else int(warmup + run_cycles)

    for _ in range(total):
        if mode == "pattern" and aggr_enabled and len(link.aggressor_ports) > 0:
            link.set_aggressor_sources(
                {int(p): float(amp) * float(rng.integers(0, 2)) for p in link.aggressor_ports}
            )
        link.run()

    eye_metrics_master = link.master_rx.get_eye_metrics()
    eye_metrics_slave = link.slave_rx.get_eye_metrics()
    diag_metrics: dict[str, Any] | None = None
    if show_plots:
        diag_metrics = link.diagnostic(show=True)
        link.plot_chan_afe_impulses(time_unit=str(_as_dict(common.get("diagnostics")).get("path_impulse_time_unit", "ns")))

    summary = {
        "active_model": "dual_dir",
        "cycles_run": total,
        "victim_ports": {
            "port_one": int(link.chan_port_one_sel),
            "port_two": int(link.chan_port_two_sel),
        },
        "aggressor_ports": [int(p) for p in link.aggressor_ports],
        "eye_metrics_master": eye_metrics_master,
        "eye_metrics_slave": eye_metrics_slave,
    }
    if diag_metrics is not None:
        summary["diagnostic"] = {
            "master_eye_metrics": diag_metrics.get("master_eye_metrics"),
            "slave_eye_metrics": diag_metrics.get("slave_eye_metrics"),
        }
    return summary


def run_from_specs(
    spec_path: Path,
    model_override: str | None = None,
    cycles_override: int | None = None,
    show_plots: bool = True,
) -> dict[str, Any]:
    """Run the selected model from spec and return a summary dictionary."""
    spec_path = Path(spec_path).resolve()
    spec = _load_spec(spec_path)
    model = str(model_override or spec.get("active_model", "uni_dir")).strip().lower()
    outputs = _as_dict(spec.get("outputs"))
    cache_cfg = _as_dict(outputs.get("cache"))
    cache_enabled = _as_bool(cache_cfg.get("enable"), default=True)
    cache_replay_plots = _as_bool(cache_cfg.get("replay_plots"), default=True)
    cache_dir = _resolve_path(cache_cfg.get("dir", outputs.get("cache_dir", "artifacts/cache/specs_wrapper")))
    cache_key, cache_fingerprint = _build_run_cache_key(
        spec_path,
        model_override=model_override,
        cycles_override=cycles_override,
        show_plots=show_plots,
    )
    cache_entry = cache_dir / cache_key
    cache_result_file = cache_entry / "result.json"
    cache_meta_file = cache_entry / "meta.json"
    cache_fig_dir = cache_entry / "figures"

    if cache_enabled and cache_result_file.exists():
        out = json.loads(cache_result_file.read_text(encoding="utf-8"))
        replay_count = 0
        if show_plots and cache_replay_plots:
            replay_count = _replay_cached_figures(cache_fig_dir)
        out["_cache"] = {
            "enabled": True,
            "hit": True,
            "key": cache_key,
            "dir": str(cache_entry),
            "figures_replayed": int(replay_count),
        }
        if show_plots:
            plt.show()
        return out

    if model == "uni_dir":
        link = _build_uni_link_from_spec(spec)
        out = _run_uni_sim(
            link=link,
            spec=spec,
            cycles_override=cycles_override,
            show_plots=show_plots,
        )
    elif model == "dual_dir":
        link = _build_dual_link_from_spec(spec)
        out = _run_dual_sim(
            link=link,
            spec=spec,
            cycles_override=cycles_override,
            show_plots=show_plots,
        )
    else:
        raise ValueError(f"Unsupported active_model '{model}'. Use 'uni_dir' or 'dual_dir'.")

    if cache_enabled:
        cache_entry.mkdir(parents=True, exist_ok=True)
        figure_files: list[str] = []
        if show_plots and cache_replay_plots:
            figure_files = _save_open_figures_to_cache(cache_fig_dir)
        cache_meta = {
            "cache_key": cache_key,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "spec_path": str(spec_path),
            "active_model": model,
            "model_override": model_override,
            "cycles_override": cycles_override,
            "show_plots": bool(show_plots),
            "fingerprint": cache_fingerprint,
            "figures": figure_files,
        }
        cache_meta_file.write_text(json.dumps(cache_meta, indent=2, default=str), encoding="utf-8")
        cache_result_file.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        out["_cache"] = {
            "enabled": True,
            "hit": False,
            "key": cache_key,
            "dir": str(cache_entry),
            "figures_saved": len(figure_files),
        }
    else:
        out["_cache"] = {
            "enabled": False,
            "hit": False,
        }

    if show_plots:
        plt.show()
    return out


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the spec wrapper entry point."""
    parser = argparse.ArgumentParser(description="Run wireline model test wrapper from specs.yaml")
    parser.add_argument(
        "--spec",
        type=str,
        default=str(ROOT / "specs.yaml"),
        help="Path to specs.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["uni_dir", "dual_dir"],
        help="Override specs active_model",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Override total simulation cycles",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run without matplotlib figures",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for running the specs-based test wrapper."""
    args = _parse_args()
    result = run_from_specs(
        spec_path=Path(args.spec).resolve(),
        model_override=args.model,
        cycles_override=args.cycles,
        show_plots=not args.no_plots,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
