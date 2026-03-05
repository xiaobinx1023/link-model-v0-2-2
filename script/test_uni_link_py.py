from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_py.link_model import Pattern
from src_py.link_model.driver import Driver
from src_py_uni.uni_link_model import UniDirLink


def _get_monitor_definitions() -> dict[str, tuple[str, Callable[[UniDirLink], float]]]:
    return {
        "tx_drv_out": ("TX driver out", lambda lk: float(lk.tx_drv_out)),
        "tx_to_chan": ("TX -> channel", lambda lk: float(lk.tx_to_chan)),
        "rx_xtalk_bump": ("RX bump (xtalk only)", lambda lk: float(lk.rx_xtalk_bump)),
        "rx_bump": ("RX bump node", lambda lk: float(lk.rx_bump)),
        "rx_xtalk_in": ("RX in (xtalk only)", lambda lk: float(lk.rx_xtalk_in)),
        "rx_in_pre_ctle": ("RX in (pre-CTLE)", lambda lk: float(lk.rx_in)),
        "rx_post_ctle": ("RX post-CTLE", lambda lk: float(lk.rx.din_ctle)),
        "rx_post_dfe": ("RX post-DFE", lambda lk: float(lk.rx.din_eq)),
        "rx_post_aperture": ("RX post-aperture", lambda lk: float(lk.rx.din_apertured)),
    }


def _resolve_monitor_keys(monitor_keys: list[str] | None) -> list[tuple[str, str, Callable[[UniDirLink], float]]]:
    defs = _get_monitor_definitions()
    if monitor_keys is None:
        keys = list(defs.keys())
    else:
        keys = []
        seen: set[str] = set()
        for raw in monitor_keys:
            key = str(raw).strip()
            if key == "" or key in seen:
                continue
            seen.add(key)
            keys.append(key)
        unknown = [k for k in keys if k not in defs]
        if unknown:
            raise ValueError(f"Unknown monitor key(s): {unknown}. Available: {list(defs.keys())}")
    return [(k, defs[k][0], defs[k][1]) for k in keys]


def _plot_monitor_traces(
    t_ns: np.ndarray,
    traces: dict[str, np.ndarray],
    monitors: list[tuple[str, str, Callable[[UniDirLink], float]]],
    title: str,
    pulse_start_ns: float | None = None,
    pulse_end_ns: float | None = None,
    cursor_samples_by_monitor: dict[str, dict[str, int]] | None = None,
) -> None:
    n = len(monitors)
    if n == 0:
        return
    n_cols = 2 if n > 1 else 1
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(3.0, 2.8 * n_rows)), sharex=True)
    axes_flat = np.asarray(axes).reshape(-1)

    for i, (key, label, _) in enumerate(monitors):
        ax = axes_flat[i]
        ax.plot(t_ns, traces[key], linewidth=1.1)
        if pulse_start_ns is not None and pulse_end_ns is not None:
            ax.axvspan(pulse_start_ns, pulse_end_ns, color="orange", alpha=0.15)
        if cursor_samples_by_monitor is not None:
            cursor_spec = [
                ("pre", "tab:purple", ":"),
                ("main", "tab:red", "--"),
                ("post1", "tab:green", "-."),
                ("post2", "tab:olive", "-."),
                ("post3", "tab:blue", "-."),
            ]
            cursor_samples = cursor_samples_by_monitor.get(key, {})
            for cname, ccolor, cstyle in cursor_spec:
                idx = int(cursor_samples.get(cname, -1))
                if idx < 0 or idx >= traces[key].size:
                    continue
                x = float(t_ns[idx])
                y = float(traces[key][idx])
                ax.axvline(x, color=ccolor, linestyle=cstyle, linewidth=1.0, alpha=0.8)
                ax.plot([x], [y], marker="o", color=ccolor, markersize=4)
                ax.annotate(
                    cname,
                    xy=(x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=ccolor,
                )
        ax.set_title(label)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (ns)")

    for i in range(n, axes_flat.size):
        fig.delaxes(axes_flat[i])

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))


def _estimate_cursor_samples_by_monitor(
    traces: dict[str, np.ndarray],
    ui_samples: int,
    monitor_order: list[str],
) -> dict[str, dict[str, int]]:
    if ui_samples <= 0:
        return {
            key: {"pre": -1, "main": -1, "post1": -1, "post2": -1, "post3": -1}
            for key in monitor_order
        }

    out: dict[str, dict[str, int]] = {}
    ui = int(ui_samples)
    for key in monitor_order:
        y = np.asarray(traces.get(key, np.array([], dtype=np.float64)), dtype=np.float64).reshape(-1)
        if y.size == 0:
            out[key] = {"pre": -1, "main": -1, "post1": -1, "post2": -1, "post3": -1}
            continue

        main_idx = int(np.argmax(np.abs(y)))
        out[key] = {
            "pre": int(main_idx - ui),
            "main": int(main_idx),
            "post1": int(main_idx + ui),
            "post2": int(main_idx + 2 * ui),
            "post3": int(main_idx + 3 * ui),
        }
    return out


def run_1ui_pulse_response(
    link: UniDirLink,
    monitor_keys: list[str] | None = None,
) -> dict[str, object]:
    """
    Force a 1-UI pulse at TX by toggling ALL_ZEROS/ALL_ONES around one UI window.
    """
    monitors = _resolve_monitor_keys(monitor_keys)

    ui_samples = int(round(link.SAMP_FREQ_HZ / link.CLK_FREQ_HZ))
    warmup = 128
    pulse_len = ui_samples
    tail = 256
    total_cycles = warmup + pulse_len + tail
    pulse_start = warmup
    pulse_end = warmup + pulse_len

    t_ns = np.arange(total_cycles, dtype=np.float64) / link.SAMP_FREQ_HZ * 1e9
    pulse_start_ns = pulse_start / link.SAMP_FREQ_HZ * 1e9
    pulse_end_ns = pulse_end / link.SAMP_FREQ_HZ * 1e9

    saved_pattern = Pattern(link.tx_pattern)
    saved_aggr_enable = bool(link.aggressor_enable)
    saved_aggr_src = dict(link.aggressor_port_src)
    traces_raw: dict[str, list[float]] = {k: [] for k, _, _ in monitors}

    try:
        link.set_aggressor_enable(False)
        if link.aggressor_ports:
            link.set_aggressor_sources({p: 0.0 for p in link.aggressor_ports})
        for t in range(total_cycles):
            link.tx_pattern = Pattern.ALL_ONES if pulse_start <= t < pulse_end else Pattern.ALL_ZEROS
            link.run()
            for key, _, getter in monitors:
                traces_raw[key].append(float(getter(link)))
    finally:
        link.tx_pattern = saved_pattern
        link.set_aggressor_enable(saved_aggr_enable)
        if link.aggressor_ports:
            link.set_aggressor_sources({p: float(saved_aggr_src.get(p, 0.0)) for p in link.aggressor_ports})

    traces = {k: np.asarray(v, dtype=np.float64) for k, v in traces_raw.items()}
    monitor_order = [k for k, _, _ in monitors]
    cursor_samples_by_monitor = _estimate_cursor_samples_by_monitor(
        traces=traces,
        ui_samples=ui_samples,
        monitor_order=monitor_order,
    )
    _plot_monitor_traces(
        t_ns=t_ns,
        traces=traces,
        monitors=monitors,
        title="UniDir 1-UI Pulse Response Monitors",
        pulse_start_ns=pulse_start_ns,
        pulse_end_ns=pulse_end_ns,
        cursor_samples_by_monitor=cursor_samples_by_monitor,
    )

    cursor_time_ns_by_monitor = {
        key: {
            cname: (float(t_ns[idx]) if 0 <= int(idx) < t_ns.size else float("nan"))
            for cname, idx in cdict.items()
        }
        for key, cdict in cursor_samples_by_monitor.items()
    }

    cursor_samples = cursor_samples_by_monitor.get(monitor_order[0], {}) if len(monitor_order) > 0 else {}
    cursor_time_ns = cursor_time_ns_by_monitor.get(monitor_order[0], {}) if len(monitor_order) > 0 else {}

    return {
        "time_ns": t_ns,
        "monitor_keys": [k for k, _, _ in monitors],
        "monitor_traces": traces,
        "cursor_samples_by_monitor": cursor_samples_by_monitor,
        "cursor_time_ns_by_monitor": cursor_time_ns_by_monitor,
        "cursor_samples": cursor_samples,
        "cursor_time_ns": cursor_time_ns,
        "ui_samples": ui_samples,
        "pulse_start_sample": pulse_start,
        "pulse_end_sample": pulse_end,
    }


def run_prbs(
    link: UniDirLink,
    num_cycles: int = 20000,
    with_aggressors: bool = False,
    aggressor_drive_mode: str = "manual_random",
    aggressor_amplitude: float = Driver.AVDD,
    aggressor_pattern_broadcast: Pattern = Pattern.PRBS31,
    aggressor_pattern_overrides: dict[int, Pattern] | None = None,
    aggressor_seed: int = 20260304,
) -> dict[str, np.ndarray]:
    link.tx_pattern = Pattern.PRBS23
    link.set_aggressor_enable(bool(with_aggressors))
    mode = str(aggressor_drive_mode).strip().lower()
    if mode not in {"manual_random", "pattern"}:
        raise ValueError("aggressor_drive_mode must be 'manual_random' or 'pattern'")
    if mode == "pattern":
        link.set_aggressor_source_mode("pattern")
        link.broadcast_aggressor_pattern(aggressor_pattern_broadcast, amplitude=float(aggressor_amplitude))
        if aggressor_pattern_overrides:
            for p, pat in aggressor_pattern_overrides.items():
                link.set_aggressor_pattern(int(p), pat, amplitude=float(aggressor_amplitude))
    else:
        link.set_aggressor_source_mode("manual")

    rx_in = np.zeros(num_cycles, dtype=np.float64)
    rx_xtalk_bump = np.zeros(num_cycles, dtype=np.float64)
    rx_xtalk_in = np.zeros(num_cycles, dtype=np.float64)
    rx_post_ctle = np.zeros(num_cycles, dtype=np.float64)
    rx_post_ap = np.zeros(num_cycles, dtype=np.float64)
    rx_data_edges: list[int] = []
    tx_data_edges: list[int] = []
    rx_pi_code: list[int] = []
    rx_pd_out: list[int] = []

    rng = np.random.default_rng(int(aggressor_seed))
    aggr_src = {p: 0.0 for p in link.aggressor_ports}
    if link.aggressor_ports and mode == "manual_random":
        link.set_aggressor_sources(aggr_src)

    for i in range(num_cycles):
        if link.aggressor_ports and mode == "manual_random":
            link.set_aggressor_sources(aggr_src)
        link.run()

        rx_in[i] = float(link.rx_in)
        rx_xtalk_bump[i] = float(link.rx_xtalk_bump)
        rx_xtalk_in[i] = float(link.rx_xtalk_in)
        rx_post_ctle[i] = float(link.rx.din_ctle)
        rx_post_ap[i] = float(link.rx.din_apertured)

        if link.tx_pi.clk_out.is_edge:
            tx_data_edges.append(int(link.tx.data_center))
            if mode == "manual_random" and with_aggressors and link.aggressor_ports:
                aggr_src = {
                    p: float(aggressor_amplitude) * float(rng.integers(0, 2))
                    for p in link.aggressor_ports
                }
            elif mode == "manual_random" and link.aggressor_ports:
                aggr_src = {p: 0.0 for p in link.aggressor_ports}

        if link.rx_pi.clk_out.is_edge:
            rx_data_edges.append(int(link.rx.data))
            rx_pi_code.append(int(link.rx.pi_code))
            rx_pd_out.append(int(link.rx._pd_out))

    return {
        "rx_in": rx_in,
        "rx_xtalk_bump": rx_xtalk_bump,
        "rx_xtalk_in": rx_xtalk_in,
        "rx_post_ctle": rx_post_ctle,
        "rx_post_aperture": rx_post_ap,
        "tx_bits_at_edges": np.asarray(tx_data_edges, dtype=np.int8),
        "rx_bits_at_edges": np.asarray(rx_data_edges, dtype=np.int8),
        "rx_pi_code": np.asarray(rx_pi_code, dtype=np.int16),
        "rx_pd_out": np.asarray(rx_pd_out, dtype=np.int8),
    }


def _build_default_link(chan_file: Path, rx_pd_out_gain: float) -> UniDirLink:
    link = UniDirLink(
        chan_file=chan_file,
        chan_port_tx_sel=7,
        chan_port_rx_sel=8,
        tx_pattern=Pattern.PRBS31,
        # Segment counts: [pre1, main, post1, post2]
        tx_ffe_taps=[0, 62, 0, 0],
        tx_pi_code=0,
        rx_pi_code=0,
        rx_term_code=63.0,
        rx_clk_ofst=4.0,
        rx_slicer_ref=0.5 * Driver.AVDD,
        rx_pd_out_gain=rx_pd_out_gain,
        channel_pairs=[(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)],
        aggressor_ports=None,
        aggressor_enable=True,
    )

    link.rx.ctle_en = False
    link.rx.ctle_dc_gain_db = 2
    link.rx.ctle_peaking_gain_db = 3.0
    link.rx.ctle_peaking_freq_hz = 32e9
    link.rx.ctle_zero_freq_hz = np.array([], dtype=np.float64)
    link.rx.ctle_pole_freq_hz = np.array([], dtype=np.float64)
    link.rx.dfe_en = False
    return link


def main() -> None:
    chan_file = ROOT / "data" / "A1_Combined_models_Slice_C_clocks_Slice_D_Data.s10p"
    rx_pd_out_gain = 0.125

    # Flag options for eye-diagram comparison.
    plot_eye_without_aggressors = True
    plot_eye_with_aggressors = True
    eye_cycles = 30000
    eye_x_unit = "ui"  # "ui" or time units such as "s", "second", "ns"
    aggressor_drive_mode = "pattern"  # "manual_random" or "pattern"
    aggressor_prbs_amplitude = Driver.AVDD
    aggressor_pattern_broadcast = Pattern.PRBS7
    aggressor_pattern_overrides: dict[int, Pattern] = {
        1: Pattern.ALL_ONES,
    }
    aggressor_seed = 20260304

    # Main waveform run flag.
    main_trace_with_aggressors = False

    # Specific aggressor -> victim FEXT pulse-response option.
    plot_specific_fext_pulse = True
    fext_aggressor_port = 5
    fext_victim_port = 8

    link = _build_default_link(chan_file=chan_file, rx_pd_out_gain=rx_pd_out_gain)

    print("TX FFE segment taps:", link.tx.ffe_taps)
    print("TX FFE normalized weights:", link.tx.normalize_weights_from_segments(link.tx.ffe_taps))
    print("RX term code:", link.rx_term_code)
    print("RX pd_out_gain:", link.rx.pd_out_gain)
    print("RX PI mode:", "CDR tracking" if link.rx.pd_out_gain != 0.0 else "fixed rx_pi_code")
    print("RX slicer ref:", link.rx.ref)
    print("RX edge clock offset (samples):", link.rx.clk_ofst)
    print("Aggressor ports (default one TX-side port per aggressor lane):", link.aggressor_ports)
    print("Eye x-axis unit:", eye_x_unit)
    print(
        "Aggressor data config:",
        {
            "drive_mode": aggressor_drive_mode,
            "broadcast_pattern": aggressor_pattern_broadcast.name,
            "pattern_overrides": {int(k): v.name for k, v in aggressor_pattern_overrides.items()},
            "amplitude": float(aggressor_prbs_amplitude),
        },
    )
    print("TX driver R(ohm) from FFE:", link.io_term.tx_driver_res_ohm_from_ffe(link.tx.ffe_taps))
    print("RX term R(ohm) from code:", link.io_term.rx_term_res_ohm_from_code(link.rx_term_code))

    link.plot_path_impulses(time_unit="ns")
    if plot_specific_fext_pulse:
        fext_data = link.plot_aggressor_to_victim_port_pulse_response(
            aggressor_port=fext_aggressor_port,
            victim_port=fext_victim_port,
            require_coupling="FEXT",
            time_unit="ns",
        )
        print(
            "Specific aggressor pulse:",
            {
                "aggressor_port": fext_data["aggressor_port"],
                "victim_port": fext_data["victim_port"],
                "coupling_type": fext_data["coupling_type"],
            },
        )

    pulse_res = run_1ui_pulse_response(
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
    print("Pulse monitor keys:", pulse_res["monitor_keys"])

    prbs = run_prbs(
        link,
        num_cycles=20000,
        with_aggressors=bool(main_trace_with_aggressors),
        aggressor_drive_mode=aggressor_drive_mode,
        aggressor_amplitude=aggressor_prbs_amplitude,
        aggressor_pattern_broadcast=aggressor_pattern_broadcast,
        aggressor_pattern_overrides=aggressor_pattern_overrides,
        aggressor_seed=aggressor_seed,
    )
    if prbs["rx_pi_code"].size > 0:
        pi_codes = np.asarray(prbs["rx_pi_code"], dtype=np.int16)
        pi_hist = np.bincount(pi_codes, minlength=128)
        lock_code = int(np.argmax(pi_hist))
        # Circular phase-code spread around dominant lock code.
        pi_delta = ((pi_codes - lock_code + 64) % 128) - 64
        top_codes = np.argsort(pi_hist)[-5:][::-1]
        print(
            "RX CDR PI code stats:",
            {
                "lock_code": lock_code,
                "delta_std": float(np.std(pi_delta)),
                "delta_pp": float(np.max(pi_delta) - np.min(pi_delta)),
                "n_unique_codes": int(np.count_nonzero(pi_hist)),
                "top_codes": [(int(c), int(pi_hist[c])) for c in top_codes if pi_hist[c] > 0],
            },
        )
    if prbs["rx_pd_out"].size > 0:
        pd = prbs["rx_pd_out"]
        print(
            "RX PD output counts:",
            {
                "-1": int(np.sum(pd == -1)),
                "0": int(np.sum(pd == 0)),
                "+1": int(np.sum(pd == 1)),
            },
        )

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

    eye_cases: list[tuple[str, UniDirLink]] = []
    if plot_eye_without_aggressors:
        link_wo = _build_default_link(chan_file=chan_file, rx_pd_out_gain=rx_pd_out_gain)
        run_prbs(
            link_wo,
            num_cycles=eye_cycles,
            with_aggressors=False,
            aggressor_drive_mode=aggressor_drive_mode,
            aggressor_amplitude=aggressor_prbs_amplitude,
            aggressor_pattern_broadcast=aggressor_pattern_broadcast,
            aggressor_pattern_overrides=aggressor_pattern_overrides,
            aggressor_seed=aggressor_seed,
        )
        eye_cases.append(("Without Aggressors", link_wo))
    if plot_eye_with_aggressors:
        link_w = _build_default_link(chan_file=chan_file, rx_pd_out_gain=rx_pd_out_gain)
        run_prbs(
            link_w,
            num_cycles=eye_cycles,
            with_aggressors=True,
            aggressor_drive_mode=aggressor_drive_mode,
            aggressor_amplitude=aggressor_prbs_amplitude,
            aggressor_pattern_broadcast=aggressor_pattern_broadcast,
            aggressor_pattern_overrides=aggressor_pattern_overrides,
            aggressor_seed=aggressor_seed,
        )
        eye_cases.append(("With Aggressors", link_w))

    if len(eye_cases) > 0:
        fig_eye, ax_eye = plt.subplots(1, len(eye_cases), figsize=(7.2 * len(eye_cases), 4), squeeze=False)
        for i, (label, eye_link) in enumerate(eye_cases):
            _, eye_metrics = eye_link.rx.plot_eye(
                ax=ax_eye[0, i],
                mask_type="diamond",
                mask_sigma=1.0,
                x_unit=eye_x_unit,
                return_metrics=True,
            )
            xc = float(eye_metrics.get("x_center_in_unit", eye_metrics.get("x_center", float("nan"))))
            ax_eye[0, i].set_title(f"{label} (x_center={xc:.3f} {eye_x_unit})")
            print(f"RX eye metrics ({label}):", eye_metrics)
        fig_eye.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
