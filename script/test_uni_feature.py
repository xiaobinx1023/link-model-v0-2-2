from __future__ import annotations

from dataclasses import dataclass
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

EYE_X_UNIT = "ui"  # "ui" or time units such as "s", "second", "ns"


MonitorGetter = Callable[[UniDirLink], float]


def _get_data_ui_samples(link: UniDirLink) -> int:
    if hasattr(link, "data_ui_samples"):
        return int(getattr(link, "data_ui_samples"))
    data_rate_hz = float(getattr(link, "data_rate_hz", float(link.CLK_FREQ_HZ)))
    if data_rate_hz <= 0.0:
        return 1
    return max(1, int(round(float(link.SAMP_FREQ_HZ) / data_rate_hz)))


@dataclass
class PulseMetric:
    peak_abs: float
    peak_signed: float
    peak_idx: int
    peak_time_ns: float


def _print_trace_ber_mapping(tag: str, metrics: dict[str, float | int | None]) -> None:
    print(
        f"{tag} trace->BER map:",
        {
            "n_traces": metrics.get("n_traces"),
            "ber_floor_estimate": metrics.get("ber_floor_estimate"),
            "sigma_limit_estimate": metrics.get("sigma_limit_estimate"),
            "sigma_to_ber": metrics.get("sigma_to_ber"),
            "sigma_to_min_traces": metrics.get("sigma_to_min_traces"),
        },
    )


def _build_link(
    *,
    rx_pd_out_gain: float = 0.0,
    rx_pi_code: int = 0,
    ctle_en: bool = False,
    random_seed: int | None = None,
    deterministic_clock: bool = False,
) -> UniDirLink:
    if random_seed is not None:
        np.random.seed(int(random_seed))

    chan_file = ROOT / "data" / "A1_Combined_models_Slice_C_clocks_Slice_D_Data.s10p"
    link = UniDirLink(
        chan_file=chan_file,
        chan_port_tx_sel=7,
        chan_port_rx_sel=8,
        tx_pattern=Pattern.PRBS13,
        tx_ffe_taps=[0, 62, 0, 0],
        tx_pi_code=0,
        rx_pi_code=rx_pi_code,
        rx_term_code=63.0,
        rx_clk_ofst=4.0,
        rx_slicer_ref=0.5 * Driver.AVDD,
        rx_pd_out_gain=rx_pd_out_gain,
        channel_pairs=[(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)],
        aggressor_ports=None,
        aggressor_enable=True,
    )
    link.rx.ctle_en = bool(ctle_en)
    link.rx.ctle_dc_gain_db = 2.0
    link.rx.ctle_peaking_gain_db = 3.0
    link.rx.ctle_peaking_freq_hz = 32e9
    link.rx.dfe_en = False

    if deterministic_clock:
        # Make eye-comparison runs deterministic and jitter-free.
        cg = link.clk_src
        cg.abs_jitter_std_sec = 0.0
        cg._abs_jitter_prev = 0.0
        cg._abs_jitter = 0.0
        cg._period_jitter = 0.0
        cg._period = cg.nominal_period
        cg._timer = 0.0
        if hasattr(cg, "_reset_cycle_timers"):
            cg._reset_cycle_timers(carry_frac=0.0)
        else:
            cg._clk_q_pos_edge_timer_val = cg._period / 4.0
            cg._clk_i_neg_edge_timer_val = cg._period / 2.0
            cg._clk_q_neg_edge_timer_val = cg._period * 3.0 / 4.0
            cg._clk_i_pos_edge_timer_val = cg._period

    return link


def _monitor_getters() -> dict[str, MonitorGetter]:
    return {
        "tx_drv_out": lambda lk: float(lk.tx_drv_out),
        "tx_to_chan": lambda lk: float(lk.tx_to_chan),
        "rx_xtalk_bump": lambda lk: float(lk.rx_xtalk_bump),
        "rx_bump": lambda lk: float(lk.rx_bump),
        "rx_xtalk_in": lambda lk: float(lk.rx_xtalk_in),
        "rx_in": lambda lk: float(lk.rx_in),
        "rx_post_ctle": lambda lk: float(lk.rx.din_ctle),
        "rx_post_aperture": lambda lk: float(lk.rx.din_apertured),
    }


def _run_constant_pattern(
    link: UniDirLink,
    pattern: Pattern,
    num_cycles: int,
    monitor_keys: list[str],
) -> dict[str, np.ndarray]:
    getters = _monitor_getters()
    traces = {k: np.zeros(num_cycles, dtype=np.float64) for k in monitor_keys}
    link.tx_pattern = Pattern(pattern)
    for i in range(num_cycles):
        link.run()
        for k in monitor_keys:
            traces[k][i] = float(getters[k](link))
    return traces


def _summarize_pulse(t_ns: np.ndarray, traces: dict[str, np.ndarray]) -> dict[str, PulseMetric]:
    out: dict[str, PulseMetric] = {}
    for k, y in traces.items():
        yi = np.asarray(y, dtype=np.float64).reshape(-1)
        idx = int(np.argmax(np.abs(yi)))
        out[k] = PulseMetric(
            peak_abs=float(np.abs(yi[idx])),
            peak_signed=float(yi[idx]),
            peak_idx=idx,
            peak_time_ns=float(t_ns[idx]),
        )
    return out


def _plot_multi_traces(
    t_ns: np.ndarray,
    traces: dict[str, np.ndarray],
    title: str,
    pulse_start_ns: float | None = None,
    pulse_end_ns: float | None = None,
) -> None:
    keys = list(traces.keys())
    n = len(keys)
    n_cols = 2 if n > 1 else 1
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(3.0, 2.8 * n_rows)), sharex=True)
    ax_flat = np.asarray(axes).reshape(-1)
    for i, k in enumerate(keys):
        ax = ax_flat[i]
        ax.plot(t_ns, traces[k], linewidth=1.2)
        if pulse_start_ns is not None and pulse_end_ns is not None:
            ax.axvspan(pulse_start_ns, pulse_end_ns, color="orange", alpha=0.15)
        ax.set_title(k)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (ns)")
    for i in range(n, ax_flat.size):
        fig.delaxes(ax_flat[i])
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))


def _run_main_1ui_pulse(
    link: UniDirLink,
    monitor_keys: list[str],
    warmup: int = 128,
    tail: int = 256,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    getters = _monitor_getters()
    ui_samples = _get_data_ui_samples(link)
    warmup_edges = max(1, int(round(float(warmup) / float(max(1, ui_samples)))))
    pulse_edges = 1
    max_cycles = int(warmup + tail + max(64, 12 * max(1, ui_samples)))

    saved = {
        "tx_pattern": Pattern(link.tx_pattern),
        "aggr_enable": bool(link.aggressor_enable),
        "aggr_mode": str(link.aggressor_source_mode),
        "aggr_src": dict(link.aggressor_port_src),
    }
    traces_raw = {k: [] for k in monitor_keys}
    edge_count = 0
    pulse_start: int | None = None
    pulse_end: int | None = None
    try:
        link.set_aggressor_enable(False)
        link.set_aggressor_source_mode("manual")
        if link.aggressor_ports:
            link.set_aggressor_sources({p: 0.0 for p in link.aggressor_ports})
        for i in range(max_cycles):
            in_pulse_symbol = warmup_edges <= edge_count < (warmup_edges + pulse_edges)
            link.tx_pattern = Pattern.ALL_ONES if in_pulse_symbol else Pattern.ALL_ZEROS
            link.run()
            for k in monitor_keys:
                traces_raw[k].append(float(getters[k](link)))
            if bool(link.tx_clk_out.is_pos_edge):
                if edge_count == warmup_edges and pulse_start is None:
                    pulse_start = int(i)
                if edge_count == (warmup_edges + pulse_edges) and pulse_end is None:
                    pulse_end = int(i)
                edge_count += 1
            if pulse_end is not None and i >= int(pulse_end + tail):
                break
    finally:
        link.tx_pattern = saved["tx_pattern"]
        link.set_aggressor_enable(bool(saved["aggr_enable"]))
        link.set_aggressor_source_mode(str(saved["aggr_mode"]))
        if link.aggressor_ports:
            link.set_aggressor_sources({p: float(saved["aggr_src"].get(p, 0.0)) for p in link.aggressor_ports})

    n_samp = len(next(iter(traces_raw.values()))) if len(traces_raw) > 0 else 0
    if pulse_start is None:
        pulse_start = min(max(0, warmup), max(0, n_samp - 1))
    if pulse_end is None:
        pulse_end = min(max(0, pulse_start + max(1, ui_samples)), max(1, n_samp))
    pulse_start = int(max(0, min(pulse_start, max(0, n_samp - 1))))
    pulse_end = int(max(pulse_start + 1, min(pulse_end, max(1, n_samp))))

    traces = {k: np.asarray(v, dtype=np.float64) for k, v in traces_raw.items()}
    t_ns = np.arange(n_samp, dtype=np.float64) / link.SAMP_FREQ_HZ * 1e9

    _plot_multi_traces(
        t_ns=t_ns,
        traces=traces,
        title="6) 1-UI Pulse Response Over Data Chain",
        pulse_start_ns=pulse_start / link.SAMP_FREQ_HZ * 1e9,
        pulse_end_ns=pulse_end / link.SAMP_FREQ_HZ * 1e9,
    )
    return t_ns, traces


def _run_aggressor_1ui_pulse_to_rx(
    link: UniDirLink,
    aggressor_port: int,
    amplitude: float = Driver.AVDD,
    warmup: int = 128,
    tail: int = 256,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    getters = _monitor_getters()
    ui_samples = _get_data_ui_samples(link)
    pulse_len = ui_samples
    total = warmup + pulse_len + tail
    pulse_start = warmup
    pulse_end = warmup + pulse_len
    t_ns = np.arange(total, dtype=np.float64) / link.SAMP_FREQ_HZ * 1e9

    if int(aggressor_port) not in link.aggressor_ports:
        link.set_aggressor_ports(list(link.aggressor_ports) + [int(aggressor_port)])

    saved = {
        "tx_pattern": Pattern(link.tx_pattern),
        "aggr_enable": bool(link.aggressor_enable),
        "aggr_mode": str(link.aggressor_source_mode),
        "aggr_src": dict(link.aggressor_port_src),
    }
    monitor_keys = ["rx_xtalk_bump", "rx_xtalk_in", "rx_in"]
    traces = {k: np.zeros(total, dtype=np.float64) for k in monitor_keys}
    try:
        link.tx_pattern = Pattern.ALL_ZEROS
        link.set_aggressor_enable(True)
        link.set_aggressor_source_mode("manual")
        link.set_aggressor_sources({p: 0.0 for p in link.aggressor_ports})
        for i in range(total):
            val = float(amplitude) if pulse_start <= i < pulse_end else 0.0
            link.set_aggressor_sources({int(aggressor_port): val})
            link.run()
            for k in monitor_keys:
                traces[k][i] = float(getters[k](link))
    finally:
        link.tx_pattern = saved["tx_pattern"]
        link.set_aggressor_enable(bool(saved["aggr_enable"]))
        link.set_aggressor_source_mode(str(saved["aggr_mode"]))
        link.set_aggressor_sources({p: float(saved["aggr_src"].get(p, 0.0)) for p in link.aggressor_ports})

    _plot_multi_traces(
        t_ns=t_ns,
        traces=traces,
        title=f"7) Aggressor 1-UI Pulse To Victim RX Input (Port {aggressor_port})",
        pulse_start_ns=pulse_start / link.SAMP_FREQ_HZ * 1e9,
        pulse_end_ns=pulse_end / link.SAMP_FREQ_HZ * 1e9,
    )
    return t_ns, traces, pulse_start / link.SAMP_FREQ_HZ * 1e9


def _sanity_check_chain_pulse(summary: dict[str, PulseMetric]) -> list[str]:
    errors: list[str] = []
    req = ["tx_drv_out", "tx_to_chan", "rx_bump", "rx_in"]
    for k in req:
        m = summary.get(k)
        if m is None or not np.isfinite(m.peak_abs) or m.peak_abs <= 1e-9:
            errors.append(f"Missing/invalid pulse metric for {k}.")
    order = ["tx_drv_out", "tx_to_chan", "rx_bump", "rx_in"]
    times = [summary[k].peak_time_ns for k in order if k in summary]
    if len(times) == len(order):
        if not (times[0] <= times[1] <= times[2] <= times[3]):
            errors.append(f"Chain pulse delay order unexpected: {dict((k, summary[k].peak_time_ns) for k in order)}")
    return errors


def _sanity_check_aggressor_pulse(
    link: UniDirLink,
    aggressor_port: int,
    victim_port: int,
    summary: dict[str, PulseMetric],
    pulse_start_ns: float,
) -> list[str]:
    errors: list[str] = []
    req = ["rx_xtalk_bump", "rx_xtalk_in"]
    for k in req:
        m = summary.get(k)
        if m is None or not np.isfinite(m.peak_abs) or m.peak_abs <= 1e-9:
            errors.append(f"Missing/invalid aggressor pulse metric for {k}.")
    if all(k in summary for k in req):
        if summary["rx_xtalk_in"].peak_time_ns < summary["rx_xtalk_bump"].peak_time_ns:
            errors.append("Aggressor pulse delay mismatch: rx_xtalk_in peaks earlier than rx_xtalk_bump.")

    model = link.get_aggressor_to_victim_port_pulse_response(
        aggressor_port=aggressor_port,
        victim_port=victim_port,
    )
    y_model = np.asarray(model["impulse"], dtype=np.float64)
    t_model_ns = np.asarray(model["time_sec"], dtype=np.float64) * 1e9
    idx_model = int(np.argmax(np.abs(y_model)))
    t_model_peak = float(t_model_ns[idx_model])
    t_meas_peak_rel = float(summary["rx_xtalk_bump"].peak_time_ns) - float(pulse_start_ns)
    data_rate_hz = float(getattr(link, "data_rate_hz", float(link.CLK_FREQ_HZ)))
    ui_ns = 1e9 / data_rate_hz
    if abs(t_model_peak - t_meas_peak_rel) > 1.5 * ui_ns:
        errors.append(
            "Aggressor bump delay mismatch versus model: "
            f"measured_rel={t_meas_peak_rel:.4f}ns model={t_model_peak:.4f}ns"
        )
    return errors


def _select_strongest_aggressor_port(link: UniDirLink) -> int:
    if len(link.aggressor_ports) == 0:
        raise ValueError("No aggressor ports are enabled.")
    peaks: list[tuple[int, float]] = []
    for p in link.aggressor_ports:
        imp = np.asarray(link.imp_xtalk_to_rx_bump_by_port.get(p, np.array([0.0])), dtype=np.float64)
        peaks.append((int(p), float(np.max(np.abs(imp)))))
    peaks.sort(key=lambda x: x[1], reverse=True)
    best_port, best_peak = peaks[0]
    if best_peak <= 1e-12:
        raise AssertionError(f"All aggressor impulses are near zero at victim RX: {peaks}")
    return int(best_port)


def test_1_dc_levels() -> dict[str, float]:
    link = _build_link(rx_pd_out_gain=0.0, ctle_en=False)
    link.set_aggressor_enable(False)
    monitor_keys = ["tx_drv_out", "rx_in"]
    n = 6000
    tr0 = _run_constant_pattern(link, Pattern.ALL_ZEROS, n, monitor_keys)
    tr1 = _run_constant_pattern(link, Pattern.ALL_ONES, n, monitor_keys)

    tail = slice(int(0.6 * n), n)
    res = {
        "tx_mean_zero": float(np.mean(tr0["tx_drv_out"][tail])),
        "tx_mean_one": float(np.mean(tr1["tx_drv_out"][tail])),
        "rx_mean_zero": float(np.mean(tr0["rx_in"][tail])),
        "rx_mean_one": float(np.mean(tr1["rx_in"][tail])),
    }
    print("1) DC level test:", res)
    if not (res["tx_mean_one"] > res["tx_mean_zero"]):
        raise AssertionError("1) DC level test failed at TX.")
    if not (res["rx_mean_one"] > res["rx_mean_zero"]):
        raise AssertionError("1) DC level test failed at RX.")

    x = np.arange(n, dtype=np.float64)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(x, tr0["tx_drv_out"], label="ALL_ZEROS")
    axs[0].plot(x, tr1["tx_drv_out"], label="ALL_ONES")
    axs[0].set_title("1) TX DC Level Test")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    axs[1].plot(x, tr0["rx_in"], label="ALL_ZEROS")
    axs[1].plot(x, tr1["rx_in"], label="ALL_ONES")
    axs[1].set_title("1) RX DC Level Test")
    axs[1].set_xlabel("Sample")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    fig.tight_layout()
    return res


def test_2_step_patterns() -> None:
    link = _build_link(rx_pd_out_gain=0.0, ctle_en=False)
    link.set_aggressor_enable(False)

    def run_step(p0: Pattern, p1: Pattern, pre: int = 512, post: int = 1024):
        n = pre + post
        tx = np.zeros(n, dtype=np.float64)
        rx = np.zeros(n, dtype=np.float64)
        for i in range(n):
            link.tx_pattern = p0 if i < pre else p1
            link.run()
            tx[i] = float(link.tx_drv_out)
            rx[i] = float(link.rx_in)
        return tx, rx

    tx_up, rx_up = run_step(Pattern.ALL_ZEROS, Pattern.ALL_ONES)
    tx_dn, rx_dn = run_step(Pattern.ALL_ONES, Pattern.ALL_ZEROS)
    print(
        "2) STEP test levels:",
        {
            "up_tx_delta": float(np.mean(tx_up[-128:]) - np.mean(tx_up[:128])),
            "up_rx_delta": float(np.mean(rx_up[-128:]) - np.mean(rx_up[:128])),
            "dn_tx_delta": float(np.mean(tx_dn[-128:]) - np.mean(tx_dn[:128])),
            "dn_rx_delta": float(np.mean(rx_dn[-128:]) - np.mean(rx_dn[:128])),
        },
    )

    x = np.arange(tx_up.size, dtype=np.float64)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(x, tx_up, label="TX 0->1")
    axs[0].plot(x, rx_up, label="RX 0->1")
    axs[0].set_title("2) STEP Pattern 0 -> 1")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    axs[1].plot(x, tx_dn, label="TX 1->0")
    axs[1].plot(x, rx_dn, label="RX 1->0")
    axs[1].set_title("2) STEP Pattern 1 -> 0")
    axs[1].set_xlabel("Sample")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    fig.tight_layout()


def test_3_cdr_loop_pi_tracking() -> None:
    link = _build_link(rx_pd_out_gain=0.125, rx_pi_code=48, ctle_en=False)
    link.tx_pattern = Pattern.PRBS13
    link.set_aggressor_enable(False)
    # Intentionally offset slicer reference so PD toggles and PI movement is visible.
    link.rx.ref = 0.2
    n = 50000
    pi_code: list[int] = []
    pd_out: list[int] = []
    for _ in range(n):
        link.run()
        if link.rx.clk.is_edge:
            pi_code.append(int(link.rx.pi_code))
            pd_out.append(int(link.rx._pd_out))
    pi = np.asarray(pi_code, dtype=np.int16)
    pd = np.asarray(pd_out, dtype=np.int8)
    print(
        "3) CDR loop stats:",
        {
            "n_edges": int(pi.size),
            "pi_min": int(np.min(pi)) if pi.size else None,
            "pi_max": int(np.max(pi)) if pi.size else None,
            "pi_std": float(np.std(pi)) if pi.size else None,
            "pd_counts": {
                "-1": int(np.sum(pd == -1)),
                "0": int(np.sum(pd == 0)),
                "+1": int(np.sum(pd == 1)),
            },
        },
    )
    if pi.size == 0:
        raise AssertionError("3) CDR loop test failed: no PI codes captured.")

    x = np.arange(pi.size, dtype=np.float64)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(x, pi, linewidth=1.0)
    axs[0].set_title("3) CDR PI Code Tracking")
    axs[0].set_ylabel("PI Code")
    axs[0].grid(True, alpha=0.3)
    axs[1].plot(x, np.cumsum(pd), linewidth=1.0)
    axs[1].set_title("3) CDR PD Output Cumulative Sum")
    axs[1].set_xlabel("RX Sampling Edge Index")
    axs[1].set_ylabel("Cum PD")
    axs[1].grid(True, alpha=0.3)
    fig.tight_layout()


def _run_eye(link: UniDirLink, *, pattern: Pattern, cycles: int, with_xtalk: bool) -> dict[str, float | int | None]:
    link.tx_pattern = Pattern(pattern)
    link.set_aggressor_enable(bool(with_xtalk))
    if with_xtalk:
        link.set_aggressor_source_mode("pattern")
        link.broadcast_aggressor_pattern(Pattern.PRBS7, amplitude=Driver.AVDD)
    else:
        link.set_aggressor_source_mode("manual")
        if link.aggressor_ports:
            link.set_aggressor_sources({p: 0.0 for p in link.aggressor_ports})
    for _ in range(cycles):
        link.run()
    return link.rx.get_eye_metrics()


def test_4_eye_prbs13_1024() -> dict[str, float | int | None]:
    link = _build_link(rx_pd_out_gain=0.0, ctle_en=False)
    m = _run_eye(link, pattern=Pattern.PRBS13, cycles=35000, with_xtalk=False)
    print("4) PRBS13 eye metrics:", m)
    _print_trace_ber_mapping("4)", m)
    if int(m.get("n_traces") or 0) < 1024:
        raise AssertionError(f"4) Eye trace count too low: {m.get('n_traces')}")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    link.rx.plot_eye(ax=ax, mask_type="diamond", mask_sigma=1.0, x_unit=EYE_X_UNIT)
    ax.set_title(f"4) Eye Diagram PRBS13 (1024 traces, x={EYE_X_UNIT})")
    fig.tight_layout()
    return m


def test_5_eye_with_without_xtalk() -> tuple[dict[str, float | int | None], dict[str, float | int | None]]:
    # Use same seed + deterministic clock so comparison is apples-to-apples.
    seed = 20260305
    link_wo = _build_link(rx_pd_out_gain=0.0, ctle_en=False, random_seed=seed, deterministic_clock=True)
    m_wo = _run_eye(link_wo, pattern=Pattern.PRBS13, cycles=35000, with_xtalk=False)
    link_w = _build_link(rx_pd_out_gain=0.0, ctle_en=False, random_seed=seed, deterministic_clock=True)
    m_w = _run_eye(link_w, pattern=Pattern.PRBS13, cycles=35000, with_xtalk=True)
    print("5) Eye wo xtalk:", m_wo)
    print("5) Eye w xtalk:", m_w)
    _print_trace_ber_mapping("5) wo", m_wo)
    _print_trace_ber_mapping("5) w", m_w)
    if (
        m_wo.get("eye_width_mean") is not None
        and m_w.get("eye_width_mean") is not None
        and m_wo.get("eye_height_mean") is not None
        and m_w.get("eye_height_mean") is not None
    ):
        print(
            "5) Eye delta (w - wo):",
            {
                "width_delta": float(m_w["eye_width_mean"] - m_wo["eye_width_mean"]),
                "height_delta": float(m_w["eye_height_mean"] - m_wo["eye_height_mean"]),
            },
        )

    fig, axs = plt.subplots(1, 2, figsize=(14, 4), squeeze=False)
    link_wo.rx.plot_eye(ax=axs[0, 0], mask_type="diamond", mask_sigma=1.0, x_unit=EYE_X_UNIT)
    axs[0, 0].set_title(f"5) Eye Without Xtalk (x={EYE_X_UNIT})")
    link_w.rx.plot_eye(ax=axs[0, 1], mask_type="diamond", mask_sigma=1.0, x_unit=EYE_X_UNIT)
    axs[0, 1].set_title(f"5) Eye With Xtalk (x={EYE_X_UNIT})")
    fig.tight_layout()
    return m_wo, m_w


def test_6_1ui_pulse_chain() -> tuple[dict[str, PulseMetric], list[str]]:
    link = _build_link(rx_pd_out_gain=0.0, ctle_en=False)
    monitor_keys = ["tx_drv_out", "tx_to_chan", "rx_bump", "rx_in", "rx_post_aperture"]
    t_ns, traces = _run_main_1ui_pulse(link, monitor_keys=monitor_keys)
    s = _summarize_pulse(t_ns, traces)
    print("6) Chain pulse summary:", {k: vars(v) for k, v in s.items()})
    errs = _sanity_check_chain_pulse(s)
    return s, errs


def test_7_aggressor_1ui_pulse_to_victim_rx() -> tuple[dict[str, PulseMetric], list[str]]:
    link = _build_link(rx_pd_out_gain=0.0, ctle_en=False)
    aggressor_port = _select_strongest_aggressor_port(link)
    victim_port = 8
    t_ns, traces, pulse_start_ns = _run_aggressor_1ui_pulse_to_rx(
        link, aggressor_port=aggressor_port, amplitude=Driver.AVDD
    )
    model = link.get_aggressor_to_victim_port_pulse_response(
        aggressor_port=aggressor_port,
        victim_port=victim_port,
    )
    link.plot_aggressor_to_victim_port_pulse_response(
        aggressor_port=aggressor_port,
        victim_port=victim_port,
        require_coupling=str(model["coupling_type"]),
        time_unit="ns",
    )
    s = _summarize_pulse(t_ns, traces)
    print(
        "7) Aggressor pulse summary:",
        {
            "aggressor_port": aggressor_port,
            "victim_port": victim_port,
            "coupling_type": model["coupling_type"],
            "metrics": {k: vars(v) for k, v in s.items()},
        },
    )
    errs = _sanity_check_aggressor_pulse(
        link=link,
        aggressor_port=aggressor_port,
        victim_port=victim_port,
        summary=s,
        pulse_start_ns=pulse_start_ns,
    )
    return s, errs


def main() -> None:
    print("Running uni-link feature tests...")
    test_1_dc_levels()
    test_2_step_patterns()
    test_3_cdr_loop_pi_tracking()
    test_4_eye_prbs13_1024()
    test_5_eye_with_without_xtalk()
    _, err6 = test_6_1ui_pulse_chain()
    _, err7 = test_7_aggressor_1ui_pulse_to_victim_rx()

    all_errors = [*err6, *err7]
    if all_errors:
        print("8) Pulse sanity check FAILED:")
        for e in all_errors:
            print("  -", e)
        raise AssertionError("8) Sanity check failed. See messages above.")
    print("8) Pulse sanity check PASS")

    plt.show()


if __name__ == "__main__":
    main()
