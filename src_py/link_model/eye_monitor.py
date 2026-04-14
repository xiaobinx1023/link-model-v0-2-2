

from __future__ import annotations

from typing import Any, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from scipy.stats import norm

from .circular_buffer import CircularBuffer
from .clock import Clock


class EyeMonitor:
    def __init__(self, num_traces: int = 4096, num_samples_per_trace: int = 16) -> None:
        self.clk_in = Clock()
        self.data_in = 0.0

        self._num_traces = int(num_traces)
        self._num_samples_per_trace = int(num_samples_per_trace)
        # One eye trace spans 2 UI by default in this model.
        self._trace_span_ui = 2.0
        self._sample_rate_hz = 1.0
        self._delay_buffer: CircularBuffer[float] = CircularBuffer(self._num_traces)
        self._trace_buffer: CircularBuffer[List[float]] = CircularBuffer(self._num_traces)

        self._new_delays: List[float] = []
        self._new_traces: List[List[float]] = []
        self._ax: Optional[Axes] = None

    def run(self) -> None:
        if self.clk_in.is_edge:
            self._new_delays.append(self.clk_in.frac_dly)
            self._new_traces.append([])

        for i in range(len(self._new_traces) - 1, -1, -1):
            trace = self._new_traces[i]
            trace.append(float(self.data_in))
            if len(trace) == self._num_samples_per_trace:
                self._trace_buffer.add(trace)
                self._delay_buffer.add(self._new_delays[i])
                del self._new_traces[i]
                del self._new_delays[i]

    def clear(self) -> None:
        """Clear captured and in-flight eye traces."""
        self._delay_buffer = CircularBuffer(self._num_traces)
        self._trace_buffer = CircularBuffer(self._num_traces)
        self._new_delays = []
        self._new_traces = []

    def configure_timing(
        self,
        num_samples_per_trace: int,
        sample_rate_hz: float | None = None,
        trace_span_ui: float | None = None,
    ) -> None:
        n_samp = max(1, int(num_samples_per_trace))
        if n_samp != self._num_samples_per_trace:
            self._num_samples_per_trace = n_samp
            self._delay_buffer = CircularBuffer(self._num_traces)
            self._trace_buffer = CircularBuffer(self._num_traces)
            self._new_delays = []
            self._new_traces = []
        if sample_rate_hz is not None and float(sample_rate_hz) > 0:
            self._sample_rate_hz = float(sample_rate_hz)
        if trace_span_ui is not None and float(trace_span_ui) > 0:
            self._trace_span_ui = float(trace_span_ui)

    def _build_interpolated_traces(
        self, interp_step: float
    ) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
        traces = self._trace_buffer.get_data()
        delays = self._delay_buffer.get_data()
        # Use one trace-span exactly (avoid plotting beyond configured UI span).
        x_grid = np.arange(0.0, float(self._num_samples_per_trace) + float(interp_step), float(interp_step))
        interp_traces: list[npt.NDArray[np.float64]] = []
        for trace, delay in zip(traces, delays):
            indices = np.arange(len(trace), dtype=np.float64) + float(delay)
            y = np.interp(x_grid, indices, np.asarray(trace, dtype=np.float64))
            interp_traces.append(y)
        return x_grid, interp_traces

    def _x_axis_transform(
        self,
        x_grid_sample: npt.NDArray[np.float64],
        x_unit: str,
    ) -> tuple[npt.NDArray[np.float64], float, str]:
        unit = str(x_unit).strip().lower()
        if unit in {"sample", "samples", "sample_index"}:
            return np.asarray(x_grid_sample, dtype=np.float64), 1.0, "Sample Index"
        if unit == "ui":
            scale = float(self._trace_span_ui) / float(max(1, self._num_samples_per_trace))
            return np.asarray(x_grid_sample * scale, dtype=np.float64), scale, "Time (UI)"
        if unit in {"sec", "second", "seconds"}:
            unit = "s"
        elif unit in {"millisecond", "milliseconds"}:
            unit = "ms"
        elif unit in {"microsecond", "microseconds"}:
            unit = "us"
        elif unit in {"nanosecond", "nanoseconds"}:
            unit = "ns"
        elif unit in {"picosecond", "picoseconds"}:
            unit = "ps"
        time_scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}
        if unit in time_scale:
            if self._sample_rate_hz <= 0:
                raise ValueError("sample_rate_hz must be >0 for time-axis units.")
            scale = float(time_scale[unit]) / float(self._sample_rate_hz)
            return np.asarray(x_grid_sample * scale, dtype=np.float64), scale, f"Time ({unit})"
        raise ValueError(
            "Unsupported x_unit. Use one of: sample, ui, s/second, ms, us, ns, ps."
        )

    @staticmethod
    def _scale_mask_x(
        geometry: tuple[float, float, float, float],
        x_scale: float,
    ) -> tuple[float, float, float, float]:
        x_left, x_right, y_bot, y_top = geometry
        return float(x_left * x_scale), float(x_right * x_scale), float(y_bot), float(y_top)

    @staticmethod
    def _circular_delta(x: float, ref: float, period: float) -> float:
        return ((x - ref + period / 2.0) % period) - period / 2.0

    @staticmethod
    def _circular_unwrap(values: list[float], ref: float, period: float) -> np.ndarray:
        if len(values) == 0:
            return np.array([], dtype=np.float64)
        return np.asarray([ref + EyeMonitor._circular_delta(v, ref, period) for v in values], dtype=np.float64)

    @staticmethod
    def _extract_crossings(
        x: np.ndarray,
        y: np.ndarray,
        threshold: float,
    ) -> list[tuple[float, float]]:
        """Return threshold crossings as (x_cross, slope)."""
        sign = y - threshold
        crossings: list[tuple[float, float]] = []
        for i in range(len(sign) - 1):
            s0 = sign[i]
            s1 = sign[i + 1]
            if s0 == 0:
                x_cross = x[i]
            elif s1 == 0:
                x_cross = x[i + 1]
            elif s0 * s1 < 0:
                frac = -s0 / (s1 - s0)
                x_cross = x[i] + frac * (x[i + 1] - x[i])
            else:
                continue
            slope = y[i + 1] - y[i]
            crossings.append((float(x_cross), float(slope)))
        return crossings

    @staticmethod
    def _extract_crossing_x(
        crossings: list[tuple[float, float]],
        center_x: float,
    ) -> tuple[Optional[float], Optional[float]]:
        left_crosses = [x for x, _ in crossings if x < center_x]
        right_crosses = [x for x, _ in crossings if x > center_x]
        left = max(left_crosses) if left_crosses else None
        right = min(right_crosses) if right_crosses else None
        return left, right

    @staticmethod
    def _ber_from_sigma(sigma: float | None) -> Optional[float]:
        if sigma is None:
            return None
        s = float(sigma)
        if not np.isfinite(s):
            return None
        return float(norm.sf(s))

    @staticmethod
    def _sigma_from_ber(ber: float | None) -> Optional[float]:
        if ber is None:
            return None
        b = float(ber)
        if not np.isfinite(b):
            return None
        b = min(max(b, 1e-300), 0.5 - 1e-15)
        return float(norm.isf(b))

    @classmethod
    def _trace_ber_mapping(cls, n_traces: int) -> dict[str, Any]:
        n = int(max(0, n_traces))
        if n <= 0:
            return {
                "ber_floor_estimate": None,
                "sigma_limit_estimate": None,
                "sigma_to_ber": {},
                "sigma_to_min_traces": {},
            }
        ber_floor = 1.0 / float(n)
        sigma_limit = cls._sigma_from_ber(ber_floor)
        sigma_points = [1, 2, 3, 4, 5, 6, 7]
        sigma_to_ber = {str(s): float(cls._ber_from_sigma(float(s)) or 0.0) for s in sigma_points}
        sigma_to_min_traces = {
            str(s): int(np.ceil(1.0 / max(sigma_to_ber[str(s)], 1e-300)))
            for s in sigma_points
        }
        return {
            "ber_floor_estimate": float(ber_floor),
            "sigma_limit_estimate": sigma_limit,
            "sigma_to_ber": sigma_to_ber,
            "sigma_to_min_traces": sigma_to_min_traces,
        }

    @classmethod
    def _attach_trace_ber_mapping(cls, metrics: dict[str, Any], n_traces: int) -> dict[str, Any]:
        out = dict(metrics)
        out.update(cls._trace_ber_mapping(n_traces))
        return out

    @staticmethod
    def _resolve_mask_sigma(metrics: dict[str, Any], requested_sigma: float) -> tuple[float, bool]:
        req = float(max(0.0, requested_sigma))
        sigma_limit = metrics.get("sigma_limit_estimate")
        if sigma_limit is None:
            return req, False
        lim = float(sigma_limit)
        if not np.isfinite(lim):
            return req, False
        if req > lim:
            return lim, True
        return req, False

    def get_eye_metrics(self, interp_step: float = 0.1) -> dict[str, Any]:
        x_grid, interp_traces = self._build_interpolated_traces(interp_step=interp_step)
        if len(interp_traces) == 0:
            return self._attach_trace_ber_mapping({
                "threshold": None,
                "x_center": self._num_samples_per_trace / 2.0,
                "upper_amp_mean": None,
                "upper_amp_std": None,
                "upper_amp_min": None,
                "lower_amp_mean": None,
                "lower_amp_std": None,
                "lower_amp_max": None,
                "left_edge_mean": None,
                "left_edge_std": None,
                "left_edge_inner": None,
                "right_edge_mean": None,
                "right_edge_std": None,
                "right_edge_inner": None,
                "left_margin_mean": None,
                "left_margin_std": None,
                "right_margin_mean": None,
                "right_margin_std": None,
                "rising_transition_mean": None,
                "rising_transition_std": None,
                "falling_transition_mean": None,
                "falling_transition_std": None,
                "eye_height_mean": None,
                "eye_height_std": None,
                "eye_width_mean": None,
                "eye_width_std": None,
                "n_traces": 0,
            }, 0)

        y_all = np.vstack(interp_traces)

        rough_threshold = float(np.median(y_all))
        above = y_all >= rough_threshold
        below = ~above
        n_above = above.sum(axis=0)
        n_below = below.sum(axis=0)
        both_present = (n_above > 0) & (n_below > 0)
        inner_opening = np.full(y_all.shape[1], -np.inf)
        for col in np.where(both_present)[0]:
            col_vals = y_all[:, col]
            upper_min_col = float(np.min(col_vals[col_vals >= rough_threshold]))
            lower_max_col = float(np.max(col_vals[col_vals < rough_threshold]))
            inner_opening[col] = upper_min_col - lower_max_col

        if np.any(inner_opening > 0):
            best_col = int(np.argmax(inner_opening))
            x_center_ref = float(x_grid[best_col])
        else:
            x_center_ref = self._num_samples_per_trace / 2.0

        center_idx = int(np.argmin(np.abs(x_grid - x_center_ref)))
        center_vals = y_all[:, center_idx]

        threshold = float(np.median(center_vals))
        high_vals = center_vals[center_vals >= threshold]
        low_vals = center_vals[center_vals < threshold]
        if high_vals.size == 0 or low_vals.size == 0:
            threshold = float(np.mean(center_vals))
            high_vals = center_vals[center_vals >= threshold]
            low_vals = center_vals[center_vals < threshold]
        if high_vals.size == 0 or low_vals.size == 0:
            return self._attach_trace_ber_mapping({
                "threshold": threshold,
                "x_center": float(x_center_ref),
                "upper_amp_mean": None,
                "upper_amp_std": None,
                "upper_amp_min": None,
                "lower_amp_mean": None,
                "lower_amp_std": None,
                "lower_amp_max": None,
                "left_edge_mean": None,
                "left_edge_std": None,
                "left_edge_inner": None,
                "right_edge_mean": None,
                "right_edge_std": None,
                "right_edge_inner": None,
                "left_margin_mean": None,
                "left_margin_std": None,
                "right_margin_mean": None,
                "right_margin_std": None,
                "rising_transition_mean": None,
                "rising_transition_std": None,
                "falling_transition_mean": None,
                "falling_transition_std": None,
                "eye_height_mean": None,
                "eye_height_std": None,
                "eye_width_mean": None,
                "eye_width_std": None,
                "n_traces": len(interp_traces),
            }, len(interp_traces))

        upper_amp_mean = float(np.mean(high_vals))
        upper_amp_std = float(np.std(high_vals))
        upper_amp_min = float(np.min(high_vals))
        lower_amp_mean = float(np.mean(low_vals))
        lower_amp_std = float(np.std(low_vals))
        lower_amp_max = float(np.max(low_vals))
        eye_height_mean = upper_amp_mean - lower_amp_mean
        eye_height_std = float(np.sqrt(upper_amp_std**2 + lower_amp_std**2))

        all_crossings = [self._extract_crossings(x_grid, y, threshold) for y in interp_traces]
        rise_left: list[float] = []
        rise_right: list[float] = []
        fall_left: list[float] = []
        fall_right: list[float] = []
        for crossings in all_crossings:
            rising = [x for x, slope in crossings if slope > 0]
            falling = [x for x, slope in crossings if slope < 0]

            rising_l = [x for x in rising if x < x_center_ref]
            rising_r = [x for x in rising if x > x_center_ref]
            falling_l = [x for x in falling if x < x_center_ref]
            falling_r = [x for x in falling if x > x_center_ref]

            if rising_l:
                rise_left.append(max(rising_l))
            if rising_r:
                rise_right.append(min(rising_r))
            if falling_l:
                fall_left.append(max(falling_l))
            if falling_r:
                fall_right.append(min(falling_r))

        def _mean_std(vals: list[float]) -> tuple[Optional[float], Optional[float]]:
            if len(vals) == 0:
                return None, None
            arr = np.asarray(vals, dtype=np.float64)
            return float(np.mean(arr)), float(np.std(arr))

        rl_mu, rl_std = _mean_std(rise_left)
        rr_mu, rr_std = _mean_std(rise_right)
        fl_mu, fl_std = _mean_std(fall_left)
        fr_mu, fr_std = _mean_std(fall_right)

        candidates: list[dict[str, float | str]] = []
        # Candidate A: rising at left edge, falling at right edge
        if rl_mu is not None and fr_mu is not None and fr_mu > rl_mu:
            c = 0.5 * (rl_mu + fr_mu)
            candidates.append(
                {
                    "name": "A",
                    "center": c,
                    "left": rl_mu,
                    "right": fr_mu,
                    "rise_mu": rl_mu,
                    "rise_std": float(rl_std or 0.0),
                    "fall_mu": fr_mu,
                    "fall_std": float(fr_std or 0.0),
                }
            )
        # Candidate B: falling at left edge, rising at right edge
        if fl_mu is not None and rr_mu is not None and rr_mu > fl_mu:
            c = 0.5 * (fl_mu + rr_mu)
            candidates.append(
                {
                    "name": "B",
                    "center": c,
                    "left": fl_mu,
                    "right": rr_mu,
                    "rise_mu": rr_mu,
                    "rise_std": float(rr_std or 0.0),
                    "fall_mu": fl_mu,
                    "fall_std": float(fl_std or 0.0),
                }
            )

        if candidates:
            best = min(candidates, key=lambda d: abs(float(d["center"]) - float(x_center_ref)))
            x_center = float(best["center"])
            rising_transition_mean = float(best["rise_mu"])
            rising_transition_std = float(best["rise_std"])
            falling_transition_mean = float(best["fall_mu"])
            falling_transition_std = float(best["fall_std"])
        else:
            all_rising = [x for crossings in all_crossings for x, slope in crossings if slope > 0]
            all_falling = [x for crossings in all_crossings for x, slope in crossings if slope < 0]
            r_mu, r_std = _mean_std(all_rising)
            f_mu, f_std = _mean_std(all_falling)
            rising_transition_mean = r_mu
            rising_transition_std = r_std
            falling_transition_mean = f_mu
            falling_transition_std = f_std
            if r_mu is not None and f_mu is not None:
                x_center = 0.5 * (r_mu + f_mu)
            else:
                x_center = float(x_center_ref)

        def _edge_stats(center_val: float):
            left_edges_local: list[float] = []
            right_edges_local: list[float] = []
            for crossings in all_crossings:
                left, right = self._extract_crossing_x(crossings, center_val)
                if left is not None:
                    left_edges_local.append(left)
                if right is not None:
                    right_edges_local.append(right)

            left_edge_mean_local = float(np.mean(left_edges_local)) if left_edges_local else None
            left_edge_std_local = float(np.std(left_edges_local)) if left_edges_local else None
            left_edge_inner_local = float(np.max(left_edges_local)) if left_edges_local else None
            right_edge_mean_local = float(np.mean(right_edges_local)) if right_edges_local else None
            right_edge_std_local = float(np.std(right_edges_local)) if right_edges_local else None
            right_edge_inner_local = float(np.min(right_edges_local)) if right_edges_local else None

            left_margins_local = [center_val - left for left in left_edges_local]
            right_margins_local = [right - center_val for right in right_edges_local]
            left_margin_mean_local = float(np.mean(left_margins_local)) if left_margins_local else None
            left_margin_std_local = float(np.std(left_margins_local)) if left_margins_local else None
            right_margin_mean_local = float(np.mean(right_margins_local)) if right_margins_local else None
            right_margin_std_local = float(np.std(right_margins_local)) if right_margins_local else None

            return (
                left_edge_mean_local,
                left_edge_std_local,
                left_edge_inner_local,
                right_edge_mean_local,
                right_edge_std_local,
                right_edge_inner_local,
                left_margin_mean_local,
                left_margin_std_local,
                right_margin_mean_local,
                right_margin_std_local,
            )

        (
            left_edge_mean,
            left_edge_std,
            left_edge_inner,
            right_edge_mean,
            right_edge_std,
            right_edge_inner,
            left_margin_mean,
            left_margin_std,
            right_margin_mean,
            right_margin_std,
        ) = _edge_stats(x_center)

        # Final center definition: midpoint of reported left/right edge means.
        if left_edge_mean is not None and right_edge_mean is not None:
            x_center = 0.5 * (left_edge_mean + right_edge_mean)
            (
                left_edge_mean,
                left_edge_std,
                left_edge_inner,
                right_edge_mean,
                right_edge_std,
                right_edge_inner,
                left_margin_mean,
                left_margin_std,
                right_margin_mean,
                right_margin_std,
            ) = _edge_stats(x_center)
            if left_edge_mean is not None and right_edge_mean is not None:
                x_center = 0.5 * (left_edge_mean + right_edge_mean)

        if left_margin_mean is not None and right_margin_mean is not None:
            eye_width_mean = left_margin_mean + right_margin_mean
        else:
            eye_width_mean = None
        if left_margin_std is not None and right_margin_std is not None:
            eye_width_std = float(np.sqrt(left_margin_std**2 + right_margin_std**2))
        else:
            eye_width_std = None

        return self._attach_trace_ber_mapping({
            "threshold": threshold,
            "x_center": float(x_center),
            "upper_amp_mean": upper_amp_mean,
            "upper_amp_std": upper_amp_std,
            "upper_amp_min": upper_amp_min,
            "lower_amp_mean": lower_amp_mean,
            "lower_amp_std": lower_amp_std,
            "lower_amp_max": lower_amp_max,
            "left_edge_mean": left_edge_mean,
            "left_edge_std": left_edge_std,
            "left_edge_inner": left_edge_inner,
            "right_edge_mean": right_edge_mean,
            "right_edge_std": right_edge_std,
            "right_edge_inner": right_edge_inner,
            "left_margin_mean": left_margin_mean,
            "left_margin_std": left_margin_std,
            "right_margin_mean": right_margin_mean,
            "right_margin_std": right_margin_std,
            "rising_transition_mean": rising_transition_mean,
            "rising_transition_std": rising_transition_std,
            "falling_transition_mean": falling_transition_mean,
            "falling_transition_std": falling_transition_std,
            "eye_height_mean": eye_height_mean,
            "eye_height_std": eye_height_std,
            "eye_width_mean": eye_width_mean,
            "eye_width_std": eye_width_std,
            "n_traces": len(interp_traces),
        }, len(interp_traces))

    @staticmethod
    def _mask_geometry(
        metrics: dict[str, Any], mask_sigma: float
    ) -> Optional[tuple[float, float, float, float]]:
        upper_mu = metrics.get("upper_amp_mean")
        upper_std = metrics.get("upper_amp_std")
        lower_mu = metrics.get("lower_amp_mean")
        lower_std = metrics.get("lower_amp_std")
        left_margin_mu = metrics.get("left_margin_mean")
        left_margin_std = metrics.get("left_margin_std")
        right_margin_mu = metrics.get("right_margin_mean")
        right_margin_std = metrics.get("right_margin_std")
        x_center = metrics.get("x_center")
        if (
            upper_mu is None
            or upper_std is None
            or lower_mu is None
            or lower_std is None
            or left_margin_mu is None
            or left_margin_std is None
            or right_margin_mu is None
            or right_margin_std is None
            or x_center is None
        ):
            return None

        y_top = float(upper_mu) - float(mask_sigma) * float(upper_std)
        y_bot = float(lower_mu) + float(mask_sigma) * float(lower_std)
        half_left = float(left_margin_mu) - float(mask_sigma) * float(left_margin_std)
        half_right = float(right_margin_mu) - float(mask_sigma) * float(right_margin_std)
        half_w = min(half_left, half_right)
        x_left = float(x_center) - half_w
        x_right = float(x_center) + half_w

        if not (x_right > x_left and y_top > y_bot):
            return None
        return x_left, x_right, y_bot, y_top

    @staticmethod
    def _internal_open_geometry(metrics: dict[str, Any]) -> Optional[tuple[float, float, float, float]]:
        """Return internal open-eye geometry from worst-case inner trace boundaries.

        Uses the innermost crossing/amplitude across all captured traces:
        left  = max of per-trace left edges  (rightmost left boundary)
        right = min of per-trace right edges  (leftmost right boundary)
        lower = max of per-trace lower amps   (highest floor)
        upper = min of per-trace upper amps   (lowest ceiling)
        Falls back to mean values when inner-boundary fields are unavailable.
        """
        def _pick(primary_key: str, fallback_key: str):
            v = metrics.get(primary_key)
            if v is not None:
                return v
            return metrics.get(fallback_key)

        left = _pick("left_edge_inner", "left_edge_mean")
        right = _pick("right_edge_inner", "right_edge_mean")
        lower = _pick("lower_amp_max", "lower_amp_mean")
        upper = _pick("upper_amp_min", "upper_amp_mean")
        if left is None or right is None or lower is None or upper is None:
            return None
        x_left = float(left)
        x_right = float(right)
        y_bot = float(lower)
        y_top = float(upper)
        if not (x_right > x_left and y_top > y_bot):
            return None
        return x_left, x_right, y_bot, y_top

    def _compute_inner_eye_boundary(
        self,
        x_grid: npt.NDArray[np.float64],
        interp_traces: list[npt.NDArray[np.float64]],
    ) -> Optional[tuple[float, float, float, float]]:
        """Compute open-eye marker geometry from the widest opening within 1.5 UI.

        Eye-width selection:
        - Search eye-center candidates inside the central 1.5 UI window.
        - For each candidate, width is the inner opening from threshold crossings:
          ``x_right_inner - x_left_inner`` where
          ``x_left_inner = max(left_cross_i)`` and
          ``x_right_inner = min(right_cross_i)`` across traces.
        - Choose the candidate with the maximum positive width.

        Eye-height definition at the chosen center:
        - ``upper_min_amplitude = min(values of class "1")``
        - ``lower_max_amplitude = max(values of class "0")``
        - ``eye_height = upper_min_amplitude - lower_max_amplitude``
        where classing uses the global-median threshold.
        """
        if len(interp_traces) < 2:
            return None
        y_all = np.vstack(interp_traces)
        n_traces, n_cols = y_all.shape
        if n_traces < 2 or n_cols < 2:
            return None

        x_min = float(x_grid[0])
        x_max = float(x_grid[-1])
        x_span = max(x_max - x_min, 1e-12)

        # Restrict search to the central 1.5 UI region within the trace span.
        span_ui = float(max(self._trace_span_ui, 1e-12))
        search_ratio = min(1.0, 1.5 / span_ui)
        search_span_x = x_span * search_ratio
        x_center_nom = 0.5 * (x_min + x_max)
        x_lo = x_center_nom - 0.5 * search_span_x
        x_hi = x_center_nom + 0.5 * search_span_x
        candidate_cols = np.where((x_grid >= x_lo) & (x_grid <= x_hi))[0]
        if candidate_cols.size == 0:
            candidate_cols = np.arange(n_cols, dtype=int)

        # Build a global 0/1 threshold from low/high rails.
        low_ref = float(np.percentile(y_all, 10.0))
        high_ref = float(np.percentile(y_all, 90.0))
        threshold = 0.5 * (low_ref + high_ref)
        if not np.isfinite(threshold):
            threshold = float(np.median(y_all))

        # Build per-column inner envelope using the fixed 0/1 threshold.
        y_top_col = np.full(n_cols, -np.inf, dtype=np.float64)
        y_bot_col = np.full(n_cols, np.inf, dtype=np.float64)
        h_col = np.full(n_cols, -np.inf, dtype=np.float64)
        for ci in candidate_cols:
            col = y_all[:, int(ci)]
            hi = col[col >= threshold]
            lo = col[col < threshold]
            if hi.size == 0 or lo.size == 0:
                continue
            y_top = float(np.min(hi))
            y_bot = float(np.max(lo))
            h = y_top - y_bot
            if h > 0.0:
                y_top_col[int(ci)] = y_top
                y_bot_col[int(ci)] = y_bot
                h_col[int(ci)] = h

        valid_cols = candidate_cols[np.isfinite(h_col[candidate_cols]) & (h_col[candidate_cols] > 0.0)]
        if valid_cols.size == 0:
            return None
        valid_h = h_col[valid_cols]
        max_h = float(np.max(valid_h))
        if not np.isfinite(max_h) or max_h <= 0.0:
            return None

        # Choose a stable eye core (avoid stitching adjacent eyes through tiny residual openings).
        core_thresh = max(1e-12, 0.35 * max_h)
        core_mask = np.isfinite(h_col) & (h_col >= core_thresh)
        core_mask &= np.isin(np.arange(n_cols), candidate_cols)
        if not np.any(core_mask):
            core_mask = np.isfinite(h_col) & (h_col > 0.0) & np.isin(np.arange(n_cols), candidate_cols)
        core_idx = np.where(core_mask)[0]
        if core_idx.size == 0:
            return None
        runs: list[tuple[int, int]] = []
        run_start = int(core_idx[0])
        prev = int(core_idx[0])
        for k in core_idx[1:]:
            kk = int(k)
            if kk == prev + 1:
                prev = kk
                continue
            runs.append((run_start, prev))
            run_start = kk
            prev = kk
        runs.append((run_start, prev))
        best_run = max(
            runs,
            key=lambda r: (
                float(x_grid[r[1]] - x_grid[r[0]]),
                float(-abs(0.5 * (x_grid[r[0]] + x_grid[r[1]]) - x_center_nom)),
            ),
        )
        core_run = np.arange(int(best_run[0]), int(best_run[1]) + 1, dtype=int)
        ci = int(core_run[np.argmax(h_col[core_run])])

        # Eye width = local open span around selected core, bounded by low opening threshold.
        candidate_lo = int(np.min(candidate_cols))
        candidate_hi = int(np.max(candidate_cols))
        edge_thresh = max(1e-12, 0.05 * max_h)
        open_mask = np.isfinite(h_col) & (h_col > edge_thresh)

        i0 = ci
        while i0 > candidate_lo and bool(open_mask[i0 - 1]):
            i0 -= 1
        i1 = ci
        while i1 < candidate_hi and bool(open_mask[i1 + 1]):
            i1 += 1

        x_left = float(x_grid[i0])
        if i0 > candidate_lo and np.isfinite(h_col[i0 - 1]) and np.isfinite(h_col[i0]):
            h0 = float(h_col[i0 - 1])
            h1 = float(h_col[i0])
            if h0 <= edge_thresh < h1 and h1 != h0:
                x0 = float(x_grid[i0 - 1])
                x1 = float(x_grid[i0])
                x_left = float(x0 + (edge_thresh - h0) * (x1 - x0) / (h1 - h0))

        x_right = float(x_grid[i1])
        if i1 < candidate_hi and np.isfinite(h_col[i1]) and np.isfinite(h_col[i1 + 1]):
            h0 = float(h_col[i1])
            h1 = float(h_col[i1 + 1])
            if h0 > edge_thresh >= h1 and h1 != h0:
                x0 = float(x_grid[i1])
                x1 = float(x_grid[i1 + 1])
                x_right = float(x0 + (edge_thresh - h0) * (x1 - x0) / (h1 - h0))

        if x_right <= x_left:
            return None

        # Eye height at selected center: upper_min("1") - lower_max("0").
        y_top = float(y_top_col[ci])
        y_bot = float(y_bot_col[ci])
        if not (np.isfinite(y_top) and np.isfinite(y_bot) and y_top > y_bot):
            return None
        return x_left, x_right, y_bot, y_top

    @staticmethod
    def _draw_internal_open_markers_matplotlib(
        ax: Axes,
        geometry: tuple[float, float, float, float],
        shape: str = "diamond",
        color: str = "darkorange",
    ) -> None:
        """Draw internal open-eye boundary on a matplotlib axis.

        Parameters
        ----------
        shape : str
            ``"diamond"`` draws a diamond mask outline,
            ``"rectangle"`` draws a rectangular mask outline,
            ``"arrows"`` draws the legacy double-arrow annotation style.
        """
        from matplotlib.patches import Polygon as MplPolygon
        x_left, x_right, y_bot, y_top = geometry
        width = x_right - x_left
        height = y_top - y_bot
        if width <= 0 or height <= 0:
            return

        x_center = 0.5 * (x_left + x_right)
        y_center = 0.5 * (y_bot + y_top)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_span = max(float(x_max) - float(x_min), 1e-12)
        y_span = max(float(y_max) - float(y_min), 1e-12)
        y_off = 0.02 * y_span
        x_off = 0.02 * x_span

        if shape == "diamond":
            verts = [
                (x_center, y_top),
                (x_right, y_center),
                (x_center, y_bot),
                (x_left, y_center),
                (x_center, y_top),
            ]
            patch = MplPolygon(
                verts, closed=True, fill=False,
                edgecolor=color, linewidth=1.8, alpha=0.9, linestyle="-",
            )
            ax.add_patch(patch)
        elif shape == "rectangle":
            ax.add_patch(plt.Rectangle(
                (x_left, y_bot), width, height,
                fill=False, edgecolor=color, linewidth=1.8, alpha=0.9, linestyle="-",
            ))
        else:
            ax.annotate(
                "",
                xy=(x_left, y_center),
                xytext=(x_right, y_center),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.3, alpha=0.9),
            )
            ax.annotate(
                "",
                xy=(x_center, y_bot),
                xytext=(x_center, y_top),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.3, alpha=0.9),
            )

        ax.plot([x_center], [y_center], marker="x", color=color, markersize=6, markeredgewidth=1.4)
        ax.text(
            x_center,
            y_bot - y_off,
            f"Open Eye Width: {width:.3f}",
            color=color,
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor=color, alpha=0.85, boxstyle="round,pad=0.2"),
        )
        ax.text(
            x_center + x_off,
            y_top + y_off,
            f"Open Eye Height: {height:.3f}",
            color=color,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor=color, alpha=0.85, boxstyle="round,pad=0.2"),
        )

    @staticmethod
    def _draw_internal_open_markers_plotly(
        fig,
        geometry: tuple[float, float, float, float],
        shape: str = "diamond",
        color: str = "darkorange",
    ) -> None:
        """Draw internal open-eye boundary on a Plotly figure.

        Parameters
        ----------
        shape : str
            ``"diamond"``, ``"rectangle"``, or ``"arrows"`` (legacy).
        """
        x_left, x_right, y_bot, y_top = geometry
        width = x_right - x_left
        height = y_top - y_bot
        if width <= 0 or height <= 0:
            return

        x_center = 0.5 * (x_left + x_right)
        y_center = 0.5 * (y_bot + y_top)

        if shape == "diamond":
            fig.add_shape(
                type="path",
                path=(
                    f"M {x_center},{y_top} "
                    f"L {x_right},{y_center} "
                    f"L {x_center},{y_bot} "
                    f"L {x_left},{y_center} Z"
                ),
                line=dict(color=color, width=1.8),
                fillcolor="rgba(255,165,0,0.08)",
            )
        elif shape == "rectangle":
            fig.add_shape(
                type="rect",
                x0=x_left, y0=y_bot, x1=x_right, y1=y_top,
                line=dict(color=color, width=1.8),
                fillcolor="rgba(255,165,0,0.08)",
            )
        else:
            for pt_x, pt_y, ax_x, ax_y in [
                (x_left, y_center, x_right, y_center),
                (x_right, y_center, x_left, y_center),
                (x_center, y_bot, x_center, y_top),
                (x_center, y_top, x_center, y_bot),
            ]:
                fig.add_annotation(
                    x=pt_x, y=pt_y, ax=ax_x, ay=ax_y,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=3, arrowsize=1,
                    arrowwidth=1.4, arrowcolor=color,
                )

        fig.add_annotation(
            x=x_center, y=y_center, yshift=-14,
            text=f"Open Eye Width: {width:.3f}",
            showarrow=False, font=dict(color=color),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=color, borderwidth=1,
        )
        fig.add_annotation(
            x=x_center, y=y_top, xshift=24, yshift=10,
            text=f"Open Eye Height: {height:.3f}",
            showarrow=False, font=dict(color=color),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=color, borderwidth=1,
        )

    @staticmethod
    def _draw_mask_matplotlib(
        ax: Axes,
        geometry: tuple[float, float, float, float],
        mask_type: str,
        color: str = "crimson",
    ) -> None:
        x_left, x_right, y_bot, y_top = geometry
        mask_w = x_right - x_left
        mask_h = y_top - y_bot
        if mask_w <= 0 or mask_h <= 0:
            return
        x_center = (x_left + x_right) / 2.0
        y_center = (y_bot + y_top) / 2.0
        if mask_type == "rectangle":
            patch = mpatches.Rectangle(
                (x_left, y_bot),
                mask_w,
                mask_h,
                edgecolor=color,
                facecolor=(1, 0, 0, 0.05),
                linewidth=2,
            )
        elif mask_type == "diamond":
            diamond_vertices = [
                (x_center, y_center + mask_h / 2.0),
                (x_center + mask_w / 2.0, y_center),
                (x_center, y_center - mask_h / 2.0),
                (x_center - mask_w / 2.0, y_center),
            ]
            patch = mpatches.Polygon(
                diamond_vertices,
                closed=True,
                edgecolor=color,
                facecolor=(1, 0, 0, 0.05),
                linewidth=2,
            )
        else:
            return
        ax.add_patch(patch)
        if mask_type == "diamond":
            vx, vy = zip(*diamond_vertices)
            ax.plot(
                vx,
                vy,
                linestyle="None",
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.2,
                alpha=0.95,
            )

    @staticmethod
    def _label_mask_matplotlib(
        ax: Axes,
        geometry: tuple[float, float, float, float],
        color: str = "crimson",
        stat_line: Optional[str] = None,
    ) -> None:
        x_left, x_right, y_bot, y_top = geometry
        mask_w = x_right - x_left
        mask_h = y_top - y_bot
        if mask_w <= 0 or mask_h <= 0:
            return

        x_center = 0.5 * (x_left + x_right)
        y_center = 0.5 * (y_bot + y_top)
        y_min, y_max = ax.get_ylim()
        y_span = max(y_max - y_min, 1e-12)
        label_off = 0.03 * y_span

        ax.plot([x_left, x_right], [y_bot, y_bot], linestyle="--", color=color, linewidth=1.0, alpha=0.8)
        ax.plot([x_left, x_left], [y_bot, y_top], linestyle="--", color=color, linewidth=1.0, alpha=0.8)

        ax.text(
            x_center,
            y_bot - label_off,
            f"Eye Width: {mask_w:.3f}",
            color=color,
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor=color, alpha=0.8, boxstyle="round,pad=0.2"),
        )
        ax.text(
            x_left - 0.02 * (x_right - x_left),
            y_center,
            f"Eye Height: {mask_h:.3f}",
            color=color,
            ha="right",
            va="center",
            rotation=90,
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor=color, alpha=0.8, boxstyle="round,pad=0.2"),
        )
        if stat_line:
            ax.text(
                x_center,
                y_top + label_off,
                stat_line,
                color=color,
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor=color, alpha=0.8, boxstyle="round,pad=0.2"),
            )

    @staticmethod
    def _label_mask_plotly(
        fig,
        geometry: tuple[float, float, float, float],
        color: str = "crimson",
        stat_line: Optional[str] = None,
    ) -> None:
        x_left, x_right, y_bot, y_top = geometry
        mask_w = x_right - x_left
        mask_h = y_top - y_bot
        if mask_w <= 0 or mask_h <= 0:
            return
        x_center = 0.5 * (x_left + x_right)
        y_center = 0.5 * (y_bot + y_top)

        fig.add_shape(
            type="line",
            x0=x_left,
            y0=y_bot,
            x1=x_right,
            y1=y_bot,
            line=dict(color=color, width=1, dash="dash"),
        )
        fig.add_shape(
            type="line",
            x0=x_left,
            y0=y_bot,
            x1=x_left,
            y1=y_top,
            line=dict(color=color, width=1, dash="dash"),
        )
        fig.add_annotation(
            x=x_center,
            y=y_bot,
            yshift=-14,
            text=f"Eye Width: {mask_w:.3f}",
            showarrow=False,
            font=dict(color=color),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=color,
            borderwidth=1,
        )
        fig.add_annotation(
            x=x_left,
            y=y_center,
            xshift=-18,
            text=f"Eye Height: {mask_h:.3f}",
            textangle=-90,
            showarrow=False,
            font=dict(color=color),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=color,
            borderwidth=1,
        )
        if stat_line:
            fig.add_annotation(
                x=x_center,
                y=y_top,
                yshift=14,
                text=stat_line,
                showarrow=False,
                font=dict(color=color),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=color,
                borderwidth=1,
            )

    def plot(
        self,
        ax: Optional[Axes] = None,
        mask_type: Optional[str] = None,
        mask_sigma: float = 1.0,
        interp_step: float = 0.1,
        x_unit: str = "ui",
        show_internal_open_markers: bool = True,
        open_eye_shape: str = "diamond",
        return_metrics: bool = False,
    ):
        if ax is None:
            if self._ax is None:
                _, self._ax = plt.subplots()
            ax = self._ax

        ax.clear()
        ax.set_ylabel("Amplitude (V)")
        ax.grid(True, which="both", alpha=0.3)

        x_grid, interp_traces = self._build_interpolated_traces(interp_step=interp_step)
        x_plot, x_scale, x_label = self._x_axis_transform(x_grid, x_unit=x_unit)
        ax.set_xlabel(x_label)
        for trace_interp in interp_traces:
            ax.plot(x_plot, trace_interp, color=(0, 0, 1, 0.2), linewidth=1)
        if x_plot.size > 0:
            ax.set_xlim(float(np.min(x_plot)), float(np.max(x_plot)))

        metrics = self.get_eye_metrics(interp_step=interp_step)
        metrics["x_axis_unit"] = str(x_unit)
        metrics["x_scale_from_sample"] = float(x_scale)
        if metrics.get("x_center") is not None:
            metrics["x_center_in_unit"] = float(metrics["x_center"]) * float(x_scale)
        if bool(show_internal_open_markers):
            open_geom = self._compute_inner_eye_boundary(x_grid, interp_traces)
            if open_geom is not None:
                open_geom_plot = self._scale_mask_x(open_geom, x_scale=float(x_scale))
                self._draw_internal_open_markers_matplotlib(ax, open_geom_plot, shape=open_eye_shape)
                eye_center_sample = 0.5 * (open_geom[0] + open_geom[1])
                metrics["x_center"] = float(eye_center_sample)
                metrics["x_center_in_unit"] = float(eye_center_sample) * float(x_scale)
                metrics["internal_open_geometry"] = {
                    "x_left": float(open_geom_plot[0]),
                    "x_right": float(open_geom_plot[1]),
                    "y_bot": float(open_geom_plot[2]),
                    "y_top": float(open_geom_plot[3]),
                }
        if mask_type in {"rectangle", "diamond"}:
            effective_sigma, sigma_limited = self._resolve_mask_sigma(metrics, requested_sigma=float(mask_sigma))
            ber_eff = self._ber_from_sigma(effective_sigma)
            stat_line = f"sigma={effective_sigma:.2f}, BER~{float(ber_eff or 0.0):.2e}"
            if sigma_limited:
                stat_line += " (trace-limited)"
            geometry = self._mask_geometry(metrics, mask_sigma=float(effective_sigma))
            if geometry is not None:
                geometry_plot = self._scale_mask_x(geometry, x_scale=float(x_scale))
                self._draw_mask_matplotlib(ax, geometry_plot, mask_type=mask_type)
                self._label_mask_matplotlib(ax, geometry_plot, stat_line=stat_line)
            metrics["mask_sigma_requested"] = float(mask_sigma)
            metrics["mask_sigma_effective"] = float(effective_sigma)
            metrics["mask_sigma_trace_limited"] = bool(sigma_limited)
            metrics["mask_ber_effective"] = ber_eff

        if return_metrics:
            return ax, metrics
        return ax

    def plot_plotly(
        self,
        mask_type: Optional[str] = None,
        mask_sigma: float = 1.0,
        interp_step: float = 0.1,
        x_unit: str = "ui",
        show_internal_open_markers: bool = True,
        open_eye_shape: str = "diamond",
        show: bool = True,
    ):
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required for plot_plotly(); install with `pip install plotly`.") from exc

        x_grid, interp_traces = self._build_interpolated_traces(interp_step=interp_step)
        x_plot, x_scale, x_label = self._x_axis_transform(x_grid, x_unit=x_unit)
        fig = go.Figure()
        for trace_interp in interp_traces:
            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=trace_interp,
                    mode="lines",
                    line=dict(color="rgba(0, 0, 255, 0.2)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        metrics = self.get_eye_metrics(interp_step=interp_step)
        metrics["x_axis_unit"] = str(x_unit)
        metrics["x_scale_from_sample"] = float(x_scale)
        if metrics.get("x_center") is not None:
            metrics["x_center_in_unit"] = float(metrics["x_center"]) * float(x_scale)
        if bool(show_internal_open_markers):
            open_geom = self._compute_inner_eye_boundary(x_grid, interp_traces)
            if open_geom is not None:
                open_geom_plot = self._scale_mask_x(open_geom, x_scale=float(x_scale))
                self._draw_internal_open_markers_plotly(fig, open_geom_plot, shape=open_eye_shape)
                eye_center_sample = 0.5 * (open_geom[0] + open_geom[1])
                metrics["x_center"] = float(eye_center_sample)
                metrics["x_center_in_unit"] = float(eye_center_sample) * float(x_scale)
                metrics["internal_open_geometry"] = {
                    "x_left": float(open_geom_plot[0]),
                    "x_right": float(open_geom_plot[1]),
                    "y_bot": float(open_geom_plot[2]),
                    "y_top": float(open_geom_plot[3]),
                }
        if mask_type in {"rectangle", "diamond"}:
            effective_sigma, sigma_limited = self._resolve_mask_sigma(metrics, requested_sigma=float(mask_sigma))
            ber_eff = self._ber_from_sigma(effective_sigma)
            stat_line = f"sigma={effective_sigma:.2f}, BER~{float(ber_eff or 0.0):.2e}"
            if sigma_limited:
                stat_line += " (trace-limited)"
            geometry = self._mask_geometry(metrics, mask_sigma=float(effective_sigma))
            if geometry is not None:
                x0, x1, y0, y1 = self._scale_mask_x(geometry, x_scale=float(x_scale))
                x_center = (x0 + x1) / 2.0
                y_center = (y0 + y1) / 2.0
                if mask_type == "rectangle":
                    fig.add_shape(
                        type="rect",
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        line=dict(color="crimson", width=2),
                        fillcolor="rgba(220, 20, 60, 0.08)",
                    )
                else:
                    fig.add_shape(
                        type="path",
                        path=(
                            f"M {x_center},{y1} "
                            f"L {x1},{y_center} "
                            f"L {x_center},{y0} "
                            f"L {x0},{y_center} Z"
                        ),
                        line=dict(color="crimson", width=2),
                        fillcolor="rgba(220, 20, 60, 0.08)",
                    )
                geometry_plot = (x0, x1, y0, y1)
                self._label_mask_plotly(fig, geometry_plot, stat_line=stat_line)
            metrics["mask_sigma_requested"] = float(mask_sigma)
            metrics["mask_sigma_effective"] = float(effective_sigma)
            metrics["mask_sigma_trace_limited"] = bool(sigma_limited)
            metrics["mask_ber_effective"] = ber_eff

        fig.update_layout(
            title="Eye Diagram (Interactive)",
            xaxis_title=x_label,
            yaxis_title="Amplitude (V)",
            template="plotly_white",
        )
        if x_plot.size > 0:
            fig.update_xaxes(range=[float(np.min(x_plot)), float(np.max(x_plot))])
        if show:
            fig.show()
        return fig, metrics
