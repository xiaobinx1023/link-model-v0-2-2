from __future__ import annotations

from typing import Any, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from .circular_buffer import CircularBuffer
from .clock import Clock


class EyeMonitor:
    def __init__(self, num_traces: int = 1024, num_samples_per_trace: int = 16) -> None:
        self.clk_in = Clock()
        self.data_in = 0.0

        self._num_traces = int(num_traces)
        self._num_samples_per_trace = int(num_samples_per_trace)
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

    def _build_interpolated_traces(
        self, interp_step: float
    ) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
        traces = self._trace_buffer.get_data()
        delays = self._delay_buffer.get_data()
        x_grid = np.arange(0.0, self._num_samples_per_trace + 1.0, float(interp_step))
        interp_traces: list[npt.NDArray[np.float64]] = []
        for trace, delay in zip(traces, delays):
            indices = np.arange(len(trace), dtype=np.float64) + float(delay)
            y = np.interp(x_grid, indices, np.asarray(trace, dtype=np.float64))
            interp_traces.append(y)
        return x_grid, interp_traces

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

    def get_eye_metrics(self, interp_step: float = 0.1) -> dict[str, Any]:
        x_grid, interp_traces = self._build_interpolated_traces(interp_step=interp_step)
        if len(interp_traces) == 0:
            return {
                "threshold": None,
                "x_center": self._num_samples_per_trace / 2.0,
                "upper_amp_mean": None,
                "upper_amp_std": None,
                "lower_amp_mean": None,
                "lower_amp_std": None,
                "left_edge_mean": None,
                "left_edge_std": None,
                "right_edge_mean": None,
                "right_edge_std": None,
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
            }

        y_all = np.vstack(interp_traces)
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
            return {
                "threshold": threshold,
                "x_center": float(x_center_ref),
                "upper_amp_mean": None,
                "upper_amp_std": None,
                "lower_amp_mean": None,
                "lower_amp_std": None,
                "left_edge_mean": None,
                "left_edge_std": None,
                "right_edge_mean": None,
                "right_edge_std": None,
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
            }

        upper_amp_mean = float(np.mean(high_vals))
        upper_amp_std = float(np.std(high_vals))
        lower_amp_mean = float(np.mean(low_vals))
        lower_amp_std = float(np.std(low_vals))
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
            right_edge_mean_local = float(np.mean(right_edges_local)) if right_edges_local else None
            right_edge_std_local = float(np.std(right_edges_local)) if right_edges_local else None

            left_margins_local = [center_val - left for left in left_edges_local]
            right_margins_local = [right - center_val for right in right_edges_local]
            left_margin_mean_local = float(np.mean(left_margins_local)) if left_margins_local else None
            left_margin_std_local = float(np.std(left_margins_local)) if left_margins_local else None
            right_margin_mean_local = float(np.mean(right_margins_local)) if right_margins_local else None
            right_margin_std_local = float(np.std(right_margins_local)) if right_margins_local else None

            return (
                left_edge_mean_local,
                left_edge_std_local,
                right_edge_mean_local,
                right_edge_std_local,
                left_margin_mean_local,
                left_margin_std_local,
                right_margin_mean_local,
                right_margin_std_local,
            )

        (
            left_edge_mean,
            left_edge_std,
            right_edge_mean,
            right_edge_std,
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
                right_edge_mean,
                right_edge_std,
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

        return {
            "threshold": threshold,
            "x_center": float(x_center),
            "upper_amp_mean": upper_amp_mean,
            "upper_amp_std": upper_amp_std,
            "lower_amp_mean": lower_amp_mean,
            "lower_amp_std": lower_amp_std,
            "left_edge_mean": left_edge_mean,
            "left_edge_std": left_edge_std,
            "right_edge_mean": right_edge_mean,
            "right_edge_std": right_edge_std,
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
        }

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
            patch = mpatches.Polygon(
                [
                    (x_center, y_center + mask_h / 2.0),
                    (x_center + mask_w / 2.0, y_center),
                    (x_center, y_center - mask_h / 2.0),
                    (x_center - mask_w / 2.0, y_center),
                ],
                closed=True,
                edgecolor=color,
                facecolor=(1, 0, 0, 0.05),
                linewidth=2,
            )
        else:
            return
        ax.add_patch(patch)

    @staticmethod
    def _label_mask_matplotlib(
        ax: Axes,
        geometry: tuple[float, float, float, float],
        color: str = "crimson",
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

    @staticmethod
    def _label_mask_plotly(fig, geometry: tuple[float, float, float, float], color: str = "crimson") -> None:
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

    def plot(
        self,
        ax: Optional[Axes] = None,
        mask_type: Optional[str] = None,
        mask_sigma: float = 1.0,
        interp_step: float = 0.1,
        return_metrics: bool = False,
    ):
        if ax is None:
            if self._ax is None:
                _, self._ax = plt.subplots()
            ax = self._ax

        ax.clear()
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude (V)")
        ax.grid(True, which="both", alpha=0.3)

        x_grid, interp_traces = self._build_interpolated_traces(interp_step=interp_step)
        for trace_interp in interp_traces:
            ax.plot(x_grid, trace_interp, color=(0, 0, 1, 0.2), linewidth=1)

        metrics = self.get_eye_metrics(interp_step=interp_step)
        if mask_type in {"rectangle", "diamond"}:
            geometry = self._mask_geometry(metrics, mask_sigma=float(mask_sigma))
            if geometry is not None:
                self._draw_mask_matplotlib(ax, geometry, mask_type=mask_type)
                self._label_mask_matplotlib(ax, geometry)

        if return_metrics:
            return ax, metrics
        return ax

    def plot_plotly(
        self,
        mask_type: Optional[str] = None,
        mask_sigma: float = 1.0,
        interp_step: float = 0.1,
        show: bool = True,
    ):
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required for plot_plotly(); install with `pip install plotly`.") from exc

        x_grid, interp_traces = self._build_interpolated_traces(interp_step=interp_step)
        fig = go.Figure()
        for trace_interp in interp_traces:
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=trace_interp,
                    mode="lines",
                    line=dict(color="rgba(0, 0, 255, 0.2)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        metrics = self.get_eye_metrics(interp_step=interp_step)
        if mask_type in {"rectangle", "diamond"}:
            geometry = self._mask_geometry(metrics, mask_sigma=float(mask_sigma))
            if geometry is not None:
                x0, x1, y0, y1 = geometry
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
                self._label_mask_plotly(fig, geometry)

        fig.update_layout(
            title="Eye Diagram (Interactive)",
            xaxis_title="Sample Index",
            yaxis_title="Amplitude (V)",
            template="plotly_white",
        )
        if show:
            fig.show()
        return fig, metrics
