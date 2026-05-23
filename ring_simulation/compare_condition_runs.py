"""
Build comparison plots across STP conditions for a single config.

Outputs under saved_runs/<cfg_name>/:
- condition_compare.pdf
- condition_compare_phase_aligned_rate.pdf
- condition_compare_asymmetry_metrics.pdf
- condition_compare_training_curve.pdf
- condition_compare_correlogram.pdf
- condition_compare_positive_lag_weighted.pdf
- condition_compare_positive_lag_weighted_diff.pdf
- condition_compare_spontaneous_cplus_assoc_correlogram.pdf
- condition_compare_replay.pdf
- condition_compare_fisher_info.json

With --paper, outputs under saved_runs/<cfg_name>/paper/ are limited to:
- condition_compare.pdf
- condition_compare_asymmetry_metrics.pdf
- condition_compare_training_curve.pdf
- condition_compare_correlogram.pdf
- condition_compare_replay.pdf
- condition_compare_fisher_info.json
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ring_stp_exactgrad import (
    DEFAULT_SAVE_DIR,
    ReplayConfig,
    RingConfig,
    centered_offset_angles,
    correlogram_by_distance_trial_shuffle,
    correlogram_epsp_weighted_positive_lag,
    ensure_fit_result_asymmetry_history,
    evaluate_FI,
    get_fit_result,
    init_params,
    sample_spikes_trials,
    simulate_replay_activity,
    try_load_fit_result,
)
from run_from_cfg import (
    STP_TYPES,
    _apply_overrides,
    _load_json,
    _split_overrides,
    _stp_overrides,
    _unique_ordered,
)


STP_LABELS = {
    "with_stp": "associative STP",
    "pre_stp": "non-associative STP",
    "no_stp": "static",
}

PAPER_STP_LABELS = {
    "with_stp": "assoc. STP",
    "pre_stp": "non-assoc. STP",
    "no_stp": "static",
}

PAPER_STP_TICK_LABELS = {
    "with_stp": "assoc.\nSTP",
    "pre_stp": "non-assoc.\nSTP",
    "no_stp": "static",
}

ASYMMETRY_METRIC_SPECS: Tuple[Tuple[str, str], ...] = (
    ("peak_phase", "peak phase (rad)"),
    ("centroid_phase", "centroid (rad)"),
    ("skewness", "skewness"),
    ("area_index", "AI"),
    ("odd_ratio", "A_odd"),
)


@dataclass
class ConditionSummary:
    cfg: RingConfig
    stp_type: str
    label: str
    run_cfg_name: str
    fit_path: Path
    w0: np.ndarray
    U: np.ndarray
    phase: np.ndarray
    rate_init: np.ndarray
    rate_final: np.ndarray
    asymmetry_init: Dict[str, float]
    asymmetry_final: Dict[str, float]
    fi_curve: Optional[np.ndarray]
    rate_penalty_curve: Optional[np.ndarray]
    total_loss_curve: Optional[np.ndarray]
    fi_init_mean: float
    fi_init_se: float
    fi_final_mean: float
    fi_final_se: float


@dataclass
class CorrelogramCompareData:
    items: List[ConditionSummary]
    zeta: np.ndarray
    evoked_mats: List[np.ndarray]
    spont_mats: List[np.ndarray]
    evoked_init_mats: List[np.ndarray]
    spont_init_mats: List[np.ndarray]
    lags_evoked_list: List[np.ndarray]
    lags_spont_list: List[np.ndarray]
    lags_evoked_ref: np.ndarray
    lags_spont_ref: np.ndarray
    cplus_evoked: List[np.ndarray]
    cplus_spont: List[np.ndarray]
    cplus_evoked_diff: List[np.ndarray]
    cplus_spont_diff: List[np.ndarray]
    vmax_evoked: float
    vmax_spont: float
    heatmap_lag_limit_s: float
    assoc_idx: Optional[int]


def _condition_label(item: ConditionSummary, paper: bool = False, *, multiline: bool = False) -> str:
    if not paper:
        return item.label
    labels = PAPER_STP_TICK_LABELS if multiline else PAPER_STP_LABELS
    return labels.get(item.stp_type, item.label)


def _figure_dir(cfg_name: str, paper: bool = False) -> Path:
    base = Path(DEFAULT_SAVE_DIR) / cfg_name
    return base / "paper" if paper else base


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


PAPER_SKIP_OUTPUTS: Tuple[str, ...] = (
    "condition_compare_phase_aligned_rate.pdf",
    "condition_compare_positive_lag_weighted.pdf",
    "condition_compare_positive_lag_weighted_diff.pdf",
    "condition_compare_spontaneous_cplus_assoc_correlogram.pdf",
)


def _remove_skipped_paper_outputs(out_dir: Path) -> None:
    for name in PAPER_SKIP_OUTPUTS:
        path = out_dir / name
        if path.exists():
            path.unlink()


def _paper_rc_params() -> Dict[str, Any]:
    # Paper PDFs are often scaled down into a two-column manuscript, so source
    # text needs to be larger than the intended final printed size.
    return {
        "font.family": "sans-serif",
        "font.size": 18.0,
        "axes.titlelocation": "left",
        "axes.titlesize": 19.0,
        "axes.labelsize": 19.0,
        "xtick.labelsize": 17.0,
        "ytick.labelsize": 17.0,
        "legend.fontsize": 17.0,
        "figure.titlesize": 19.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.xmargin": 0.01,
        "axes.ymargin": 0.05,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": False,
        "legend.fancybox": False,
        "image.interpolation": "none",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def _savefig_kwargs(paper: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
    if paper:
        kwargs.update({"dpi": 300, "facecolor": "white"})
    return kwargs


def _paper_figsize(size: Tuple[float, float], paper: bool, max_width: float = 7.0) -> Tuple[float, float]:
    width, height = float(size[0]), float(size[1])
    if (not paper) or (width <= max_width):
        return (width, height)
    return (max_width, height)


def _with_paper_rc(func: Callable[..., Any]) -> Callable[..., Any]:
    sig = inspect.signature(func)

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any):
        try:
            bound = sig.bind_partial(*args, **kwargs)
            paper = bool(bound.arguments.get("paper", False))
        except Exception:
            paper = bool(kwargs.get("paper", False))
        if not paper:
            return func(*args, **kwargs)
        import matplotlib as mpl

        with mpl.rc_context(rc=_paper_rc_params()):
            return func(*args, **kwargs)

    return wrapped


def _condition_color(stp_type: str) -> str:
    if stp_type == "with_stp":
        return "tab:blue"
    if stp_type == "pre_stp":
        return "tab:orange"
    if stp_type == "no_stp":
        return "tab:green"
    return "tab:gray"


def _circular_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=float).copy()
    if window % 2 == 0:
        raise ValueError("window must be odd for symmetric smoothing")
    half = window // 2
    values_arr = np.asarray(values, dtype=float)
    values_wrap = np.concatenate([values_arr[-half:], values_arr, values_arr[:half]])
    mask = np.isfinite(values_wrap)
    values_filled = np.where(mask, values_wrap, 0.0)
    kernel = np.ones(window, dtype=float)
    numer = np.convolve(values_filled, kernel, mode="valid")
    denom = np.convolve(mask.astype(float), kernel, mode="valid")
    out = np.full(numer.shape, np.nan, dtype=float)
    np.divide(numer, denom, out=out, where=denom > 0.0)
    return out


def _stacked_mean(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if len(arrays) == 0:
        return np.array([], dtype=float)
    min_len = min(arr.size for arr in arrays)
    if min_len <= 0:
        return np.array([], dtype=float)
    mat = np.vstack([np.asarray(arr[:min_len], dtype=float) for arr in arrays])
    return np.nanmean(mat, axis=0)


def _mean_and_combined_se(means: np.ndarray, ses: np.ndarray) -> Tuple[float, float]:
    means = np.asarray(means, dtype=float)
    ses = np.asarray(ses, dtype=float)
    finite_mean = means[np.isfinite(means)]
    if finite_mean.size == 0:
        return float("nan"), float("nan")
    mean_val = float(np.mean(finite_mean))

    finite_se = ses[np.isfinite(ses)]
    if finite_se.size == 0:
        return mean_val, float("nan")
    combined_se = float(np.sqrt(np.sum(finite_se ** 2)) / max(finite_se.size, 1))
    return mean_val, combined_se


def _legend_outside(
    fig: plt.Figure,
    handles: Sequence[Any],
    labels: Sequence[str],
    paper: bool,
    *,
    anchor_x: float = 1.0,
    anchor_y: float = 1.0,
) -> None:
    if len(handles) == 0:
        return
    if paper:
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(anchor_x, anchor_y),
            borderaxespad=0.0,
            frameon=False,
        )


def _condition_summaries_ordered(summaries: Sequence[ConditionSummary]) -> List[ConditionSummary]:
    stp_order = {name: idx for idx, name in enumerate(STP_TYPES)}
    return sorted(summaries, key=lambda item: stp_order.get(item.stp_type, len(stp_order)))


def _as_float_curve(history: dict, key: str) -> Optional[np.ndarray]:
    val = history.get(key)
    if val is None:
        return None
    arr = np.asarray(val, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.ravel()


def _training_curves(cfg: RingConfig, history: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    fi_curve = _as_float_curve(history, "fi_rate")
    if fi_curve is None:
        J_curve = _as_float_curve(history, "J")
        if J_curve is not None:
            n_steps = int(np.round(cfg.T / cfg.dt))
            duration = float(n_steps) * float(cfg.dt)
            norm = float(cfg.N) * duration
            if norm > 0.0:
                fi_curve = J_curve / norm

    rate_penalty = _as_float_curve(history, "rate_penalty")
    total_loss = _as_float_curve(history, "total_loss")

    if rate_penalty is None or total_loss is None:
        rate_mean = _as_float_curve(history, "rate_mean")
        if rate_mean is None:
            rate_mean = _as_float_curve(history, "spike_rate")
        if rate_mean is not None and cfg.rate_target is not None:
            rate_error = rate_mean - float(cfg.rate_target)
            rate_penalty = 0.5 * float(cfg.rate_reg_lambda) * (rate_error ** 2)
            if fi_curve is not None and fi_curve.size == rate_penalty.size:
                total_loss = fi_curve - rate_penalty

    return fi_curve, rate_penalty, total_loss


def _safe_nanmax(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=float).ravel()
    if vals.size == 0 or not np.any(np.isfinite(vals)):
        return float("nan")
    return float(np.nanmax(vals))


def _empty_asymmetry_metrics() -> Dict[str, float]:
    return {key: float("nan") for key, _ in ASYMMETRY_METRIC_SPECS}


def _as_float_scalar(history: dict, key: str) -> float:
    val = history.get(key)
    if val is None:
        return float("nan")
    try:
        arr = np.asarray(val, dtype=float).ravel()
    except Exception:
        return float("nan")
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def _load_saved_asymmetry_from_history(
    history: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]:
    phase_init = _as_float_curve(history, "asymmetry_phase_init")
    rate_init = _as_float_curve(history, "asymmetry_rate_init")
    phase_final = _as_float_curve(history, "asymmetry_phase_final")
    rate_final = _as_float_curve(history, "asymmetry_rate_final")
    if phase_init is None or rate_init is None or phase_final is None or rate_final is None:
        raise RuntimeError("Missing saved asymmetry phase/rate arrays in fit history.")

    init_metrics = _empty_asymmetry_metrics()
    final_metrics = _empty_asymmetry_metrics()
    for metric_key, _ in ASYMMETRY_METRIC_SPECS:
        init_metrics[metric_key] = _as_float_scalar(history, f"asymmetry_{metric_key}_init")
        final_metrics[metric_key] = _as_float_scalar(history, f"asymmetry_{metric_key}_final")

    return phase_final, rate_init, rate_final, init_metrics, final_metrics


def _collect_asymmetry_metric_series(
    summaries: Sequence[ConditionSummary],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    init_series: Dict[str, List[float]] = {key: [] for key, _ in ASYMMETRY_METRIC_SPECS}
    final_series: Dict[str, List[float]] = {key: [] for key, _ in ASYMMETRY_METRIC_SPECS}
    for item in summaries:
        for metric_key, _ in ASYMMETRY_METRIC_SPECS:
            init_series[metric_key].append(float(item.asymmetry_init.get(metric_key, float("nan"))))
            final_series[metric_key].append(float(item.asymmetry_final.get(metric_key, float("nan"))))
    init_arr = {key: np.asarray(values, dtype=float) for key, values in init_series.items()}
    final_arr = {key: np.asarray(values, dtype=float) for key, values in final_series.items()}
    return init_arr, final_arr


def _load_or_fit_condition(
    cfg_path: Path,
    base_cfg: RingConfig,
    cfg_name: str,
    stp_type: str,
    reuse_only: bool,
    force_retrain: bool,
    phase_trials: int,
    fi_trials: int,
    seed_base: int,
) -> ConditionSummary:
    _ = phase_trials  # retained for CLI compatibility; asymmetry profiles are read from saved history.
    cfg = RingConfig(**asdict(base_cfg))
    _apply_overrides(cfg, _stp_overrides(stp_type), label=f"{cfg_path} ({stp_type})")
    run_cfg_name = str(Path(cfg_name) / stp_type)

    if reuse_only:
        res, status, fit_path = try_load_fit_result(run_cfg_name, cfg)
        if res is None:
            raise RuntimeError(
                f"Could not reuse cached run for '{stp_type}' (status={status}, path={fit_path})."
            )
        print(f"Loaded cached run '{run_cfg_name}' from {fit_path}")
    else:
        res = get_fit_result(cfg, cfg_name=run_cfg_name, force_retrain=force_retrain)
        fit_path = DEFAULT_SAVE_DIR / run_cfg_name / "fit.npz"

    res, _, fit_path = ensure_fit_result_asymmetry_history(
        run_cfg_name,
        cfg,
        res,
        save_dir=DEFAULT_SAVE_DIR,
        verbose=True,
    )

    phase, rate_init, rate_final, asymmetry_init, asymmetry_final = _load_saved_asymmetry_from_history(res.history)

    fi_curve, rate_penalty_curve, total_loss_curve = _training_curves(cfg, res.history)
    w0_init, U_init = init_params(cfg)
    fi_init_mean, fi_init_se = evaluate_FI(
        cfg, w0_init, U_init, n_trials=fi_trials, seed=seed_base + 21
    )
    fi_final_mean, fi_final_se = evaluate_FI(
        cfg, res.w0, res.U, n_trials=fi_trials, seed=seed_base + 22
    )

    return ConditionSummary(
        cfg=cfg,
        stp_type=stp_type,
        label=STP_LABELS.get(stp_type, stp_type),
        run_cfg_name=run_cfg_name,
        fit_path=fit_path,
        w0=np.asarray(res.w0, dtype=float).copy(),
        U=np.asarray(res.U, dtype=float).copy(),
        phase=phase,
        rate_init=rate_init,
        rate_final=rate_final,
        asymmetry_init=asymmetry_init,
        asymmetry_final=asymmetry_final,
        fi_curve=fi_curve,
        rate_penalty_curve=rate_penalty_curve,
        total_loss_curve=total_loss_curve,
        fi_init_mean=float(fi_init_mean),
        fi_init_se=float(fi_init_se),
        fi_final_mean=float(fi_final_mean),
        fi_final_se=float(fi_final_se),
    )


def _fisher_series(
    summaries: Sequence[ConditionSummary],
    paper: bool = False,
    *,
    multiline: bool = False,
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
    init_means = np.array([item.fi_init_mean for item in summaries], dtype=float)
    init_ses = np.array([item.fi_init_se for item in summaries], dtype=float)
    init_mean, init_se = _mean_and_combined_se(init_means, init_ses)

    labels: List[str] = ["init"]
    values: List[float] = [init_mean]
    errors: List[float] = [init_se]
    colors: List[str] = ["0.35"]

    for item in summaries:
        labels.append(_condition_label(item, paper=paper, multiline=multiline))
        values.append(float(item.fi_final_mean))
        errors.append(float(item.fi_final_se))
        colors.append(_condition_color(item.stp_type))

    return labels, np.asarray(values, dtype=float), np.asarray(errors, dtype=float), colors


def _draw_phase_aligned_rate(
    ax: plt.Axes,
    summaries: Sequence[ConditionSummary],
    paper: bool = False,
) -> Tuple[List[Any], List[str]]:
    phase_ref = np.array([], dtype=float)
    if len(summaries) > 0:
        phase_ref = np.asarray(summaries[0].phase, dtype=float)
        init_curves = [np.asarray(item.rate_init, dtype=float) for item in summaries if item.rate_init is not None]
        init_mean = _stacked_mean(init_curves)
        n_init = min(phase_ref.size, init_mean.size)
        if n_init > 0:
            ax.plot(
                phase_ref[:n_init],
                init_mean[:n_init],
                linestyle="--",
                linewidth=1.7,
                color="black",
                label="init",
            )
    for item in summaries:
        phase = np.asarray(item.phase, dtype=float)
        rate_final = np.asarray(item.rate_final, dtype=float)
        n = min(phase.size, rate_final.size)
        if n <= 0:
            continue
        ax.plot(
            phase[:n],
            rate_final[:n],
            linestyle="-",
            linewidth=1.8,
            color=_condition_color(item.stp_type),
            label=_condition_label(item, paper=paper),
        )
    ax.set_xlabel("input phase (rad)")
    ax.set_ylabel("rate (Hz)")
    handles, labels = ax.get_legend_handles_labels()
    return list(handles), list(labels)


def _draw_fisher_info(ax: plt.Axes, summaries: Sequence[ConditionSummary], paper: bool = False) -> None:
    labels, values, errors, colors = _fisher_series(summaries, paper=paper, multiline=paper)
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=3, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0 if paper else 15)
    ax.set_ylabel("FI (per neuron per s)")


def _draw_spontaneous_cplus_profile(
    ax: plt.Axes,
    corr_data: CorrelogramCompareData,
    paper: bool = False,
) -> Tuple[List[Any], List[str]]:
    handles: List[Any] = []
    labels: List[str] = []
    ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    if paper:
        ax.axvline(0.0, color="0.35", linestyle=":", linewidth=1.1, zorder=0)
    for item, profile in zip(corr_data.items, corr_data.cplus_spont):
        (line,) = ax.plot(
            corr_data.zeta,
            profile,
            color=_condition_color(item.stp_type),
            linewidth=1.6,
            label=_condition_label(item, paper=paper),
        )
        handles.append(line)
        labels.append(_condition_label(item, paper=paper))
    ax.set_xlabel(r"offset $\Delta z$")
    ax.set_ylabel(r"$C_+^\kappa(\Delta z)$")
    return handles, labels


def _draw_assoc_spontaneous_correlogram(
    ax: plt.Axes,
    corr_data: CorrelogramCompareData,
) -> Optional[Any]:
    assoc_idx = corr_data.assoc_idx
    if assoc_idx is None:
        ax.text(0.5, 0.5, "associative STP missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None

    lag_arr = corr_data.lags_spont_list[assoc_idx]
    extent = [float(lag_arr[0]), float(lag_arr[-1]), float(corr_data.zeta[0]), float(corr_data.zeta[-1])]
    im = ax.imshow(
        np.ma.masked_invalid(corr_data.spont_mats[assoc_idx]),
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=-corr_data.vmax_spont,
        vmax=corr_data.vmax_spont,
        cmap="coolwarm",
    )
    ax.set_xlim(-corr_data.heatmap_lag_limit_s, corr_data.heatmap_lag_limit_s)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel(r"offset $\Delta z$ (rad)")
    return im


@_with_paper_rc
def _plot_condition_compare(
    out_dir: Path,
    summaries: Sequence[ConditionSummary],
    paper: bool = False,
    *,
    corr_data: Optional[CorrelogramCompareData] = None,
) -> Path:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=_paper_figsize((11.2, 8.8), paper=paper, max_width=11.2),
        gridspec_kw={"height_ratios": [1.0, 1.0], "width_ratios": [1.0, 1.05]},
    )
    axes = np.asarray(axes, dtype=object).reshape(2, 2)
    axes_flat = list(axes.ravel())

    phase_handles, phase_labels = _draw_phase_aligned_rate(axes[0, 0], summaries, paper=paper)
    _draw_fisher_info(axes[0, 1], summaries, paper=paper)

    profile_handles: List[Any] = []
    profile_labels: List[str] = []
    im = None
    if corr_data is not None:
        profile_handles, profile_labels = _draw_spontaneous_cplus_profile(axes[1, 0], corr_data, paper=paper)
        im = _draw_assoc_spontaneous_correlogram(axes[1, 1], corr_data)
    else:
        for ax in axes[1, :]:
            ax.text(0.5, 0.5, "correlogram data missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

    panel_labels = ["A", "B", "C", "D"]
    panel_descriptions = [
        "firing rate",
        "Fisher information",
        "spontaneous profile",
        "associative STP spontaneous correlogram",
    ]
    for ax, letter, desc in zip(axes_flat, panel_labels, panel_descriptions):
        ax.set_title(letter, loc="left")
        if not paper:
            ax.text(0.5, 1.02, desc, transform=ax.transAxes, ha="center", va="bottom")

    if im is not None:
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    legend_handles = profile_handles if profile_handles else phase_handles
    legend_labels = profile_labels if profile_labels else phase_labels
    if paper:
        _legend_outside(fig, legend_handles, legend_labels, paper=True, anchor_x=0.94, anchor_y=0.98)
        fig.tight_layout(rect=(0, 0, 0.935, 1))
    else:
        legend_ax = axes[1, 0] if profile_handles else axes[0, 0]
        if legend_handles:
            legend_ax.legend(frameon=False)
        fig.tight_layout()

    out_path = out_dir / "condition_compare.pdf"
    fig.savefig(out_path, **_savefig_kwargs(paper))
    plt.close(fig)
    return out_path


@_with_paper_rc
def _plot_phase_aligned_rate(out_dir: Path, summaries: Sequence[ConditionSummary], paper: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=_paper_figsize((6.4, 4.2), paper=paper, max_width=6.4))
    handles, labels = _draw_phase_aligned_rate(ax, summaries, paper=paper)
    if paper:
        _legend_outside(fig, handles, labels, paper=True, anchor_x=0.995, anchor_y=0.98)
        fig.tight_layout(rect=(0, 0, 0.88, 1))
    else:
        ax.legend(frameon=False)
        fig.tight_layout()
    out_path = out_dir / "condition_compare_phase_aligned_rate.pdf"
    fig.savefig(out_path, **_savefig_kwargs(paper))
    plt.close(fig)
    return out_path


@_with_paper_rc
def _plot_asymmetry_metrics(out_dir: Path, summaries: Sequence[ConditionSummary], paper: bool = False) -> Path:
    init_series, final_series = _collect_asymmetry_metric_series(summaries)
    metric_specs = ASYMMETRY_METRIC_SPECS
    if paper:
        metric_specs = tuple((key, label) for key, label in ASYMMETRY_METRIC_SPECS if key != "peak_phase")

    if paper:
        n_rows, n_cols = 2, 2
        figsize = (10.8, 7.1)
    else:
        n_rows, n_cols = 2, 3
        figsize = (12.6, 6.8)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=_paper_figsize(figsize, paper=paper, max_width=12.6),
        sharex=False,
    )
    axes_flat = list(np.asarray(axes, dtype=object).ravel())
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    x = np.arange(len(summaries), dtype=float)
    width = 0.36
    final_colors = [_condition_color(item.stp_type) for item in summaries]
    cond_labels = [_condition_label(item, paper=paper, multiline=paper) for item in summaries]
    na_fontsize: Any = "x-small" if paper else 8

    for idx, (key, ylabel) in enumerate(metric_specs):
        ax = axes_flat[idx]
        init_vals = init_series[key]
        final_vals = final_series[key]

        if paper:
            finite_init = init_vals[np.isfinite(init_vals)]
            init_value = float(np.mean(finite_init)) if finite_init.size > 0 else float("nan")
            labels = ["init", *cond_labels]
            values = np.asarray([init_value, *final_vals.tolist()], dtype=float)
            colors = ["0.35", *final_colors]
            x_plot = np.arange(values.size, dtype=float)
            ax.bar(
                x_plot,
                np.where(np.isfinite(values), values, 0.0),
                width=0.7,
                color=colors,
                edgecolor="0.25",
                linewidth=0.6,
            )
            ax.set_xticks(x_plot)
            ax.set_xticklabels(labels, rotation=0)
            for x_val, val in zip(x_plot, values):
                if not np.isfinite(val):
                    ax.text(
                        x_val,
                        0.0,
                        "na",
                        ha="center",
                        va="bottom",
                        fontsize=na_fontsize,
                        rotation=90,
                        color="0.35",
                    )
        else:
            init_plot = np.where(np.isfinite(init_vals), init_vals, 0.0)
            final_plot = np.where(np.isfinite(final_vals), final_vals, 0.0)
            ax.bar(
                x - 0.5 * width,
                init_plot,
                width=width,
                color="0.72",
                edgecolor="0.35",
                linewidth=0.6,
                hatch="//",
                label="init" if idx == 0 else None,
            )
            ax.bar(
                x + 0.5 * width,
                final_plot,
                width=width,
                color=final_colors,
                edgecolor="0.25",
                linewidth=0.6,
                label="final" if idx == 0 else None,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(cond_labels, rotation=12)

            for x_init, val in zip(x - 0.5 * width, init_vals):
                if not np.isfinite(val):
                    ax.text(
                        x_init,
                        0.0,
                        "na",
                        ha="center",
                        va="bottom",
                        fontsize=na_fontsize,
                        rotation=90,
                        color="0.35",
                    )
            for x_final, val in zip(x + 0.5 * width, final_vals):
                if not np.isfinite(val):
                    ax.text(
                        x_final,
                        0.0,
                        "na",
                        ha="center",
                        va="bottom",
                        fontsize=na_fontsize,
                        rotation=90,
                        color="0.35",
                    )

        ax.axhline(0.0, color="0.3", linewidth=0.8, alpha=0.5)
        ax.set_title(letters[idx], loc="left")
        ax.set_ylabel(ylabel)

    for idx in range(len(metric_specs), len(axes_flat)):
        axes_flat[idx].axis("off")

    if paper:
        fig.tight_layout()
    else:
        axes_flat[0].legend(frameon=False)
        fig.tight_layout()

    out_path = out_dir / "condition_compare_asymmetry_metrics.pdf"
    fig.savefig(out_path, **_savefig_kwargs(paper))
    plt.close(fig)
    return out_path


@_with_paper_rc
def _plot_training_curves(
    out_dir: Path, summaries: Sequence[ConditionSummary], paper: bool = False
) -> Path:
    specs = [
        ("fi_curve", "FI (per neuron per s)", "FI"),
        ("rate_penalty_curve", "rate penalty", "Rate error term"),
        ("total_loss_curve", "FI - rate penalty", "Total loss"),
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=_paper_figsize((12.0, 3.8), paper=paper, max_width=12.0),
        sharex=True,
    )
    axes = np.atleast_1d(np.asarray(axes, dtype=object))
    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    for ax, (attr, ylabel, title) in zip(axes, specs):
        has_any = False
        panel_handles: List[Any] = []
        panel_labels: List[str] = []
        for item in summaries:
            curve = getattr(item, attr)
            if curve is None:
                continue
            label = _condition_label(item, paper=paper)
            it = np.arange(curve.size)
            (line,) = ax.plot(
                it,
                curve,
                color=_condition_color(item.stp_type),
                linewidth=1.4,
                label=label,
            )
            panel_handles.append(line)
            panel_labels.append(label)
            has_any = True
        if (not legend_handles) and panel_handles:
            legend_handles = panel_handles
            legend_labels = panel_labels
        if not has_any:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    if paper and legend_handles:
        _legend_outside(fig, legend_handles, legend_labels, paper=True, anchor_x=0.995, anchor_y=0.98)
        fig.tight_layout(rect=(0, 0, 0.88, 1))
    else:
        if legend_handles:
            axes[0].legend(frameon=False)
        fig.tight_layout()

    out_path = out_dir / "condition_compare_training_curve.pdf"
    fig.savefig(out_path, **_savefig_kwargs(paper))
    plt.close(fig)
    return out_path


def _max_abs_mats(mats: Sequence[np.ndarray]) -> float:
    vals = [float(np.nanmax(np.abs(m))) for m in mats if np.any(np.isfinite(m))]
    if len(vals) == 0:
        return 1.0
    vmax = max(vals)
    return vmax if np.isfinite(vmax) and vmax > 0.0 else 1.0


def _cplus_profiles(
    items: Sequence[ConditionSummary],
    mats: Sequence[np.ndarray],
    lag_list: Sequence[np.ndarray],
) -> List[np.ndarray]:
    profiles: List[np.ndarray] = []
    smooth_window = 7
    for item, mat, lag_arr in zip(items, mats, lag_list):
        tau0 = 5.0 * float(item.cfg.tau_s)
        profile = correlogram_epsp_weighted_positive_lag(
            mat,
            lag_arr,
            float(item.cfg.tau_s),
            tau_max=tau0,
        )
        profile = np.asarray(profile, dtype=float).copy()
        # Keep Delta z=0 in this 1D profile; only the heatmap center cell is masked.
        profiles.append(_circular_moving_average(profile, smooth_window))
    return profiles


def _compute_correlogram_compare_data(
    summaries: Sequence[ConditionSummary],
    n_corr_trials: int = 8,
    heatmap_lag_limit_s: float = 0.1,
) -> CorrelogramCompareData:
    items = [item for item in _condition_summaries_ordered(summaries) if item.stp_type in STP_TYPES]
    if len(items) == 0:
        raise RuntimeError("No valid STP-condition summaries for correlogram comparison.")

    N_ref = int(items[0].cfg.N)
    zeta, order = centered_offset_angles(N_ref)
    idx_z0 = int(np.argmin(np.abs(zeta)))

    max_lag_bins = max(1, int(np.ceil(heatmap_lag_limit_s / max(float(items[0].cfg.bin_dt), 1e-9))))

    evoked_mats: List[np.ndarray] = []
    spont_mats: List[np.ndarray] = []
    evoked_init_mats: List[np.ndarray] = []
    spont_init_mats: List[np.ndarray] = []
    lags_evoked_list: List[np.ndarray] = []
    lags_spont_list: List[np.ndarray] = []
    lags_evoked_ref: Optional[np.ndarray] = None
    lags_spont_ref: Optional[np.ndarray] = None

    for col, item in enumerate(items):
        if int(item.cfg.N) != N_ref:
            raise ValueError("All conditions must share the same N to compare correlograms.")

        w0_init, U_init = init_params(item.cfg)

        spikes_evoked_init = sample_spikes_trials(
            item.cfg, w0_init, U_init, n_trials=n_corr_trials, seed=8050 + 1000 * col
        )
        _, _, corr_evoked_init = correlogram_by_distance_trial_shuffle(
            spikes_evoked_init, max_lag_bins=max_lag_bins, shuffle_mode="roll"
        )
        spikes_evoked = sample_spikes_trials(
            item.cfg, item.w0, item.U, n_trials=n_corr_trials, seed=8100 + 1000 * col
        )
        _, _, corr_evoked = correlogram_by_distance_trial_shuffle(
            spikes_evoked, max_lag_bins=max_lag_bins, shuffle_mode="roll"
        )
        lags_evoked = np.arange(-max_lag_bins, max_lag_bins + 1, dtype=float) * float(item.cfg.bin_dt)
        corr_evoked_init = np.asarray(corr_evoked_init, dtype=float)[order, :].copy()
        corr_evoked = np.asarray(corr_evoked, dtype=float)[order, :].copy()
        idx_t0 = int(np.argmin(np.abs(lags_evoked)))
        corr_evoked_init[idx_z0, idx_t0] = np.nan
        corr_evoked[idx_z0, idx_t0] = np.nan
        evoked_init_mats.append(corr_evoked_init)
        evoked_mats.append(corr_evoked)
        lags_evoked_list.append(lags_evoked)
        if lags_evoked_ref is None:
            lags_evoked_ref = lags_evoked

        cfg_spont = RingConfig(**asdict(item.cfg))
        cfg_spont.A = 0.0
        cfg_spont.h_bg = item.cfg.spont_h_bg
        spikes_spont_init = sample_spikes_trials(
            cfg_spont, w0_init, U_init, n_trials=n_corr_trials, seed=8550 + 1000 * col
        )
        _, _, corr_spont_init = correlogram_by_distance_trial_shuffle(
            spikes_spont_init, max_lag_bins=max_lag_bins, shuffle_mode="roll"
        )
        spikes_spont = sample_spikes_trials(
            cfg_spont, item.w0, item.U, n_trials=n_corr_trials, seed=8600 + 1000 * col
        )
        _, _, corr_spont = correlogram_by_distance_trial_shuffle(
            spikes_spont, max_lag_bins=max_lag_bins, shuffle_mode="roll"
        )
        lags_spont = np.arange(-max_lag_bins, max_lag_bins + 1, dtype=float) * float(cfg_spont.bin_dt)
        corr_spont_init = np.asarray(corr_spont_init, dtype=float)[order, :].copy()
        corr_spont = np.asarray(corr_spont, dtype=float)[order, :].copy()
        idx_t0_spont = int(np.argmin(np.abs(lags_spont)))
        corr_spont_init[idx_z0, idx_t0_spont] = np.nan
        corr_spont[idx_z0, idx_t0_spont] = np.nan
        spont_init_mats.append(corr_spont_init)
        spont_mats.append(corr_spont)
        lags_spont_list.append(lags_spont)
        if lags_spont_ref is None:
            lags_spont_ref = lags_spont

    if lags_evoked_ref is None or lags_spont_ref is None:
        raise RuntimeError("Could not build correlogram comparison data.")

    cplus_evoked = _cplus_profiles(items, evoked_mats, lags_evoked_list)
    cplus_spont = _cplus_profiles(items, spont_mats, lags_spont_list)
    cplus_evoked_diff = _cplus_profiles(
        items,
        [learn - init for learn, init in zip(evoked_mats, evoked_init_mats)],
        lags_evoked_list,
    )
    cplus_spont_diff = _cplus_profiles(
        items,
        [learn - init for learn, init in zip(spont_mats, spont_init_mats)],
        lags_spont_list,
    )

    return CorrelogramCompareData(
        items=list(items),
        zeta=np.asarray(zeta, dtype=float),
        evoked_mats=evoked_mats,
        spont_mats=spont_mats,
        evoked_init_mats=evoked_init_mats,
        spont_init_mats=spont_init_mats,
        lags_evoked_list=lags_evoked_list,
        lags_spont_list=lags_spont_list,
        lags_evoked_ref=np.asarray(lags_evoked_ref, dtype=float),
        lags_spont_ref=np.asarray(lags_spont_ref, dtype=float),
        cplus_evoked=cplus_evoked,
        cplus_spont=cplus_spont,
        cplus_evoked_diff=cplus_evoked_diff,
        cplus_spont_diff=cplus_spont_diff,
        vmax_evoked=_max_abs_mats(evoked_mats),
        vmax_spont=_max_abs_mats(spont_mats),
        heatmap_lag_limit_s=float(heatmap_lag_limit_s),
        assoc_idx=next((idx for idx, item in enumerate(items) if item.stp_type == "with_stp"), None),
    )


@_with_paper_rc
def _plot_correlogram_compare(
    out_dir: Path,
    summaries: Sequence[ConditionSummary],
    paper: bool = False,
    n_corr_trials: int = 8,
    heatmap_lag_limit_s: float = 0.1,
    corr_data: Optional[CorrelogramCompareData] = None,
) -> List[Path]:
    if corr_data is None:
        corr_data = _compute_correlogram_compare_data(
            summaries,
            n_corr_trials=n_corr_trials,
            heatmap_lag_limit_s=heatmap_lag_limit_s,
        )

    items = corr_data.items
    zeta = corr_data.zeta
    evoked_mats = corr_data.evoked_mats
    spont_mats = corr_data.spont_mats
    lags_evoked_ref = corr_data.lags_evoked_ref
    lags_spont_ref = corr_data.lags_spont_ref
    vmax_evoked = corr_data.vmax_evoked
    vmax_spont = corr_data.vmax_spont
    heatmap_lag_limit_s = corr_data.heatmap_lag_limit_s

    n_cols = len(items)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=_paper_figsize((3.6 * n_cols + 1.6, 6.3), paper=paper, max_width=12.8),
        sharex=False,
        sharey=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(2, n_cols)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    im_evoked = None
    im_spont = None
    for col, item in enumerate(items):
        extent_e = [float(lags_evoked_ref[0]), float(lags_evoked_ref[-1]), float(zeta[0]), float(zeta[-1])]
        ax_e = axes[0, col]
        im_evoked = ax_e.imshow(
            np.ma.masked_invalid(evoked_mats[col]),
            aspect="auto",
            origin="lower",
            extent=extent_e,
            vmin=-vmax_evoked,
            vmax=vmax_evoked,
            cmap="coolwarm",
        )
        ax_e.set_xlim(-heatmap_lag_limit_s, heatmap_lag_limit_s)
        ax_e.set_title(letters[col], loc="left")
        ax_e.text(
            0.5,
            1.03,
            _condition_label(item, paper=paper),
            transform=ax_e.transAxes,
            ha="center",
            va="bottom",
        )
        if col == 0:
            ax_e.set_ylabel("evoked\noffset Δz (rad)")

        extent_s = [float(lags_spont_ref[0]), float(lags_spont_ref[-1]), float(zeta[0]), float(zeta[-1])]
        ax_s = axes[1, col]
        im_spont = ax_s.imshow(
            np.ma.masked_invalid(spont_mats[col]),
            aspect="auto",
            origin="lower",
            extent=extent_s,
            vmin=-vmax_spont,
            vmax=vmax_spont,
            cmap="coolwarm",
        )
        ax_s.set_xlim(-heatmap_lag_limit_s, heatmap_lag_limit_s)
        ax_s.set_title(letters[n_cols + col], loc="left")
        if col == 0:
            ax_s.set_ylabel("spontaneous\noffset Δz (rad)")
        ax_s.set_xlabel("lag (s)")

    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.1, top=0.93, wspace=0.24, hspace=0.28)
    cbar_width = 0.015
    cbar_pad = 0.01
    if im_evoked is not None:
        pos_row0 = [axes[0, c].get_position() for c in range(n_cols)]
        y0 = min(p.y0 for p in pos_row0)
        y1 = max(p.y1 for p in pos_row0)
        x1 = max(p.x1 for p in pos_row0)
        cax0 = fig.add_axes([x1 + cbar_pad, y0, cbar_width, y1 - y0])
        fig.colorbar(im_evoked, cax=cax0)
    if im_spont is not None:
        pos_row1 = [axes[1, c].get_position() for c in range(n_cols)]
        y0 = min(p.y0 for p in pos_row1)
        y1 = max(p.y1 for p in pos_row1)
        x1 = max(p.x1 for p in pos_row1)
        cax1 = fig.add_axes([x1 + cbar_pad, y0, cbar_width, y1 - y0])
        fig.colorbar(im_spont, cax=cax1)
    heatmap_path = out_dir / "condition_compare_correlogram.pdf"
    fig.savefig(heatmap_path, **_savefig_kwargs(paper))
    plt.close(fig)
    if paper:
        return [heatmap_path]

    def _plot_spontaneous_cplus_assoc_correlogram() -> Optional[Path]:
        if corr_data.assoc_idx is None:
            return None

        fig_combo, axes_combo = plt.subplots(
            1,
            2,
            figsize=_paper_figsize((10.8, 4.3), paper=paper, max_width=10.8),
            gridspec_kw={"width_ratios": [1.0, 1.12]},
        )
        ax_profile, ax_hm = np.atleast_1d(np.asarray(axes_combo, dtype=object))

        handles, labels = _draw_spontaneous_cplus_profile(ax_profile, corr_data, paper=paper)
        im = _draw_assoc_spontaneous_correlogram(ax_hm, corr_data)

        if paper:
            ax_profile.set_title("A", loc="left")
            ax_hm.set_title("B", loc="left")
            ax_profile.text(
                0.5, 1.03, "spontaneous profile", transform=ax_profile.transAxes, ha="center", va="bottom"
            )
            ax_hm.text(
                0.5,
                1.03,
                "associative STP spontaneous correlogram",
                transform=ax_hm.transAxes,
                ha="center",
                va="bottom",
            )
            _legend_outside(fig_combo, handles, labels, paper=True, anchor_x=0.995, anchor_y=0.98)
            fig_combo.tight_layout(rect=(0, 0, 0.86, 1))
        else:
            ax_profile.set_title("A  spontaneous profile", loc="left")
            ax_hm.set_title("B  associative STP spontaneous correlogram", loc="left")
            ax_profile.legend(frameon=False)
            fig_combo.tight_layout()
        if im is not None:
            fig_combo.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

        out_path = out_dir / "condition_compare_spontaneous_cplus_assoc_correlogram.pdf"
        fig_combo.savefig(out_path, **_savefig_kwargs(paper))
        plt.close(fig_combo)
        return out_path

    def _plot_cplus_profile_compare(
        evoked_profiles: Sequence[np.ndarray],
        spont_profiles: Sequence[np.ndarray],
        filename: str,
        *,
        ylabel: str = r"$C_+^\kappa(\Delta z)$",
        title_suffix: str = "",
    ) -> Path:
        fig_cplus, axes_cplus = plt.subplots(
            1,
            2,
            figsize=_paper_figsize((10.8, 4.2), paper=paper, max_width=10.8),
            sharex=True,
        )
        axes_cplus = np.atleast_1d(np.asarray(axes_cplus, dtype=object))
        panel_specs = [
            (axes_cplus[0], evoked_profiles, "stimulus presentation"),
            (axes_cplus[1], spont_profiles, "spontaneous"),
        ]
        handles: List[Any] = []
        labels: List[str] = []
        for panel_idx, (ax, profiles, panel_title) in enumerate(panel_specs):
            ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
            for item, profile in zip(items, profiles):
                (line,) = ax.plot(
                    zeta,
                    profile,
                    color=_condition_color(item.stp_type),
                    linewidth=1.6,
                    label=_condition_label(item, paper=paper),
                )
                if panel_idx == 0:
                    handles.append(line)
                    labels.append(_condition_label(item, paper=paper))
            ax.set_xlabel(r"offset $\Delta z$")
            ax.set_ylabel(ylabel)
            if not paper:
                ax.set_title(f"{panel_title}{title_suffix}")
            else:
                ax.set_title(letters[panel_idx], loc="left")
                ax.text(
                    0.5,
                    1.03,
                    f"{panel_title}{title_suffix}",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                )

        if paper:
            _legend_outside(fig_cplus, handles, labels, paper=True, anchor_x=0.995, anchor_y=0.98)
            fig_cplus.tight_layout(rect=(0, 0, 0.88, 1))
        else:
            axes_cplus[0].legend(frameon=False)
            fig_cplus.tight_layout()
        out_path = out_dir / filename
        fig_cplus.savefig(out_path, **_savefig_kwargs(paper))
        plt.close(fig_cplus)
        return out_path

    cplus_path = _plot_cplus_profile_compare(
        corr_data.cplus_evoked,
        corr_data.cplus_spont,
        "condition_compare_positive_lag_weighted.pdf",
    )
    cplus_diff_path = _plot_cplus_profile_compare(
        corr_data.cplus_evoked_diff,
        corr_data.cplus_spont_diff,
        "condition_compare_positive_lag_weighted_diff.pdf",
        ylabel=r"$\Delta C_+^\kappa(\Delta z)$",
        title_suffix=" (learned - init)",
    )
    combo_path = _plot_spontaneous_cplus_assoc_correlogram()
    out_paths = [heatmap_path, cplus_path, cplus_diff_path]
    if combo_path is not None:
        out_paths.append(combo_path)
    return out_paths


def _replay_panel_data(
    replay_out: Dict[str, np.ndarray],
    trial: int = 0,
    plot_t_start: Optional[float] = None,
    plot_t_end: Optional[float] = None,
) -> Dict[str, Any]:
    spikes = np.asarray(replay_out["spikes_binned"][trial], dtype=float)
    t = np.asarray(replay_out["t_binned"], dtype=float)
    if t.size == 0:
        raise ValueError("Replay output has empty time bins.")
    z = np.asarray(replay_out["z"], dtype=float)
    bin_dt = float(replay_out.get("bin_dt", (t[1] - t[0]) if t.size > 1 else 1.0))

    if plot_t_start is None:
        plot_t_start = replay_out.get("plot_t_start")
    if plot_t_end is None:
        plot_t_end = replay_out.get("plot_t_end")
    t_lo_full = float(t[0] - 0.5 * bin_dt)
    t_hi_full = float(t[-1] + 0.5 * bin_dt)
    if plot_t_start is None:
        plot_t_start = t_lo_full
    if plot_t_end is None:
        plot_t_end = t_hi_full
    plot_t_start = float(plot_t_start)
    plot_t_end = float(plot_t_end)
    if plot_t_end <= plot_t_start:
        raise ValueError(f"plot_t_end must be greater than plot_t_start (got {plot_t_start}, {plot_t_end}).")

    t_left = t - 0.5 * bin_dt
    t_right = t + 0.5 * bin_dt
    mask = (t_right > plot_t_start) & (t_left < plot_t_end)
    if not np.any(mask):
        raise ValueError(
            f"Requested plot window [{plot_t_start}, {plot_t_end}] does not overlap data "
            f"[{t_lo_full}, {t_hi_full}]."
        )

    spikes = spikes[mask]
    t_left = t_left[mask]
    t_right = t_right[mask]
    z_wrapped = (z + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(z_wrapped)
    z_plot = z_wrapped[order]
    spikes_plot = spikes[:, order] / max(bin_dt, 1e-9)

    cue_window = None
    if "cue_window" in replay_out:
        try:
            cue = replay_out["cue_window"]
            cue_window = (float(cue[0]), float(cue[1]))
        except Exception:
            cue_window = None

    return {
        "spikes": spikes_plot,
        "z_plot": z_plot,
        "x_min": float(t_left[0]),
        "x_max": float(t_right[-1]),
        "plot_t_start": plot_t_start,
        "plot_t_end": plot_t_end,
        "cue_window": cue_window,
    }


@_with_paper_rc
def _plot_replay_compare(
    out_dir: Path,
    summaries: Sequence[ConditionSummary],
    replay_cfg: ReplayConfig,
    paper: bool = False,
) -> Path:
    items = [item for item in _condition_summaries_ordered(summaries) if item.stp_type in STP_TYPES]
    if len(items) == 0:
        raise RuntimeError("No valid STP-condition summaries for replay comparison.")

    panels: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        replay = simulate_replay_activity(
            item.cfg,
            item.w0,
            item.U,
            replay_cfg,
            seed=int(replay_cfg.seed) + idx * 1000,
            stp_type=item.stp_type,
        )
        panel = _replay_panel_data(
            replay,
            trial=0,
            plot_t_start=replay_cfg.plot_t_start,
            plot_t_end=replay_cfg.plot_t_end,
        )
        panels.append(panel)

    n_cols = len(panels)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=_paper_figsize((3.3 * n_cols + 1.0, 3.6), paper=paper, max_width=12.8),
        sharey=True,
    )
    axes = np.atleast_1d(np.asarray(axes, dtype=object))
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for col, (ax, panel) in enumerate(zip(axes, panels)):
        ax.imshow(
            panel["spikes"].T,
            aspect="auto",
            origin="lower",
            extent=[panel["x_min"], panel["x_max"], float(panel["z_plot"][0]), float(panel["z_plot"][-1])],
            interpolation="none",
            cmap="viridis",
        )
        cue_window = panel["cue_window"]
        if cue_window is not None:
            cue_lo = max(float(cue_window[0]), panel["plot_t_start"])
            cue_hi = min(float(cue_window[1]), panel["plot_t_end"])
            if cue_hi > cue_lo:
                ax.axvspan(cue_lo, cue_hi, color="white", alpha=0.16)
        ax.set_xlim(float(panel["plot_t_start"]), float(panel["plot_t_end"]))
        ax.set_xlabel("time (s)")
        if col == 0:
            ax.set_ylabel("position z (rad)")
        ax.set_title(letters[col], loc="left")

    fig.tight_layout()
    out_path = out_dir / "condition_compare_replay.pdf"
    fig.savefig(out_path, **_savefig_kwargs(paper))
    plt.close(fig)
    return out_path


def _save_fisher_summary(
    out_dir: Path,
    cfg_path: Path,
    cfg_name: str,
    phase_trials: int,
    fi_trials: int,
    summaries: Sequence[ConditionSummary],
) -> Path:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cfg_path": _portable_path(cfg_path),
        "cfg_name": cfg_name,
        "phase_trials": int(phase_trials),
        "fi_trials": int(fi_trials),
        "conditions": {},
    }
    for item in summaries:
        payload["conditions"][item.stp_type] = {
            "label": item.label,
            "run_cfg_name": item.run_cfg_name,
            "fit_path": _portable_path(item.fit_path),
            "fi_init": {
                "mean": float(item.fi_init_mean),
                "se": float(item.fi_init_se),
                "n_trials": int(fi_trials),
            },
            "fi_final": {
                "mean": float(item.fi_final_mean),
                "se": float(item.fi_final_se),
                "n_trials": int(fi_trials),
            },
            "rate_init_peak_hz": _safe_nanmax(item.rate_init),
            "rate_final_peak_hz": _safe_nanmax(item.rate_final),
            "asymmetry_init": {key: float(item.asymmetry_init.get(key, float("nan"))) for key, _ in ASYMMETRY_METRIC_SPECS},
            "asymmetry_final": {key: float(item.asymmetry_final.get(key, float("nan"))) for key, _ in ASYMMETRY_METRIC_SPECS},
        }

    out_path = out_dir / "condition_compare_fisher_info.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ring runs across STP conditions.")
    parser.add_argument("cfg_path", help="Path to JSON config used by run_from_cfg.py")
    parser.add_argument(
        "--stp-type",
        action="append",
        choices=STP_TYPES,
        help="Condition to include; can be repeated (default: all).",
    )
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="Use only cached fit.npz (do not retrain missing runs).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain each condition instead of using cache (ignored with --reuse-only).",
    )
    parser.add_argument(
        "--phase-trials",
        type=int,
        default=8,
        help="Retained for command-line compatibility; phase/asymmetry profiles are loaded from cached history.",
    )
    parser.add_argument(
        "--fi-trials",
        type=int,
        default=6,
        help="Number of trials for FI estimates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7000,
        help="Base seed for comparison sampling/evaluation.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Save paper-ready figures under saved_runs/<cfg>/paper and adjust plotting.",
    )
    parser.add_argument(
        "--corr-trials",
        type=int,
        default=8,
        help="Number of trials per condition for correlogram heatmap comparisons.",
    )
    parser.add_argument(
        "--corr-lag-limit",
        type=float,
        default=0.1,
        help="Absolute lag limit (s) for correlogram heatmap x-axis.",
    )
    args = parser.parse_args()

    if args.fi_trials <= 0:
        raise ValueError("--fi-trials must be positive")
    if args.corr_trials < 2:
        raise ValueError("--corr-trials must be >= 2")
    if args.corr_lag_limit <= 0.0:
        raise ValueError("--corr-lag-limit must be positive")

    cfg_path = Path(args.cfg_path)
    data = _load_json(cfg_path)
    cfg_name, ring_overrides, replay_overrides = _split_overrides(data)
    if not cfg_name:
        cfg_name = cfg_path.stem

    stp_types = _unique_ordered(args.stp_type) if args.stp_type else list(STP_TYPES)
    base_cfg = RingConfig()
    _apply_overrides(base_cfg, ring_overrides, label=str(cfg_path))
    replay_cfg = ReplayConfig()
    if replay_overrides:
        _apply_overrides(replay_cfg, replay_overrides, label=str(cfg_path))

    out_dir = _figure_dir(cfg_name, paper=bool(args.paper))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.paper:
        _remove_skipped_paper_outputs(out_dir)

    summaries = []
    for idx, stp_type in enumerate(stp_types):
        print(f"[compare] Processing condition: {stp_type}")
        item = _load_or_fit_condition(
            cfg_path=cfg_path,
            base_cfg=base_cfg,
            cfg_name=cfg_name,
            stp_type=stp_type,
            reuse_only=bool(args.reuse_only),
            force_retrain=(bool(args.force_retrain) and not bool(args.reuse_only)),
            phase_trials=int(args.phase_trials),
            fi_trials=int(args.fi_trials),
            seed_base=int(args.seed) + idx * 1000,
        )
        summaries.append(item)
    summaries = _condition_summaries_ordered(summaries)

    corr_data = _compute_correlogram_compare_data(
        summaries,
        n_corr_trials=int(args.corr_trials),
        heatmap_lag_limit_s=float(args.corr_lag_limit),
    )

    compare_plot = _plot_condition_compare(out_dir, summaries, paper=bool(args.paper), corr_data=corr_data)
    phase_plot = None
    if not args.paper:
        phase_plot = _plot_phase_aligned_rate(out_dir, summaries, paper=False)
    asym_plot = _plot_asymmetry_metrics(out_dir, summaries, paper=bool(args.paper))
    train_plot = _plot_training_curves(out_dir, summaries, paper=bool(args.paper))
    corr_plots = _plot_correlogram_compare(
        out_dir,
        summaries,
        paper=bool(args.paper),
        n_corr_trials=int(args.corr_trials),
        heatmap_lag_limit_s=float(args.corr_lag_limit),
        corr_data=corr_data,
    )
    replay_plot = _plot_replay_compare(out_dir, summaries, replay_cfg=replay_cfg, paper=bool(args.paper))
    fi_json = _save_fisher_summary(
        out_dir=out_dir,
        cfg_path=cfg_path,
        cfg_name=cfg_name,
        phase_trials=int(args.phase_trials),
        fi_trials=int(args.fi_trials),
        summaries=summaries,
    )

    print(f"Saved figure: {compare_plot}")
    if phase_plot is not None:
        print(f"Saved figure: {phase_plot}")
    print(f"Saved figure: {asym_plot}")
    print(f"Saved figure: {train_plot}")
    for corr_plot in corr_plots:
        print(f"Saved figure: {corr_plot}")
    print(f"Saved figure: {replay_plot}")
    print(f"Saved summary: {fi_json}")


if __name__ == "__main__":
    main()
