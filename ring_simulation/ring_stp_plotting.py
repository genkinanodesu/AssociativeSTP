"""
Plotting and demo runners for the ring STP exact-gradient model.
"""
from __future__ import annotations

import inspect
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ring_stp_exactgrad import (
    RingConfig,
    ReplayConfig,
    DEFAULT_SAVE_DIR,
    _normalize_snapshot_iters,
    centered_offset_angles,
    correlogram_by_distance_trial_shuffle,
    correlogram_epsp_weighted_positive_lag,
    correlogram_window_summary,
    evaluate_FI,
    get_fit_result,
    init_params,
    phase_aligned_rate,
    phase_aligned_synapse_vars,
    sample_spikes_trials,
    save_replay_config,
    simulate_replay_activity,
)


def _figure_dir(cfg_name: str, save_dir: Path = DEFAULT_SAVE_DIR, paper: bool = False) -> Path:
    base = Path(save_dir) / cfg_name
    return base / "paper" if paper else base


def _paper_rc_params() -> Dict[str, Any]:
    """
    Matplotlib defaults tuned for manuscript-ready figures.
    """
    # Paper PDFs are often scaled down into a two-column manuscript, so source
    # text needs to be larger than the intended final printed size.
    return {
        "font.family": "sans-serif",
        "font.size": 18.0,
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
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": False,
        "image.interpolation": "none",
        "savefig.dpi": 300,
        # Keep editable text in vector exports (avoid Type 3 fonts).
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def _savefig_kwargs(paper: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
    if paper:
        kwargs.update({"dpi": 300, "facecolor": "white"})
    return kwargs


def _paper_figsize(
    size: Tuple[float, float],
    paper: bool,
    max_width: float = 7.0,
) -> Tuple[float, float]:
    """
    Clamp very wide diagnostic figures in paper mode to reduce downstream scaling.
    """
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


def _profile_snapshot_iters(n_iter: int, n_snapshots: int = 5) -> List[int]:
    if n_iter <= 0 or n_snapshots <= 0:
        return []
    steps = []
    for k in range(1, n_snapshots + 1):
        step = int(round(k * n_iter / n_snapshots))
        step = max(1, min(step, n_iter))
        steps.append(step)
    out = []
    seen = set()
    for step in steps:
        if step not in seen:
            out.append(step)
            seen.add(step)
    return out


@_with_paper_rc
def plot_replay_heatmap(
    replay_out: Dict[str, np.ndarray],
    trial: int = 0,
    cue_window: Optional[Tuple[float, float]] = None,
    title: Optional[str] = "Replay activity (background-only drive)",
    show: bool = True,
    paper: bool = False,
    plot_t_start: Optional[float] = None,
    plot_t_end: Optional[float] = None,
):
    """
    Quick visualization of replay trajectory from simulate_replay_activity output.
    """
    import matplotlib.pyplot as plt

    spikes = replay_out["spikes_binned"][trial]
    t = replay_out["t_binned"]
    if t.size == 0:
        raise ValueError("replay_out['t_binned'] is empty. Increase replay T or reduce bin_dt.")
    z = replay_out["z"]
    bin_dt = replay_out.get("bin_dt", (t[1] - t[0]) if len(t) > 1 else 1.0)
    if cue_window is None and ("cue_window" in replay_out):
        try:
            cue_w = replay_out["cue_window"]
            cue_window = (float(cue_w[0]), float(cue_w[1]))
        except Exception:
            cue_window = None

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
    time_mask = (t_right > plot_t_start) & (t_left < plot_t_end)
    if not np.any(time_mask):
        raise ValueError(
            f"Requested plot window [{plot_t_start}, {plot_t_end}] does not overlap data "
            f"[{t_lo_full}, {t_hi_full}]."
        )

    spikes = spikes[time_mask]
    t = t[time_mask]
    t_left = t_left[time_mask]
    t_right = t_right[time_mask]
    ph = replay_out["phase_unwrapped"][trial]
    ph = ph[time_mask]
    ph_wrapped = np.angle(np.exp(1j * ph))

    # wrap positions to (-pi, pi] and sort for display
    z_wrapped = (z + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(z_wrapped)
    z_plot = z_wrapped[order]
    spikes_plot = spikes[:, order]
    # Keep native bin width in the image, then clip with xlim to the requested window.
    x_img_min = float(t_left[0])
    x_img_max = float(t_right[-1])

    figsize = (6, 4) if paper else (8, 4)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        spikes_plot.T / max(bin_dt, 1e-9),
        aspect="auto",
        origin="lower",
        extent=[x_img_min, x_img_max, z_plot[0], z_plot[-1]],
        interpolation="none",
    )
    if cue_window is not None:
        cue_lo = max(float(cue_window[0]), plot_t_start)
        cue_hi = min(float(cue_window[1]), plot_t_end)
        if cue_hi > cue_lo:
            ax.axvspan(cue_lo, cue_hi, color="gray", alpha=0.2, label="cue")
    ax.set_xlim(plot_t_start, plot_t_end)
    mask = ~np.isnan(ph_wrapped)
    # if np.any(mask):
    #     ax.plot(t[mask], ph_wrapped[mask], color="white", linewidth=0.1, label="phase")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position z (rad)")
    if (title is not None) and (not paper): # avoid title and legend in paper figures.
        ax.set_title(title + f" | trial {trial} | vel={replay_out['phase_velocity'][trial]:+.2f} rad/s")
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(frameon=False)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# --------------------------
# Example run (fig-like outputs)
# --------------------------

@_with_paper_rc
def run_example_with_cfg(
    cfg: RingConfig,
    cfg_name: str = "default",
    force_retrain: bool = False,
    show: bool = True,
    paper: bool = False,
):
    # baseline (pre-learning) parameters
    w0_init, U_init = init_params(cfg)

    cfg_name = cfg_name or "default"
    if cfg.snapshot_iters is not None:
        try:
            snapshot_iters = [int(v) for v in cfg.snapshot_iters]
        except Exception:
            snapshot_iters = list(cfg.snapshot_iters)
        snapshot_iters = _normalize_snapshot_iters(snapshot_iters, cfg.n_iter)
    else:
        snapshot_iters = _profile_snapshot_iters(cfg.n_iter, n_snapshots=5)
    res = get_fit_result(
        cfg,
        cfg_name=cfg_name,
        force_retrain=force_retrain,
        snapshot_iters=snapshot_iters,
        require_snapshots=True,
    )
    snapshot_iters = res.history.get("snapshot_iters", snapshot_iters)
    J_mean, J_se = evaluate_FI(cfg, res.w0, res.U, n_trials=6)
    print(f"Final FI estimate (per neuron per s): J = {J_mean:.3e} +/- {J_se:.1e} (SE)")

    fig_dir = _figure_dir(cfg_name, paper=paper)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paper_skip_fig_names = {
        "grad_profiles_snapshots_grid",
        "grad_profiles_snapshots_overlay",
        "phase_aligned_d",
        "phase_aligned_weff",
        "phase_aligned_EU",
        "phase_aligned_Ew0",
        "phase_aligned_sU",
        "correlogram_window_summary",
        "correlogram_window_summary_spontaneous",
        "grad_norm_history",
        "spike_rate_history",
        "param_mean_std_history",
        "grad_component_history",
        "grad_component_ratio_history",
    }

    def _save_fig(fig, name: str):
        if paper and (name in paper_skip_fig_names):
            print(f"Skipping figure in paper mode: {name}")
            import matplotlib.pyplot as plt

            plt.close(fig)
            return
        out_path = fig_dir / f"{name}.pdf"
        fig.savefig(out_path, **_savefig_kwargs(paper))
        print(f"Saved figure: {out_path}")
        if not show:
            import matplotlib.pyplot as plt

            plt.close(fig)

    # Plot
    import matplotlib.pyplot as plt
    zeta, order = centered_offset_angles(cfg.N)
    n_corr_trials = 8
    shuffle_mode = "roll"
    heatmap_lag_limit_s = 0.1
    if cfg.bin_dt <= 0.0:
        raise ValueError("bin_dt must be positive for correlogram plotting.")
    max_lag_bins = max(1, int(np.ceil(heatmap_lag_limit_s / cfg.bin_dt)))
    spikes_init_trials = sample_spikes_trials(cfg, w0_init, U_init, n_trials=n_corr_trials, seed=41)
    spikes_trials = sample_spikes_trials(cfg, res.w0, res.U, n_trials=n_corr_trials, seed=42)
    _, _, correlogram_init = correlogram_by_distance_trial_shuffle(
        spikes_init_trials, max_lag_bins=max_lag_bins, shuffle_mode=shuffle_mode
    )
    _, _, correlogram = correlogram_by_distance_trial_shuffle(
        spikes_trials, max_lag_bins=max_lag_bins, shuffle_mode=shuffle_mode
    )
    lags = np.arange(-max_lag_bins, max_lag_bins + 1) * cfg.bin_dt

    # Spontaneous activity (no external drive).
    cfg_spont = RingConfig(**asdict(cfg))
    cfg_spont.A = 0.0
    cfg_spont.h_bg = cfg.spont_h_bg
    spikes_init_trials_spont = sample_spikes_trials(
        cfg_spont, w0_init, U_init, n_trials=n_corr_trials, seed=141
    )
    spikes_trials_spont = sample_spikes_trials(
        cfg_spont, res.w0, res.U, n_trials=n_corr_trials, seed=142
    )
    _, _, correlogram_init_spont = correlogram_by_distance_trial_shuffle(
        spikes_init_trials_spont, max_lag_bins=max_lag_bins, shuffle_mode=shuffle_mode
    )
    _, _, correlogram_spont = correlogram_by_distance_trial_shuffle(
        spikes_trials_spont, max_lag_bins=max_lag_bins, shuffle_mode=shuffle_mode
    )
    lags_spont = np.arange(-max_lag_bins, max_lag_bins + 1) * cfg_spont.bin_dt


    w_eff_init = w0_init * U_init
    w_eff_learn = res.w0 * res.U
    fig_profiles, axes = plt.subplots(1, 3, figsize=_paper_figsize((12, 3.6), paper=paper), sharex=True)
    profile_specs = [
        (r"w0($\Delta z$)", w0_init[order], res.w0[order], "Baseline weight w0"),
        (r"U($\Delta z$)", U_init[order], res.U[order], "Release probability U"),
        (r"w_eff($\Delta z$)", w_eff_init[order], w_eff_learn[order], "Effective weight w0*U"),
    ]
    for ax, (ylabel, init_vals, learned_vals, title) in zip(axes, profile_specs):
        ax.plot(zeta, init_vals, linestyle="--", marker="o", linewidth=1, label="init")
        ax.plot(zeta, learned_vals, marker="o", linewidth=1, label="learned")
        ax.set_xlabel(r"offset $\Delta z$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
    fig_profiles.tight_layout()
    _save_fig(fig_profiles, "weight_profiles")

    snapshot_iters = res.history.get("snapshot_iters")
    snapshot_w0 = res.history.get("snapshot_w0")
    snapshot_U = res.history.get("snapshot_U")
    snapshot_gw0 = res.history.get("snapshot_gw0")
    snapshot_gU = res.history.get("snapshot_gU")
    if snapshot_iters is not None and snapshot_w0 is not None and snapshot_U is not None:
        snap_iters = np.asarray(snapshot_iters, dtype=int)
        snap_w0 = np.asarray(snapshot_w0, dtype=float)
        snap_U = np.asarray(snapshot_U, dtype=float)
        if snap_iters.size == 0:
            print("Skipping snapshot profile plots; no snapshots captured.")
        elif (
            snap_w0.ndim == 2
            and snap_U.ndim == 2
            and snap_w0.shape == snap_U.shape
            and snap_w0.shape[0] == snap_iters.size
        ):
            snap_order = np.argsort(snap_iters)
            snap_iters = snap_iters[snap_order]
            snap_w0 = snap_w0[snap_order]
            snap_U = snap_U[snap_order]
            snap_w_eff = snap_w0 * snap_U

            snap_w0_plot = snap_w0[:, order]
            snap_U_plot = snap_U[:, order]
            snap_w_eff_plot = snap_w_eff[:, order]

            n_rows = snap_iters.size
            fig_grid_h = 2.2 * n_rows
            fig_grid, axes_grid = plt.subplots(
                n_rows, 3, figsize=_paper_figsize((12, fig_grid_h), paper=paper), sharex=True
            )
            axes_grid = np.array(axes_grid).reshape(n_rows, 3)
            for r, it_val in enumerate(snap_iters):
                axes_grid[r, 0].plot(zeta, snap_w0_plot[r], color="tab:blue")
                axes_grid[r, 1].plot(zeta, snap_U_plot[r], color="tab:orange")
                axes_grid[r, 2].plot(zeta, snap_w_eff_plot[r], color="tab:green")
                axes_grid[r, 0].set_ylabel(f"iter {it_val}")
            axes_grid[0, 0].set_title(r"w0($\Delta z$)")
            axes_grid[0, 1].set_title(r"U($\Delta z$)")
            axes_grid[0, 2].set_title(r"w_eff($\Delta z$)")
            for ax in axes_grid[-1, :]:
                ax.set_xlabel(r"offset $\Delta z$")
            fig_grid.tight_layout()
            _save_fig(fig_grid, "weight_profiles_snapshots_grid")

            fig_overlay, axes_overlay = plt.subplots(
                1, 3, figsize=_paper_figsize((12, 3.6), paper=paper), sharex=True
            )
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_rows))
            overlay_specs = [
                (r"w0($\Delta z$)", w0_init[order], snap_w0_plot),
                (r"U($\Delta z$)", U_init[order], snap_U_plot),
                (r"w_eff($\Delta z$)", (w0_init * U_init)[order], snap_w_eff_plot),
            ]
            for ax, (ylabel, init_vals, snap_vals) in zip(axes_overlay, overlay_specs):
                ax.plot(zeta, init_vals, linestyle="--", color="black", label="init")
                for r, it_val in enumerate(snap_iters):
                    ax.plot(zeta, snap_vals[r], color=colors[r], linewidth=1, label=f"iter {it_val}")
                ax.set_xlabel(r"offset $\Delta z$")
                ax.set_ylabel(ylabel)
            axes_overlay[2].legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
            fig_overlay.tight_layout(rect=(0, 0, 0.9, 1))
            _save_fig(fig_overlay, "weight_profiles_snapshots_overlay")
        else:
            print("Skipping snapshot profile plots; snapshot shapes do not match.")
    else:
        print("Skipping snapshot profile plots; snapshot history missing.")

    # Gradient snapshots (gw0, gU)
    if snapshot_iters is not None and snapshot_gw0 is not None and snapshot_gU is not None:
        snap_iters = np.asarray(snapshot_iters, dtype=int)
        snap_gw0 = np.asarray(snapshot_gw0, dtype=float)
        snap_gU = np.asarray(snapshot_gU, dtype=float)
        if snap_iters.size == 0:
            print("Skipping gradient snapshot plots; no snapshots captured.")
        elif (
            snap_gw0.ndim == 2
            and snap_gU.ndim == 2
            and snap_gw0.shape == snap_gU.shape
            and snap_gw0.shape[0] == snap_iters.size
        ):
            snap_order = np.argsort(snap_iters)
            snap_iters = snap_iters[snap_order]
            snap_gw0 = snap_gw0[snap_order]
            snap_gU = snap_gU[snap_order]

            snap_gw0_plot = snap_gw0[:, order]
            snap_gU_plot = snap_gU[:, order]

            n_rows = snap_iters.size
            fig_grad_h = 2.2 * n_rows
            fig_grad_grid, axes_grad_grid = plt.subplots(
                n_rows, 2, figsize=_paper_figsize((10, fig_grad_h), paper=paper), sharex=True
            )
            axes_grad_grid = np.array(axes_grad_grid).reshape(n_rows, 2)
            for r, it_val in enumerate(snap_iters):
                axes_grad_grid[r, 0].plot(zeta, snap_gw0_plot[r], color="tab:blue")
                axes_grad_grid[r, 1].plot(zeta, snap_gU_plot[r], color="tab:orange")
                axes_grad_grid[r, 0].axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
                axes_grad_grid[r, 1].axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
                axes_grad_grid[r, 0].set_ylabel(f"iter {it_val}")
            axes_grad_grid[0, 0].set_title(r"grad w0($\Delta z$)")
            axes_grad_grid[0, 1].set_title(r"grad U($\Delta z$)")
            for ax in axes_grad_grid[-1, :]:
                ax.set_xlabel(r"offset $\Delta z$")
            fig_grad_grid.tight_layout()
            _save_fig(fig_grad_grid, "grad_profiles_snapshots_grid")

            fig_grad_overlay, axes_grad_overlay = plt.subplots(
                1, 2, figsize=_paper_figsize((10, 3.6), paper=paper), sharex=True
            )
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_rows))
            overlay_specs = [
                (r"grad w0($\Delta z$)", snap_gw0_plot),
                (r"grad U($\Delta z$)", snap_gU_plot),
            ]
            for ax, (ylabel, snap_vals) in zip(axes_grad_overlay, overlay_specs):
                ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
                for r, it_val in enumerate(snap_iters):
                    ax.plot(zeta, snap_vals[r], color=colors[r], linewidth=1, label=f"iter {it_val}")
                ax.set_xlabel(r"offset $\Delta z$")
                ax.set_ylabel(ylabel)
            axes_grad_overlay[1].legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
            fig_grad_overlay.tight_layout(rect=(0, 0, 0.9, 1))
            _save_fig(fig_grad_overlay, "grad_profiles_snapshots_overlay")
        else:
            print("Skipping gradient snapshot plots; snapshot shapes do not match.")
    else:
        print("Skipping gradient snapshot plots; gradient snapshot history missing.")

    # Heatmap of cross-correlogram vs offset/lag (init, learned, diff)
    corr_init_mat = correlogram_init[order, :]
    corr_learn_mat = correlogram[order, :]
    corr_diff_mat = corr_learn_mat - corr_init_mat
    # mask the delta-like peak at (zeta~0, tau=0) to avoid color scale collapse
    idx_z0 = int(np.argmin(np.abs(zeta)))
    idx_t0 = int(np.argmin(np.abs(lags)))

    def _mask_center(mat: np.ndarray) -> np.ndarray:
        m = mat.copy()
        m[idx_z0, idx_t0] = np.nan
        return m

    corr_init_plot = _mask_center(corr_init_mat)
    corr_learn_plot = _mask_center(corr_learn_mat)
    corr_diff_plot = _mask_center(corr_diff_mat)

    if paper:
        vmax = np.nanmax(np.abs(corr_learn_plot)) + 1e-12
        fig_hm, ax_hm = plt.subplots(1, 1, figsize=(5, 4))
        # zeta is already centered/sorted from centered_offset_angles
        extent = [lags[0], lags[-1], zeta[0], zeta[-1]]
        im1 = ax_hm.imshow(
            np.ma.masked_invalid(corr_learn_plot),
            aspect="auto",
            origin="lower",
            extent=extent,
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
        )
        ax_hm.set_xlabel("lag (s)")
        ax_hm.set_ylabel(r"offset $\Delta z$ (rad)")
        ax_hm.set_xlim(-heatmap_lag_limit_s, heatmap_lag_limit_s)
        fig_hm.tight_layout()
        fig_hm.colorbar(im1, ax=ax_hm, fraction=0.046, pad=0.04)
    else:
        vmax = np.nanmax([np.nanmax(np.abs(corr_init_plot)), np.nanmax(np.abs(corr_learn_plot))]) + 1e-12
        vmax_diff = np.nanmax(np.abs(corr_diff_plot)) + 1e-12

        fig_hm, axes_hm = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        # zeta is already centered/sorted from centered_offset_angles
        extent = [lags[0], lags[-1], zeta[0], zeta[-1]]
        im0 = axes_hm[0].imshow(
            np.ma.masked_invalid(corr_init_plot),
            aspect="auto",
            origin="lower",
            extent=extent,
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
        )
        axes_hm[0].set_title("init")
        im1 = axes_hm[1].imshow(
            np.ma.masked_invalid(corr_learn_plot),
            aspect="auto",
            origin="lower",
            extent=extent,
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
        )
        axes_hm[1].set_title("learned")
        im2 = axes_hm[2].imshow(np.ma.masked_invalid(corr_diff_plot), aspect="auto", origin="lower", extent=extent, vmin=-vmax_diff, vmax=vmax_diff, cmap="coolwarm")
        axes_hm[2].set_title("learned - init")
        for ax in axes_hm:
            ax.set_xlabel("lag (s)")
            ax.set_xlim(-heatmap_lag_limit_s, heatmap_lag_limit_s)
        axes_hm[0].set_ylabel(r"offset $\Delta z$ (rad)")
        fig_hm.suptitle("Cross-correlogram vs offset/lag (trial-shuffle corrected)")
        fig_hm.tight_layout(rect=(0, 0, 1, 0.92))
        # add colorbars
        fig_hm.colorbar(im0, ax=axes_hm[0], fraction=0.046, pad=0.04)
        fig_hm.colorbar(im1, ax=axes_hm[1], fraction=0.046, pad=0.04)
        fig_hm.colorbar(im2, ax=axes_hm[2], fraction=0.046, pad=0.04)
    _save_fig(fig_hm, "correlogram_heatmap")

    # Heatmap of cross-correlogram vs offset/lag (spontaneous, init, learned, diff)
    corr_init_spont_mat = correlogram_init_spont[order, :]
    corr_learn_spont_mat = correlogram_spont[order, :]
    corr_diff_spont_mat = corr_learn_spont_mat - corr_init_spont_mat
    idx_t0_spont = int(np.argmin(np.abs(lags_spont)))
    corr_init_spont_plot = corr_init_spont_mat.copy()
    corr_learn_spont_plot = corr_learn_spont_mat.copy()
    corr_diff_spont_plot = corr_diff_spont_mat.copy()
    corr_init_spont_plot[idx_z0, idx_t0_spont] = np.nan
    corr_learn_spont_plot[idx_z0, idx_t0_spont] = np.nan
    corr_diff_spont_plot[idx_z0, idx_t0_spont] = np.nan

    if paper:
        vmax_spont = np.nanmax(np.abs(corr_learn_spont_plot)) + 1e-12
        fig_hm_spont, ax_hm_spont = plt.subplots(1, 1, figsize=(5, 4))
        extent_spont = [lags_spont[0], lags_spont[-1], zeta[0], zeta[-1]]
        im1_spont = ax_hm_spont.imshow(
            np.ma.masked_invalid(corr_learn_spont_plot),
            aspect="auto",
            origin="lower",
            extent=extent_spont,
            vmin=-vmax_spont,
            vmax=vmax_spont,
            cmap="coolwarm",
        )
        ax_hm_spont.set_xlabel("lag (s)")
        ax_hm_spont.set_ylabel(r"offset $\Delta z$ (rad)")
        ax_hm_spont.set_xlim(-heatmap_lag_limit_s, heatmap_lag_limit_s)
        fig_hm_spont.tight_layout()
        fig_hm_spont.colorbar(im1_spont, ax=ax_hm_spont, fraction=0.046, pad=0.04)
    else:
        vmax_spont = np.nanmax(
            [np.nanmax(np.abs(corr_init_spont_plot)), np.nanmax(np.abs(corr_learn_spont_plot))]
        ) + 1e-12
        vmax_diff_spont = np.nanmax(np.abs(corr_diff_spont_plot)) + 1e-12

        fig_hm_spont, axes_hm_spont = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        extent_spont = [lags_spont[0], lags_spont[-1], zeta[0], zeta[-1]]
        im0_spont = axes_hm_spont[0].imshow(
            np.ma.masked_invalid(corr_init_spont_plot),
            aspect="auto",
            origin="lower",
            extent=extent_spont,
            vmin=-vmax_spont,
            vmax=vmax_spont,
            cmap="coolwarm",
        )
        axes_hm_spont[0].set_title("init")
        im1_spont = axes_hm_spont[1].imshow(
            np.ma.masked_invalid(corr_learn_spont_plot),
            aspect="auto",
            origin="lower",
            extent=extent_spont,
            vmin=-vmax_spont,
            vmax=vmax_spont,
            cmap="coolwarm",
        )
        axes_hm_spont[1].set_title("learned")
        im2_spont = axes_hm_spont[2].imshow(
            np.ma.masked_invalid(corr_diff_spont_plot),
            aspect="auto",
            origin="lower",
            extent=extent_spont,
            vmin=-vmax_diff_spont,
            vmax=vmax_diff_spont,
            cmap="coolwarm",
        )
        axes_hm_spont[2].set_title("learned - init")
        for ax in axes_hm_spont:
            ax.set_xlabel("lag (s)")
            ax.set_xlim(-heatmap_lag_limit_s, heatmap_lag_limit_s)
        axes_hm_spont[0].set_ylabel(r"offset $\Delta z$ (rad)")
        fig_hm_spont.suptitle(
            f"Cross-correlogram vs offset/lag (spontaneous, h_bg={cfg_spont.h_bg:g}, trial-shuffle corrected)"
        )
        fig_hm_spont.tight_layout(rect=(0, 0, 1, 0.92))
        fig_hm_spont.colorbar(im0_spont, ax=axes_hm_spont[0], fraction=0.046, pad=0.04)
        fig_hm_spont.colorbar(im1_spont, ax=axes_hm_spont[1], fraction=0.046, pad=0.04)
        fig_hm_spont.colorbar(im2_spont, ax=axes_hm_spont[2], fraction=0.046, pad=0.04)
    _save_fig(fig_hm_spont, "correlogram_heatmap_spontaneous")

    # Phase-aligned rate (all neurons/time, aligned to input phase)
    n_phase_bins = int(getattr(cfg, "snapshot_phase_bins", 60))
    if n_phase_bins < 1:
        n_phase_bins = 60
    phase, rate_init = phase_aligned_rate(spikes_init_trials, cfg, n_phase_bins=n_phase_bins)
    _, rate_learn = phase_aligned_rate(spikes_trials, cfg, n_phase_bins=n_phase_bins)

    fig_phase = plt.figure(figsize=_paper_figsize((6.4, 4.8), paper=paper, max_width=6.0))
    plt.plot(phase, rate_init, linestyle="--", label="init")
    plt.plot(phase, rate_learn, label="learned")
    plt.xlabel("input phase (rad)")
    plt.ylabel("rate (Hz)")
    if not paper:
        plt.title("Phase-aligned firing rate (all neurons/time)")
    plt.legend(frameon=False)
    _save_fig(fig_phase, "phase_aligned_rate")

    # Phase-aligned rate snapshots across training iterations
    snapshot_rate = res.history.get("snapshot_rate")
    snapshot_phase = res.history.get("snapshot_phase")
    if snapshot_iters is not None and snapshot_rate is not None:
        snap_iters = np.asarray(snapshot_iters, dtype=int)
        snap_rate = np.asarray(snapshot_rate, dtype=float)
        if snapshot_phase is not None:
            snap_phase = np.asarray(snapshot_phase, dtype=float)
        else:
            snap_phase = phase
        if (
            snap_iters.size > 0
            and snap_rate.ndim == 2
            and snap_phase.ndim == 1
            and snap_rate.shape[0] == snap_iters.size
            and snap_rate.shape[1] == snap_phase.size
        ):
            snap_order = np.argsort(snap_iters)
            snap_iters = snap_iters[snap_order]
            snap_rate = snap_rate[snap_order]
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, snap_iters.size))

            fig_snap_rate = plt.figure(figsize=_paper_figsize((6.4, 4.8), paper=paper, max_width=6.0))
            plt.plot(snap_phase, rate_init, linestyle="--", color="black", label="init")
            for r, it_val in enumerate(snap_iters):
                plt.plot(snap_phase, snap_rate[r], color=colors[r], label=f"iter {it_val}")
            plt.xlabel("input phase (rad)")
            plt.ylabel("rate (Hz)")
            if not paper:
                plt.title("Phase-aligned firing rate snapshots")
            plt.legend(frameon=False, ncol=1)
            _save_fig(fig_snap_rate, "phase_aligned_rate_snapshots")
        else:
            print("Skipping phase-aligned rate snapshots; snapshot shapes do not match.")
    else:
        print("Skipping phase-aligned rate snapshots; snapshot history missing.")

    # Phase-aligned synapse variables per offset
    n_syn_trials = n_corr_trials
    syn_init = phase_aligned_synapse_vars(
        w0_init, U_init, cfg, n_trials=n_syn_trials, seed=51, n_phase_bins=n_phase_bins
    )
    syn_learn = phase_aligned_synapse_vars(
        res.w0, res.U, cfg, n_trials=n_syn_trials, seed=52, n_phase_bins=n_phase_bins
    )
    phase_syn = syn_init["phase"]
    zeta_targets = np.arange(-np.pi, np.pi, np.pi / 5.0)
    target_idx = []
    for target in zeta_targets:
        idx = int(np.argmin(np.abs(zeta - target)))
        if idx not in target_idx:
            target_idx.append(idx)
    target_idx = np.array(target_idx, dtype=int)
    zeta_sel = zeta[target_idx]
    k_sel = order[target_idx]
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(k_sel)))

    def _plot_phase_offsets(data_init: np.ndarray, data_learn: np.ndarray,
                            ylabel: str, title: str, fig_name: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=_paper_figsize((10, 4), paper=paper), sharex=True, sharey=True)
        for color, k, zlab in zip(colors, k_sel, zeta_sel):
            axes[0].plot(phase_syn, data_init[k], color=color, linewidth=1.2, label=rf"$\Delta z\approx{zlab:+.2f}$")
            axes[1].plot(phase_syn, data_learn[k], color=color, linewidth=1.2, label=rf"$\Delta z\approx{zlab:+.2f}$")
        axes[0].set_title(f"{title} \n (initial)")
        axes[1].set_title(f"{title} \n (learned)")
        for ax in axes:
            ax.set_xlabel("input phase (rad)")
            ax.set_ylabel(ylabel)
        axes[1].legend(frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout(rect=(0, 0, 0.9, 1))
        _save_fig(fig, fig_name)

    _plot_phase_offsets(syn_init["d"], syn_learn["d"], "d", "Phase-aligned depression", "phase_aligned_d")
    _plot_phase_offsets(
        syn_init["w_eff"], syn_learn["w_eff"], "w_eff", "Phase-aligned effective weight", "phase_aligned_weff"
    )
    _plot_phase_offsets(
        syn_init["E_U"], syn_learn["E_U"], "E_U", r"Phase-aligned eligibility $E_U$", "phase_aligned_EU"
    )
    _plot_phase_offsets(
        syn_init["E_w0"], syn_learn["E_w0"], "E_w0", r"Phase-aligned eligibility $E_{w0}$", "phase_aligned_Ew0"
    )
    _plot_phase_offsets(syn_init["sU"], syn_learn["sU"], "sU", "Phase-aligned auxiliary sU", "phase_aligned_sU")

    # Summary stats around zero lag
    tau0 = 5.0 * cfg.tau_s
    summary_init = correlogram_window_summary(correlogram_init, lags, tau0)
    summary_learn = correlogram_window_summary(correlogram, lags, tau0)
    summary_diff = correlogram_window_summary(correlogram - correlogram_init, lags, tau0)
    idx_k0 = 0  # k=0 corresponds to self-correlation (offset zeta=0)
    for key in summary_init.keys():
        summary_init[key] = summary_init[key].copy()
        summary_learn[key] = summary_learn[key].copy()
        summary_diff[key] = summary_diff[key].copy()
        summary_init[key][idx_k0] = np.nan
        summary_learn[key][idx_k0] = np.nan
        summary_diff[key][idx_k0] = np.nan

    # Smooth summary vs offset with a circular moving average and plot at full resolution.
    zeta_smooth_window = 7  # points
    zeta_plot = zeta
    idx_z0 = int(np.argmin(np.abs(zeta)))

    def _circular_moving_average(values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return values.copy()
        if window % 2 == 0:
            raise ValueError("window must be odd for symmetric smoothing")
        half = window // 2
        values_wrap = np.concatenate([values[-half:], values, values[:half]])
        mask = np.isfinite(values_wrap)
        values_filled = np.where(mask, values_wrap, 0.0)
        kernel = np.ones(window, dtype=float)
        numer = np.convolve(values_filled, kernel, mode="valid")
        denom = np.convolve(mask.astype(float), kernel, mode="valid")
        out = numer / denom
        out[denom == 0] = np.nan
        return out

    summary_init_smooth = {}
    summary_learn_smooth = {}
    summary_diff_smooth = {}
    for key in summary_init.keys():
        summary_init_smooth[key] = _circular_moving_average(summary_init[key][order], zeta_smooth_window)
        summary_learn_smooth[key] = _circular_moving_average(summary_learn[key][order], zeta_smooth_window)
        summary_diff_smooth[key] = _circular_moving_average(summary_diff[key][order], zeta_smooth_window)
        summary_init_smooth[key][idx_z0] = np.nan
        summary_learn_smooth[key][idx_z0] = np.nan
        summary_diff_smooth[key][idx_z0] = np.nan

    # EPSP-weighted positive-lag profile C_+^kappa(Delta z).
    cplus_init = correlogram_epsp_weighted_positive_lag(correlogram_init, lags, cfg.tau_s, tau_max=tau0)
    cplus_learn = correlogram_epsp_weighted_positive_lag(correlogram, lags, cfg.tau_s, tau_max=tau0)
    cplus_diff = correlogram_epsp_weighted_positive_lag(
        correlogram - correlogram_init, lags, cfg.tau_s, tau_max=tau0
    )
    cplus_init = cplus_init.copy()
    cplus_learn = cplus_learn.copy()
    cplus_diff = cplus_diff.copy()
    cplus_init_smooth = _circular_moving_average(cplus_init[order], zeta_smooth_window)
    cplus_learn_smooth = _circular_moving_average(cplus_learn[order], zeta_smooth_window)
    cplus_diff_smooth = _circular_moving_average(cplus_diff[order], zeta_smooth_window)

    fig_cplus, ax_cplus = plt.subplots(figsize=_paper_figsize((6.4, 4.2), paper=paper, max_width=6.0))
    ax_cplus.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    ax_cplus.plot(zeta_plot, cplus_init_smooth, linestyle="--", label="init")
    ax_cplus.plot(zeta_plot, cplus_learn_smooth, label="learned")
    ax_cplus.plot(zeta_plot, cplus_diff_smooth, linestyle=":", label="learned - init")
    ax_cplus.set_xlabel(r"offset $\Delta z$")
    ax_cplus.set_ylabel(r"$C_+^\kappa(\Delta z)$")
    if not paper:
        ax_cplus.set_title(f"EPSP-weighted positive-lag correlogram (0 < lag <= {tau0:.3f} s)")
    ax_cplus.legend(frameon=False)
    fig_cplus.tight_layout()
    _save_fig(fig_cplus, "correlogram_positive_lag_weighted")

    fig_sum, axes_sum = plt.subplots(1, 3, figsize=_paper_figsize((12, 4), paper=paper), sharex=True)
    metrics = [
        ("peak", "peak"),
        ("area", "area (corr*s)"),
        ("asym", "asym (corr*s)"),
    ]
    for ax, (key, ylabel) in zip(axes_sum, metrics):
        init_vals = summary_init_smooth[key]
        learn_vals = summary_learn_smooth[key]
        diff_vals = summary_diff_smooth[key]
        ax.plot(zeta_plot, init_vals, linestyle="--", label="init")
        ax.plot(zeta_plot, learn_vals, label="learned")
        ax.plot(zeta_plot, diff_vals, linestyle=":", label="learned - init")
        ax.set_xlabel(r"offset $\Delta z$")
        ax.set_ylabel(ylabel)
    axes_sum[0].legend(frameon=False)
    fig_sum.suptitle(f"Correlogram summary (trial-shuffle corrected, |tau| < {tau0:.3f} s)")
    fig_sum.tight_layout(rect=(0, 0, 1, 0.92))
    _save_fig(fig_sum, "correlogram_window_summary")

    # Summary stats around zero lag (spontaneous activity).
    summary_init_spont = correlogram_window_summary(correlogram_init_spont, lags_spont, tau0)
    summary_learn_spont = correlogram_window_summary(correlogram_spont, lags_spont, tau0)
    summary_diff_spont = correlogram_window_summary(correlogram_spont - correlogram_init_spont, lags_spont, tau0)
    for key in summary_init_spont.keys():
        summary_init_spont[key] = summary_init_spont[key].copy()
        summary_learn_spont[key] = summary_learn_spont[key].copy()
        summary_diff_spont[key] = summary_diff_spont[key].copy()
        summary_init_spont[key][idx_k0] = np.nan
        summary_learn_spont[key][idx_k0] = np.nan
        summary_diff_spont[key][idx_k0] = np.nan

    summary_init_spont_smooth = {}
    summary_learn_spont_smooth = {}
    summary_diff_spont_smooth = {}
    for key in summary_init_spont.keys():
        summary_init_spont_smooth[key] = _circular_moving_average(summary_init_spont[key][order], zeta_smooth_window)
        summary_learn_spont_smooth[key] = _circular_moving_average(summary_learn_spont[key][order], zeta_smooth_window)
        summary_diff_spont_smooth[key] = _circular_moving_average(summary_diff_spont[key][order], zeta_smooth_window)
        summary_init_spont_smooth[key][idx_z0] = np.nan
        summary_learn_spont_smooth[key][idx_z0] = np.nan
        summary_diff_spont_smooth[key][idx_z0] = np.nan

    cplus_init_spont = correlogram_epsp_weighted_positive_lag(
        correlogram_init_spont, lags_spont, cfg.tau_s, tau_max=tau0
    )
    cplus_learn_spont = correlogram_epsp_weighted_positive_lag(
        correlogram_spont, lags_spont, cfg.tau_s, tau_max=tau0
    )
    cplus_diff_spont = correlogram_epsp_weighted_positive_lag(
        correlogram_spont - correlogram_init_spont, lags_spont, cfg.tau_s, tau_max=tau0
    )
    cplus_init_spont = cplus_init_spont.copy()
    cplus_learn_spont = cplus_learn_spont.copy()
    cplus_diff_spont = cplus_diff_spont.copy()
    cplus_init_spont_smooth = _circular_moving_average(cplus_init_spont[order], zeta_smooth_window)
    cplus_learn_spont_smooth = _circular_moving_average(cplus_learn_spont[order], zeta_smooth_window)
    cplus_diff_spont_smooth = _circular_moving_average(cplus_diff_spont[order], zeta_smooth_window)

    fig_cplus_spont, ax_cplus_spont = plt.subplots(
        figsize=_paper_figsize((6.4, 4.2), paper=paper, max_width=6.0)
    )
    ax_cplus_spont.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    ax_cplus_spont.plot(zeta_plot, cplus_init_spont_smooth, linestyle="--", label="init")
    ax_cplus_spont.plot(zeta_plot, cplus_learn_spont_smooth, label="learned")
    ax_cplus_spont.plot(zeta_plot, cplus_diff_spont_smooth, linestyle=":", label="learned - init")
    ax_cplus_spont.set_xlabel(r"offset $\Delta z$")
    ax_cplus_spont.set_ylabel(r"$C_+^\kappa(\Delta z)$")
    if not paper:
        ax_cplus_spont.set_title(
            f"EPSP-weighted positive-lag correlogram (spontaneous, h_bg={cfg_spont.h_bg:g})"
        )
    ax_cplus_spont.legend(frameon=False)
    fig_cplus_spont.tight_layout()
    _save_fig(fig_cplus_spont, "correlogram_positive_lag_weighted_spontaneous")

    fig_sum_spont, axes_sum_spont = plt.subplots(
        1, 3, figsize=_paper_figsize((12, 4), paper=paper), sharex=True
    )
    for ax, (key, ylabel) in zip(axes_sum_spont, metrics):
        init_vals = summary_init_spont_smooth[key]
        learn_vals = summary_learn_spont_smooth[key]
        diff_vals = summary_diff_spont_smooth[key]
        ax.plot(zeta_plot, init_vals, linestyle="--", label="init")
        ax.plot(zeta_plot, learn_vals, label="learned")
        ax.plot(zeta_plot, diff_vals, linestyle=":", label="learned - init")
        ax.set_xlabel(r"offset $\Delta z$")
        ax.set_ylabel(ylabel)
    axes_sum_spont[0].legend(frameon=False)
    fig_sum_spont.suptitle(
        f"Correlogram summary (spontaneous, h_bg={cfg_spont.h_bg:g}, trial-shuffle corrected, |tau| < {tau0:.3f} s)"
    )
    fig_sum_spont.tight_layout(rect=(0, 0, 1, 0.92))
    _save_fig(fig_sum_spont, "correlogram_window_summary_spontaneous")

    fig_train, axes_train = plt.subplots(1, 3, figsize=_paper_figsize((12, 3.6), paper=paper), sharex=True)
    n_steps = int(np.round(cfg.T / cfg.dt))
    duration = n_steps * cfg.dt
    norm = float(cfg.N) * duration
    if norm <= 0.0:
        raise ValueError("Fisher information normalization requires positive duration and N.")
    fi_curve = res.history["fi_rate"] if "fi_rate" in res.history else res.history["J"] / norm
    rate_penalty_curve = res.history.get("rate_penalty")
    total_loss_curve = res.history.get("total_loss")
    if rate_penalty_curve is None or total_loss_curve is None:
        rate_mean_curve = res.history.get("rate_mean")
        if rate_mean_curve is None and "spike_rate" in res.history:
            rate_mean_curve = res.history["spike_rate"]
        if rate_mean_curve is not None and cfg.rate_target is not None:
            rate_error_curve = rate_mean_curve - float(cfg.rate_target)
            rate_penalty_curve = 0.5 * float(cfg.rate_reg_lambda) * (rate_error_curve ** 2)
            total_loss_curve = fi_curve - rate_penalty_curve

    axes_train[0].plot(fi_curve)
    axes_train[0].set_xlabel("iteration")
    axes_train[0].set_ylabel("FI (per neuron per s)")
    axes_train[0].set_title("FI")

    if rate_penalty_curve is not None:
        axes_train[1].plot(rate_penalty_curve)
        axes_train[1].set_xlabel("iteration")
        axes_train[1].set_ylabel("rate penalty")
    else:
        axes_train[1].text(0.5, 0.5, "rate penalty\nmissing", ha="center", va="center")
        axes_train[1].set_xlabel("iteration")
    axes_train[1].set_title("Rate error term")

    if total_loss_curve is not None:
        axes_train[2].plot(total_loss_curve)
        axes_train[2].set_xlabel("iteration")
        axes_train[2].set_ylabel("FI - rate penalty")
    else:
        axes_train[2].text(0.5, 0.5, "total loss\nmissing", ha="center", va="center")
        axes_train[2].set_xlabel("iteration")
    axes_train[2].set_title("Total loss")

    fig_train.tight_layout()
    _save_fig(fig_train, "training_curve")

    grad_norm_keys = ["gnorm_w0", "gnorm_U"]
    if all(k in res.history for k in grad_norm_keys):
        fig_gnorm, ax_gnorm = plt.subplots(figsize=(6, 4))
        it = np.arange(len(res.history["gnorm_w0"]))
        ax_gnorm.plot(it, res.history["gnorm_w0"], label="||grad w0||", linewidth=1.0)
        ax_gnorm.plot(it, res.history["gnorm_U"], label="||grad U||", linewidth=1.0)
        ax_gnorm.set_xlabel("iteration")
        ax_gnorm.set_ylabel("gradient norm")
        ax_gnorm.set_yscale("log")
        ax_gnorm.set_title("Gradient norm during training")
        ax_gnorm.legend(frameon=False)
        fig_gnorm.tight_layout()
        _save_fig(fig_gnorm, "grad_norm_history")
    else:
        missing = [k for k in grad_norm_keys if k not in res.history]
        print(f"Skipping grad-norm plot; missing history keys: {missing}")

    spike_keys = ["spike_rate", "spike_rate_h0"]
    if all(k in res.history for k in spike_keys):
        fig_spk, ax_spk = plt.subplots(figsize=(6, 4))
        it = np.arange(len(res.history["spike_rate"]))
        ax_spk.plot(it, res.history["spike_rate"], label="total")
        ax_spk.plot(it, res.history["spike_rate_h0"], label="h_p = 0")
        ax_spk.set_xlabel("iteration")
        ax_spk.set_ylabel("spike rate (Hz per neuron)")
        ax_spk.set_title("Spike rate during training")
        ax_spk.legend(frameon=False)
        fig_spk.tight_layout()
        _save_fig(fig_spk, "spike_rate_history")
    else:
        missing = [k for k in spike_keys if k not in res.history]
        print(f"Skipping spike-rate plot; missing history keys: {missing}")

    param_stat_keys = ["w0_mean", "w0_std", "U_mean", "U_std"]
    if all(k in res.history for k in param_stat_keys):
        w0_l2_hist = res.history.get("w0_l2")
        has_w0_l2 = w0_l2_hist is not None and len(w0_l2_hist) == len(res.history["w0_mean"])
        n_param_rows = 3 if has_w0_l2 else 2
        fig_param, axes_param = plt.subplots(n_param_rows, 1, figsize=(6, 8 if has_w0_l2 else 6), sharex=True)
        it = np.arange(len(res.history["w0_mean"]))

        w0_mean = res.history["w0_mean"]
        w0_std = res.history["w0_std"]
        axes_param[0].plot(it, w0_mean, color="tab:blue", label="mean")
        axes_param[0].fill_between(it, w0_mean - w0_std, w0_mean + w0_std, color="tab:blue", alpha=0.2, label="+/-1 std")
        axes_param[0].set_ylabel(r"w0($\Delta z$)")
        axes_param[0].set_title("w0 mean +/- std")
        axes_param[0].legend(frameon=False)

        U_mean = res.history["U_mean"]
        U_std = res.history["U_std"]
        axes_param[1].plot(it, U_mean, color="tab:orange", label="mean")
        axes_param[1].fill_between(it, U_mean - U_std, U_mean + U_std, color="tab:orange", alpha=0.2, label="+/-1 std")
        axes_param[1].set_ylabel(r"U($\Delta z$)")
        axes_param[1].set_title("U mean +/- std")
        axes_param[1].legend(frameon=False)

        if has_w0_l2:
            w0_l2 = np.asarray(w0_l2_hist, dtype=float)
            axes_param[2].plot(it, w0_l2, color="tab:green", label=r"$||w_0||_2$")
            if getattr(cfg, "constrain_w0_l2", False):
                if getattr(cfg, "w0_l2_rms_max", None) is None:
                    cap_l2 = abs(float(cfg.w0_init)) / np.sqrt(float(cfg.N))
                else:
                    cap_l2 = float(cfg.w0_l2_rms_max) * np.sqrt(float(cfg.N))
                if np.isfinite(cap_l2):
                    axes_param[2].axhline(cap_l2, color="tab:red", linestyle="--", linewidth=1.0, label="constraint")
            axes_param[2].set_ylabel(r"$||w_0||_2$")
            axes_param[2].set_title("w0 L2 norm")
            axes_param[2].legend(frameon=False)
            axes_param[2].set_xlabel("iteration")
        else:
            axes_param[1].set_xlabel("iteration")

        fig_param.tight_layout()
        _save_fig(fig_param, "param_mean_std_history")
    else:
        missing = [k for k in param_stat_keys if k not in res.history]
        print(f"Skipping w0/U mean-std plot; missing history keys: {missing}")

    comp_keys = ["gnorm_w0_A", "gnorm_w0_score", "gnorm_U_A", "gnorm_U_score"]
    if all(k in res.history for k in comp_keys):
        fig_comp, axes_comp = plt.subplots(1, 2, figsize=_paper_figsize((10, 4), paper=paper), sharex=True)
        it = np.arange(len(res.history["gnorm_w0_A"]))
        axes_comp[0].plot(it, res.history["gnorm_w0_A"], label="direct term", linewidth=1.0)
        axes_comp[0].plot(it, res.history["gnorm_w0_score"], label="score term", linewidth=1.0)
        axes_comp[0].set_title("grad components (w0)")
        axes_comp[0].set_xlabel("iteration")
        axes_comp[0].set_ylabel("component norm")
        axes_comp[0].set_yscale("log")
        axes_comp[0].legend(frameon=False)

        axes_comp[1].plot(it, res.history["gnorm_U_A"], label="direct term", linewidth=1.0)
        axes_comp[1].plot(it, res.history["gnorm_U_score"], label="score term", linewidth=1.0)
        axes_comp[1].set_title("grad components (U)")
        axes_comp[1].set_xlabel("iteration")
        axes_comp[1].set_ylabel("component norm")
        axes_comp[1].set_yscale("log")
        axes_comp[1].legend(frameon=False)

        fig_comp.suptitle("Gradient component magnitudes")
        fig_comp.tight_layout(rect=(0, 0, 1, 0.93))
        _save_fig(fig_comp, "grad_component_history")

        ratio_w0 = np.divide(
            res.history["gnorm_w0_score"],
            res.history["gnorm_w0_A"],
            out=np.full_like(res.history["gnorm_w0_score"], np.nan),
            where=res.history["gnorm_w0_A"] > 0,
        )
        ratio_U = np.divide(
            res.history["gnorm_U_score"],
            res.history["gnorm_U_A"],
            out=np.full_like(res.history["gnorm_U_score"], np.nan),
            where=res.history["gnorm_U_A"] > 0,
        )
        fig_ratio, ax_ratio = plt.subplots(figsize=(6, 4))
        ax_ratio.plot(it, ratio_w0, label="w0 score/direct", linewidth=1.0)
        ax_ratio.plot(it, ratio_U, label="U score/direct", linewidth=1.0)
        ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_ratio.set_xlabel("iteration")
        ax_ratio.set_ylabel("||score|| / ||direct||")
        ax_ratio.set_yscale("log")
        ax_ratio.set_title("Gradient component ratio")
        ax_ratio.legend(frameon=False)
        fig_ratio.tight_layout()
        _save_fig(fig_ratio, "grad_component_ratio_history")
    else:
        missing = [k for k in comp_keys if k not in res.history]
        print(f"Skipping grad-component plot; missing history keys: {missing}")

    if show:
        plt.show()


@_with_paper_rc
def run_replay_demo_with_cfg(
    cfg: RingConfig,
    rp: Optional[ReplayConfig] = None,
    cfg_name: str = "default",
    force_retrain: bool = False,
    show: bool = True,
    paper: bool = False,
):
    """
    Train (or load) parameters, then simulate replay under background-only drive.
    """
    import matplotlib.pyplot as plt

    cfg_name = cfg_name or "default"
    res = get_fit_result(cfg, cfg_name=cfg_name, force_retrain=force_retrain)

    if rp is None:
        rp = ReplayConfig()

    replay_cfg_path = save_replay_config(cfg_name, rp)
    print(f"Saved replay config to {replay_cfg_path}")

    replay = simulate_replay_activity(cfg, res.w0, res.U, rp)
    print(f"Replay phase velocities (rad/s) over {rp.n_trials} trial(s): {replay['phase_velocity']}")

    fig_dir = _figure_dir(cfg_name, paper=paper)
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig(fig, name: str):
        out_path = fig_dir / f"{name}.pdf"
        fig.savefig(out_path, **_savefig_kwargs(paper))
        print(f"Saved figure: {out_path}")
        if not show:
            plt.close(fig)

    fig_replay = plot_replay_heatmap(
        replay,
        trial=0,
        cue_window=(rp.t_start + rp.cue_start, rp.t_start + rp.cue_start + rp.cue_duration),
        title="Replay after exact-gradient training",
        show=False,
        paper=paper,
        plot_t_start=rp.plot_t_start,
        plot_t_end=rp.plot_t_end,
    )
    _save_fig(fig_replay, "replay_heatmap")
    if show:
        plt.show()


def run_example(cfg_name: str = "default", force_retrain: bool = False, paper: bool = False):
    cfg = RingConfig()  # use defaults
    run_example_with_cfg(
        cfg,
        cfg_name=cfg_name,
        force_retrain=force_retrain,
        show=True,
        paper=paper,
    )


def run_replay_demo(cfg_name: str = "default", force_retrain: bool = False, paper: bool = False):
    """
    Train (or load) parameters, then simulate replay under background-only drive.
    """
    cfg = RingConfig()  # use defaults
    run_replay_demo_with_cfg(
        cfg,
        rp=None,
        cfg_name=cfg_name,
        force_retrain=force_retrain,
        show=True,
        paper=paper,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ring STP exact-gradient demos.")
    parser.add_argument(
        "--mode",
        choices=["all", "example", "replay"],
        default="all",
        help="Select which demo to run (default: all).",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Save paper-ready figures under saved_runs/<cfg>/paper and adjust plotting.",
    )
    parser.add_argument("--force-retrain", action="store_true", help="Ignore cached fits and retrain.")
    parser.add_argument("--cfg-name", type=str, default=None, help="Name for cache/save (optional).")
    args = parser.parse_args()

    # Use the same default when running both demos to share caches.
    if args.mode == "all":
        run_example(cfg_name=args.cfg_name or "default", force_retrain=args.force_retrain, paper=args.paper)
        run_replay_demo(cfg_name=args.cfg_name or "default", force_retrain=False, paper=args.paper) # don't retrain again
    elif args.mode == "example":
        run_example(cfg_name=args.cfg_name or "default", force_retrain=args.force_retrain, paper=args.paper)
    elif args.mode == "replay":
        run_replay_demo(cfg_name=args.cfg_name or "default", force_retrain=args.force_retrain, paper=args.paper)
