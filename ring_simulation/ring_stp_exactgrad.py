
"""
Core simulation and analysis for exact-gradient learning on a ring network with
associative presynaptic STP (STD-only), matching the "Traveling Waves on a Circle"
case study.

Implements:
- Stochastic spiking (inhomogeneous Poisson) with intensity rho_i(t)=g(u_i(t))
- g(u) can be swapped (e.g., exp escape-rate or sigmoid), with derivative terms
  defined in one place for the exact-gradient estimator.
- Traveling-wave external input h(z,t)=A [cos(omega t - z) - cos(theta_c)]_+
- Tsodyks-Markram STD-only synapses: w_ij(t)=w0(ζ) U(ζ) d_ij(t), with offset ζ = (pre - post) on the ring
- Exact gradient estimator via score-function identity + eligibility traces
  (see manuscript main text and Appendix for eligibility updates)
- Parameterization:
    w0: initialized uniformly to c/N and updated by SGD (optional momentum/Nesterov)
         with optional block-wise scalar adaptive LR
    U : updated by SGD (optional momentum/Nesterov) and clipped to [U_lo, U_hi],
         with optional block-wise scalar adaptive LR
    Optional hard constraints (projection):
        w0: sum constraint (target set per config), optional L2-ball cap
        U : box + sum constraint (target set per config)

Also provides:
- Correlation/phase-alignment analysis helpers
- Fisher information evaluation at convergence (non-perturbative regime)
- Background-only replay simulation

Plotting and demo runners live in ring_stp_plotting.py.

No external dependencies beyond numpy (Numba optional for JIT).
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List
import json
import multiprocessing as mp
import os
import sys
import numpy as np
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

# Set RING_STP_USE_NUMBA=0 to force the non-JIT path.
_USE_NUMBA = _NUMBA_AVAILABLE and os.getenv("RING_STP_USE_NUMBA", "1").lower() not in ("0", "false", "no")
# Set RING_STP_CORR_FFT=0 to use the legacy non-FFT correlation path.
_USE_CORR_FFT = os.getenv("RING_STP_CORR_FFT", "1").lower() not in ("0", "false", "no")
_DEFAULT_KL_RATE_TARGET_HZ = 0.5
_PHASE_ASYMMETRY_METRIC_KEYS: Tuple[str, ...] = (
    "peak_phase",
    "centroid_phase",
    "skewness",
    "area_index",
    "odd_ratio",
)


def _normalize_rate_reg_type(rate_reg_type: Any) -> str:
    s = str(rate_reg_type).strip().lower()
    aliases = {
        "l2": "l2_mean",
        "l2mean": "l2_mean",
        "l2-rate": "l2_mean",
        "l2_rate": "l2_mean",
        "kl_div": "kl",
        "kl-div": "kl",
        "kl_divergence": "kl",
    }
    s = aliases.get(s, s)
    if s not in ("l2_mean", "kl"):
        raise ValueError(f"rate_reg_type must be one of {{'l2_mean', 'kl'}}, got {rate_reg_type!r}")
    return s


# --------------------------
# Optimizer: SGD (+ optional momentum / Nesterov)
# --------------------------

def sgd_step(param: np.ndarray, grad: np.ndarray, lr: float,
             update_clip: Optional[float] = None) -> np.ndarray:
    delta = lr * grad

    if update_clip is not None:
        dnorm = float(np.linalg.norm(delta))
        dmax = float(update_clip) * (float(np.linalg.norm(param)) + 1e-12)
        if dnorm > dmax:
            delta = delta * (dmax / (dnorm + 1e-12))

    return param + delta


def sgd_momentum_step(
    param: np.ndarray,
    grad: np.ndarray,
    velocity: np.ndarray,
    lr: float,
    momentum: float = 0.0,
    nesterov: bool = True,
    update_clip: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SGD step with optional momentum / Nesterov acceleration.
    - momentum=0 reproduces plain SGD exactly.
    - Update is additive (gradient ascent), matching this codebase's objective sign.
    """
    mu = float(momentum)
    if not np.isfinite(mu) or mu < 0.0 or mu >= 1.0:
        raise ValueError(f"momentum must satisfy 0 <= momentum < 1, got {momentum}")

    v_new = mu * velocity + grad
    if nesterov and mu > 0.0:
        direction = grad + mu * v_new
    else:
        direction = v_new

    delta = lr * direction
    if update_clip is not None:
        dnorm = float(np.linalg.norm(delta))
        dmax = float(update_clip) * (float(np.linalg.norm(param)) + 1e-12)
        if dnorm > dmax:
            delta = delta * (dmax / (dnorm + 1e-12))

    return param + delta, v_new


def blockwise_rmsprop_scale(
    grad: np.ndarray,
    ema_g2_prev: float,
    beta: float,
    eps: float,
    scale_min: float,
    scale_max: float,
) -> Tuple[float, float]:
    """
    Scalar RMSProp scale for one parameter block.
    The returned scale is spatially uniform within the block.
    """
    if grad.size == 0:
        g2 = 0.0
    else:
        g2 = float(np.mean(np.square(grad)))
    ema_g2 = beta * float(ema_g2_prev) + (1.0 - beta) * g2
    if (not np.isfinite(ema_g2)) or ema_g2 < 0.0:
        ema_g2 = max(g2, 0.0)
    scale = 1.0 / np.sqrt(ema_g2 + eps)
    scale = float(np.clip(scale, scale_min, scale_max))
    return scale, float(ema_g2)


# --------------------------
# Projection helpers (hard constraints)
# --------------------------

def project_sum(x: np.ndarray, target_sum: float) -> np.ndarray:
    """
    Euclidean projection onto the hyperplane sum(x)=target_sum.
    """
    if x.size == 0:
        return x
    shift = (float(target_sum) - float(np.sum(x))) / float(x.size)
    return x + shift


def project_l2_ball(x: np.ndarray, l2_radius: float, eps: float = 1e-12) -> np.ndarray:
    """
    Euclidean projection onto the L2 ball {x | ||x||_2 <= l2_radius}.
    """
    if x.size == 0:
        return x
    l2_radius = float(l2_radius)
    if l2_radius < 0.0:
        raise ValueError(f"project_l2_ball: l2_radius must be >= 0, got {l2_radius}")
    x_norm = float(np.linalg.norm(x))
    if x_norm <= l2_radius + eps:
        return x
    if x_norm <= eps:
        return np.zeros_like(x, dtype=float)
    return x * (l2_radius / (x_norm + eps))


def project_sum_l2_ball(
    x: np.ndarray,
    target_sum: float,
    l2_radius: float,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Euclidean projection onto {x | sum(x)=target_sum, ||x||_2 <= l2_radius}.
    """
    if x.size == 0:
        return x
    n = int(x.size)
    target_sum = float(target_sum)
    l2_radius = float(l2_radius)
    if l2_radius < 0.0:
        raise ValueError(f"project_sum_l2_ball: l2_radius must be >= 0, got {l2_radius}")

    min_feasible_norm = abs(target_sum) / np.sqrt(float(n))
    if l2_radius + tol < min_feasible_norm:
        raise ValueError(
            "project_sum_l2_ball: infeasible constraints "
            f"(target_sum={target_sum}, l2_radius={l2_radius}, min_feasible={min_feasible_norm})"
        )

    x_h = project_sum(x, target_sum)
    if float(np.linalg.norm(x_h)) <= l2_radius + tol:
        return x_h

    mean_target = target_sum / float(n)
    v = x_h - mean_target
    v_norm = float(np.linalg.norm(v))
    radius_sq = max(l2_radius * l2_radius - float(n) * mean_target * mean_target, 0.0)
    v_radius = float(np.sqrt(radius_sq))
    if v_norm <= v_radius + tol:
        return x_h
    if v_norm <= tol:
        return np.full_like(x_h, mean_target, dtype=float)
    return mean_target + v * (v_radius / (v_norm + tol))


def project_box_sum(x: np.ndarray, lo: float, hi: float, target_sum: float,
                    tol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
    """
    Euclidean projection onto {x | lo <= x_i <= hi, sum(x)=target_sum}.
    Uses the fact that the solution has the form:
        x_i = clip(u_i - lambda, lo, hi),
    with lambda found by 1D bisection.
    """
    if x.size == 0:
        return x
    lo = float(lo)
    hi = float(hi)
    if lo > hi:
        raise ValueError(f"project_box_sum: lo > hi (lo={lo}, hi={hi})")
    n = int(x.size)
    target_sum = float(target_sum)
    lo_sum = n * lo
    hi_sum = n * hi
    if target_sum < lo_sum - 1e-12 or target_sum > hi_sum + 1e-12:
        raise ValueError(
            f"project_box_sum: target_sum={target_sum} outside [{lo_sum}, {hi_sum}]"
        )
    if abs(target_sum - lo_sum) <= tol:
        return np.full_like(x, lo, dtype=float)
    if abs(target_sum - hi_sum) <= tol:
        return np.full_like(x, hi, dtype=float)

    lam_lo = float(np.min(x - hi))
    lam_hi = float(np.max(x - lo))
    for _ in range(max_iter):
        lam = 0.5 * (lam_lo + lam_hi)
        proj = np.clip(x - lam, lo, hi)
        if float(np.sum(proj)) > target_sum:
            lam_lo = lam  # need larger lambda to reduce the sum
        else:
            lam_hi = lam
        if lam_hi - lam_lo <= tol:
            break
    lam = 0.5 * (lam_lo + lam_hi)
    return np.clip(x - lam, lo, hi)
# --------------------------
# Model config
# --------------------------

@dataclass
class RingConfig:
    # network
    N: int = 64
    dt: float = 0.001  # s
    T: float = 5.0     # s (prefer multiple of period)
    seed: int = 0

    # external traveling wave
    A: float = 2.0
    omega: float = 2*np.pi*1.0   # rad/s 2 * np.pi * freq
    freq: Optional[float] = None  # Hz alias; when set, omega is updated to 2*pi*freq
    theta_c: float = np.pi/2     # bump half-width parameter (see manuscript)
    h_bg: float = 0.0            # constant background input
    spont_h_bg: float = 0.5      # background input for spontaneous-corr analysis
    # neuron nonlinearity
    g_type: str = "exp"  # "exp" or "sigmoid" (extend as needed)
    beta: float = 2.0
    g_c: float = 10.0          # exp scale (g = g_c * exp(beta*(u-u_c)))
    u_c: float = 1.0
    g_max: float = 200.0        # exp clamp to avoid blow-up (ignored for sigmoid)
    g_m: float = 200.0          # sigmoid max rate (g = g_m * sigmoid(beta*(u-u_c)))

    # synapse kernels
    tau_s: float = 0.01   # EPSP kernel time constant (s)
    tau_d: float = 0.5    # depression recovery time constant (s)
    stp_enabled: bool = True  # if False: no STP (d≡1), static synapse

    # initialization / bounds
    w0_init: float = 1.0  # initial c so that w0[k]=c/N
    U_init: float = 0.15   # initial release probability
    U_lo: float = 0.01     # lower bound for U
    U_hi: float = 1.0      # upper bound for U
    U_init_jitter: float = 0.0  # init jitter std for U (0 disables)

    # optional hard constraints (projection)
    constrain_w0_sum: bool = False
    w0_sum_target: Optional[float] = None  # if None, use initial sum
    constrain_w0_l2: bool = False
    w0_l2_rms_max: Optional[float] = None  # if None, use initial ||w0||_2 / sqrt(N)
    constrain_U_sum: bool = False
    U_sum_target: Optional[float] = None   # if None, use initial sum (after init/clip)

    # Monte Carlo / training
    batch_size: int = 64
    n_iter: int = 300
    lr_w0: float = 2e-3
    lr_U: float = 1e-3
    momentum: float = 0.0  # 0.0 keeps exact backward-compatibility with vanilla SGD
    nesterov: bool = True  # used only when momentum > 0
    update_clip: float = 0.01  # optional update-norm clip for SGD
    score_clip_percentile: Optional[float] = None  # per-batch percentile for per-trial score-term norm clipping (None or >=100 disables)
    score_median_clip_window: int = 0  # recent iteration window for median-based score-norm clipping (0 disables)
    score_median_clip_mult: float = 5.0  # clip score norms to (median * mult) if enabled

    parallel_batch: bool = True  # run trials in parallel across CPU processes
    num_workers: Optional[int] = None  # number of worker processes (None -> auto)

    # metrics
    bin_dt: float = 0.005   # for spike binning/correlation

    # firing-rate regularization (soft constraint on mean rate)
    rate_reg_lambda: float = 0.0  # strength (0 disables)
    rate_reg_type: str = "l2_mean"  # "l2_mean" (existing) or "kl"
    # target mean rate (Hz):
    # - l2_mean: None -> estimate at init
    # - kl: None -> use _DEFAULT_KL_RATE_TARGET_HZ
    rate_target: Optional[float] = None
    rate_target_trials: int = 1  # trials to estimate default rate target (l2_mean only)

    # snapshots (phase-aligned firing rate)
    snapshot_iters: Optional[List[int]] = None
    snapshot_rate_trials: int = 4
    snapshot_phase_bins: int = 60
    snapshot_rate_seed: Optional[int] = None
    # phase-aligned rate asymmetry summary (computed at init/final and stored in history)
    asymmetry_rate_trials: int = 8
    asymmetry_phase_bins: int = 60
    asymmetry_rate_seed: Optional[int] = None
    # optional block-wise scalar adaptive LR (RMSProp-style):
    # - one scalar for all w0[k], one scalar for all U[k]
    adaptive_lr_blockwise: bool = False
    adaptive_lr_beta: float = 0.99
    adaptive_lr_eps: float = 1e-8
    adaptive_lr_w0_scale_min: float = 0.1
    adaptive_lr_w0_scale_max: float = 10.0
    adaptive_lr_U_scale_min: float = 0.1
    adaptive_lr_U_scale_max: float = 10.0

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "freq":
            if value is None:
                object.__setattr__(self, name, None)
                return
            freq = float(value)
            if not np.isfinite(freq):
                raise ValueError(f"freq must be finite, got {value!r}")
            object.__setattr__(self, name, freq)
            object.__setattr__(self, "omega", 2.0 * np.pi * freq)
            return
        if name == "omega":
            omega = float(value)
            if not np.isfinite(omega):
                raise ValueError(f"omega must be finite, got {value!r}")
            object.__setattr__(self, name, omega)
            freq = getattr(self, "freq", None)
            if freq is not None and not np.isclose(omega, 2.0 * np.pi * float(freq), atol=1e-12, rtol=0.0):
                object.__setattr__(self, "freq", None)
            return
        object.__setattr__(self, name, value)


@dataclass
class ReplayConfig:
    """
    Background/cue replay simulation parameters (no traveling-wave drive).
    """
    T: float = 1.0           # duration (s)
    t_start: float = 0.0     # simulation start time (s), end is t_start + T
    h_bg: float = 1.0        # default background input
    h_bg_with_stp: Optional[float] = None  # override for associative STP condition
    h_bg_pre_stp: Optional[float] = None   # override for non-associative STP condition
    h_bg_no_stp: Optional[float] = None    # override for static condition
    cue_A: float = 5.0       # brief localized bump amplitude
    cue_theta: float = np.pi / 20  # bump half-width (rad)
    cue_center: float = 0.0        # bump center on ring (rad)
    cue_start: float = 0.0         # cue start time from simulation start (s)
    cue_duration: float = 0.05     # cue duration (s)
    bin_dt: float = 0.01           # bin size for visualization (s)
    plot_t_start: Optional[float] = None  # replay-heatmap plot window start (absolute time)
    plot_t_end: Optional[float] = None    # replay-heatmap plot window end (absolute time)
    n_trials: int = 5
    seed: int = 123


# --------------------------
# External input
# --------------------------

def ring_positions(N: int) -> np.ndarray:
    return 2*np.pi*np.arange(N)/N


def traveling_wave_input(z: np.ndarray, t: float, A: float, omega: float, theta_c: float) -> np.ndarray:
    """
    h(z,t)=A [cos(omega t - z) - cos(theta_c)]_+
    """
    x = np.cos(omega * t - z) - np.cos(theta_c)
    return A * np.maximum(x, 0.0)


def traveling_wave_input_derivative_theta_c(z: np.ndarray, t: float, A: float, omega: float, theta_c: float) -> np.ndarray:
    """
    h'(z,t) = d/d(theta_c) A [cos(omega t - z) - cos(theta_c)]_+
           ≈ A * sin(theta_c) * 1{cos(omega t - z) > cos(theta_c)}
    (Ignoring the measure-zero boundary where x=0.)
    """
    active = (np.cos(omega * t - z) > np.cos(theta_c)).astype(float)
    return A * np.sin(theta_c) * active


def cosine_bump(z: np.ndarray, center: float, theta: float) -> np.ndarray:
    """
    Cosine bump used for brief cues during replay: [cos(z-center) - cos(theta)]_+.
    """
    return np.maximum(np.cos(z - center) - np.cos(theta), 0.0)


def centered_offset_angles(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (zeta_sorted, order) where zeta spans [-pi, pi) for offset k=pre-post.
    order can be used to reorder arrays indexed by k=0..N-1 into centered angles.
    """
    k = np.arange(N)
    k_centered = ((k + N // 2) % N) - N // 2  # integers in [-N/2, N/2)
    order = np.argsort(k_centered)
    zeta = 2 * np.pi * k_centered / N
    return zeta[order], order


# --------------------------
# Nonlinearity
# --------------------------

def g_exp(u: np.ndarray, beta: float, g_c: float, u_c: float, g_max: float) -> np.ndarray:
    # escape rate (clamped)
    x = beta * (u - u_c)
    # clip exponent to avoid overflow
    x = np.clip(x, -50.0, 50.0)
    rho = g_c * np.exp(x)
    if g_max is not None and float(g_max) > 0.0:
        rho = np.minimum(rho, g_max)
    return rho


def g_sigmoid(u: np.ndarray, beta: float, g_m: float, u_c: float) -> np.ndarray:
    """
    Sigmoid firing-rate nonlinearity: g(u) = g_m / (1 + exp(-beta (u - u_c))).
    """
    x = beta * (u - u_c)
    # clip exponent to avoid overflow
    x = np.clip(x, -50.0, 50.0)
    s = 1.0 / (1.0 + np.exp(-x))
    return g_m * s


def _normalize_g_type(g_type: Optional[str]) -> str:
    if g_type is None:
        return "exp"
    return str(g_type).strip().lower()


def _g_type_code(g_type: Optional[str]) -> int:
    g_norm = _normalize_g_type(g_type)
    if g_norm in ("exp", "exponential"):
        return 0
    if g_norm in ("sigmoid", "logistic"):
        return 1
    raise ValueError(f"Unsupported g_type: {g_type!r}")


def rate_from_cfg(u: np.ndarray, cfg: "RingConfig") -> np.ndarray:
    """
    Return firing rate rho = g(u) for the configured nonlinearity.
    """
    g_type = _normalize_g_type(cfg.g_type)
    if g_type in ("exp", "exponential"):
        return g_exp(u, cfg.beta, cfg.g_c, cfg.u_c, cfg.g_max)
    if g_type in ("sigmoid", "logistic"):
        return g_sigmoid(u, cfg.beta, cfg.g_m, cfg.u_c)
    raise ValueError(f"Unsupported g_type: {cfg.g_type!r}")


def nonlinearity_terms(
    u: np.ndarray,
    h_p: np.ndarray,
    cfg: "RingConfig",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (rho, dlogg, gprime, eta) for the estimator:
      rho   = g(u)
      dlogg = d/du log g(u) = g'(u)/g(u)
      gprime = g'(u)
      eta   = (h_p * g'(u)/g(u))**2 * (2 * g''(u)/g'(u) - g'(u)/g(u))
    """
    g_type = _normalize_g_type(cfg.g_type)
    if g_type in ("exp", "exponential"):
        x = cfg.beta * (u - cfg.u_c)
        x = np.clip(x, -50.0, 50.0)
        rho_raw = cfg.g_c * np.exp(x)
        if cfg.g_max is not None and float(cfg.g_max) > 0.0:
            g_max = float(cfg.g_max)
            rho = np.minimum(rho_raw, g_max)
            # Clipped region is constant in u, so g'=dlogg=eta=0 there
            unsat = rho_raw < g_max
        else:
            rho = rho_raw
            unsat = np.ones_like(u, dtype=bool)
        dlogg = np.where(unsat, cfg.beta, 0.0)
        gprime = np.where(unsat, cfg.beta * rho_raw, 0.0)
        eta = np.where(unsat, (cfg.beta ** 3) * (h_p ** 2), 0.0)
        return rho, dlogg, gprime, eta
    if g_type in ("sigmoid", "logistic"):
        rho = g_sigmoid(u, cfg.beta, cfg.g_m, cfg.u_c)
        s = rho / cfg.g_m
        one_minus_s = 1.0 - s
        dlogg = cfg.beta * one_minus_s
        gprime = cfg.beta * rho * one_minus_s
        eta = (cfg.beta ** 3) * (h_p ** 2) * (one_minus_s ** 2) * (1.0 - 3.0 * s)
        return rho, dlogg, gprime, eta
    raise ValueError(f"Unsupported g_type: {cfg.g_type!r}")


# --------------------------
# Optional numba acceleration
# --------------------------

@njit(cache=True)
def _rho_exp_nb(u: float, beta: float, g_c: float, u_c: float, g_max: float) -> float:
    x = beta * (u - u_c)
    if x < -50.0:
        x = -50.0
    elif x > 50.0:
        x = 50.0
    rho = g_c * np.exp(x)
    if g_max > 0.0 and rho > g_max:
        rho = g_max
    return rho


@njit(cache=True)
def _rho_sigmoid_nb(u: float, beta: float, g_m: float, u_c: float) -> float:
    x = beta * (u - u_c)
    if x < -50.0:
        x = -50.0
    elif x > 50.0:
        x = 50.0
    s = 1.0 / (1.0 + np.exp(-x))
    return g_m * s


@njit(cache=True)
def _simulate_trial_exact_grad_nb(
    w0: np.ndarray,
    U: np.ndarray,
    N: int,
    dt: float,
    T: float,
    A: float,
    omega: float,
    theta_c: float,
    h_bg: float,
    beta: float,
    g_c: float,
    u_c: float,
    g_max: float,
    g_m: float,
    g_type_code: int,
    tau_s: float,
    tau_d: float,
    stp_enabled: bool,
    bin_dt: float,
    rand: np.ndarray,
    record_spikes: bool,
) -> Tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    n_steps = int(np.round(T / dt))
    z = 2.0 * np.pi * np.arange(N) / N
    cos_theta_c = np.cos(theta_c)
    sin_theta_c = np.sin(theta_c)
    A_sin_theta_c = A * sin_theta_c
    beta2 = beta * beta
    beta3 = beta2 * beta

    S = np.zeros(N, dtype=np.float64)
    D = np.ones((N, N), dtype=np.float64)
    sU = np.zeros((N, N), dtype=np.float64)
    E_w0 = np.zeros((N, N), dtype=np.float64)
    E_U = np.zeros((N, N), dtype=np.float64)

    dec_s = np.exp(-dt / tau_s)
    dec_d = np.exp(-dt / tau_d)

    A_w0 = np.zeros(N, dtype=np.float64)
    B_w0 = np.zeros(N, dtype=np.float64)
    C_w0 = np.zeros(N, dtype=np.float64)

    A_U = np.zeros(N, dtype=np.float64)
    B_U = np.zeros(N, dtype=np.float64)
    C_U = np.zeros(N, dtype=np.float64)

    J_hat = 0.0
    rate_int = 0.0
    spike_count_total = 0.0
    spike_count_h0 = 0.0
    spike_log_rho_sum = 0.0

    n_bins = 1
    if record_spikes:
        n_bins = int(np.round(T / bin_dt))
        if n_bins < 1:
            n_bins = 1
    spikes_bin = np.zeros((n_bins, N), dtype=np.int32)
    bin_acc = 0.0
    bin_idx = 0

    spk = np.zeros(N, dtype=np.int32)
    dlogg_vec = np.zeros(N, dtype=np.float64)

    for step in range(n_steps):
        t = step * dt

        for i in range(N):
            x = np.cos(omega * t - z[i]) - cos_theta_c
            if x > 0.0:
                h = A * x + h_bg
                h_p = A_sin_theta_c
            else:
                h = h_bg
                h_p = 0.0

            u = h + S[i]
            h_p2 = h_p * h_p
            if g_type_code == 0:
                x_u = beta * (u - u_c)
                if x_u < -50.0:
                    x_u = -50.0
                elif x_u > 50.0:
                    x_u = 50.0
                rho_raw = g_c * np.exp(x_u)
                if g_max > 0.0 and rho_raw >= g_max:
                    rho = g_max
                    dlogg = 0.0
                    gprime = 0.0
                    post_weight = 0.0
                else:
                    rho = rho_raw
                    dlogg = beta
                    gprime = beta * rho
                    J_hat += (beta2 * h_p2) * rho * dt
                    post_weight = rho * (beta3 * h_p2)
                dlogg_vec[i] = dlogg
            else:
                rho = _rho_sigmoid_nb(u, beta, g_m, u_c)
                s = rho / g_m
                one_minus_s = 1.0 - s
                dlogg = beta * one_minus_s
                gprime = beta * rho * one_minus_s
                J_hat += (dlogg * dlogg) * h_p2 * rho * dt
                eta = beta3 * h_p2 * (one_minus_s * one_minus_s) * (1.0 - 3.0 * s)
                post_weight = rho * eta
                dlogg_vec[i] = dlogg
            rate_int += rho * dt

            for k in range(N):
                Ew0 = E_w0[i, k]
                Eu = E_U[i, k]
                A_w0[k] += post_weight * Ew0 * dt
                B_w0[k] += gprime * Ew0 * dt
                A_U[k] += post_weight * Eu * dt
                B_U[k] += gprime * Eu * dt

            p = 1.0 - np.exp(-rho * dt)
            spk[i] = 1 if rand[step, i] < p else 0
            if spk[i] != 0:
                spike_count_total += 1.0
                if h_p == 0.0:
                    spike_count_h0 += 1.0
                rho_for_log = rho
                if rho_for_log < 1e-12:
                    rho_for_log = 1e-12
                spike_log_rho_sum += np.log(rho_for_log)

        if record_spikes:
            bin_acc += dt

        for i in range(N):
            if spk[i] != 0:
                dlogg = dlogg_vec[i]
                for k in range(N):
                    C_w0[k] += dlogg * E_w0[i, k]
                    C_U[k] += dlogg * E_U[i, k]
                if record_spikes:
                    spikes_bin[bin_idx, i] += 1

        if record_spikes and (bin_acc + 1e-12 >= bin_dt):
            bin_acc -= bin_dt
            if bin_idx < n_bins - 1:
                bin_idx += 1

        S *= dec_s
        E_w0 *= dec_s
        E_U *= dec_s
        if stp_enabled:
            sU *= dec_d
            D = 1.0 - (1.0 - D) * dec_d

        for j in range(N):
            if spk[j] != 0:
                for k in range(N):
                    i = j - k
                    if i < 0:
                        i += N
                    if stp_enabled:
                        d_pre = D[i, k]
                        sU_pre = sU[i, k]
                    else:
                        d_pre = 1.0
                        sU_pre = 0.0

                    w_eff = w0[k] * U[k] * d_pre
                    S[i] += w_eff
                    E_w0[i, k] += U[k] * d_pre
                    E_U[i, k] += w0[k] * (d_pre + U[k] * sU_pre)

                    if stp_enabled:
                        sU[i, k] = (1.0 - U[k]) * sU_pre - d_pre
                        D[i, k] = d_pre * (1.0 - U[k])

    score_base_w0 = C_w0 - B_w0
    score_base_U = C_U - B_U

    return (
        J_hat,
        spikes_bin,
        A_w0,
        A_U,
        score_base_w0,
        score_base_U,
        B_w0,
        B_U,
        spike_count_total,
        spike_count_h0,
        rate_int,
        spike_log_rho_sum,
    )

# --------------------------
# Simulation + gradient (exact estimator)
# --------------------------

@dataclass
class TrialOutputs:
    J_hat: float
    grad_w0: Optional[np.ndarray] = None
    grad_U: Optional[np.ndarray] = None
    spikes_bin: Optional[np.ndarray] = None  # shape (T_bins, N)
    A_w0: Optional[np.ndarray] = None
    B_w0: Optional[np.ndarray] = None
    score_w0: Optional[np.ndarray] = None
    score_base_w0: Optional[np.ndarray] = None  # (C_w0 - B_w0) before baseline scaling (C weighted by dlogg, B by g')
    A_U: Optional[np.ndarray] = None
    B_U: Optional[np.ndarray] = None
    score_U: Optional[np.ndarray] = None
    score_base_U: Optional[np.ndarray] = None  # (C_U - B_U) before baseline scaling (C weighted by dlogg, B by g')
    spike_count_total: Optional[float] = None
    spike_count_h0: Optional[float] = None
    rate_int: Optional[float] = None
    spike_log_rho_sum: Optional[float] = None

def simulate_trial_exact_grad(
    w0: np.ndarray,
    U: np.ndarray,
    cfg: RingConfig,
    rng: np.random.Generator,
    baseline_b: Optional[float] = None,
    record_spikes: bool = False,
) -> TrialOutputs:
    use_numba = _USE_NUMBA and _normalize_g_type(cfg.g_type) in ("exp", "exponential", "sigmoid", "logistic")
    if use_numba:
        n_steps = int(np.round(cfg.T / cfg.dt))
        rand = rng.random((n_steps, cfg.N))
        g_type_code = _g_type_code(cfg.g_type)
        (
            J_hat,
            spikes_bin,
            A_w0,
            A_U,
            score_base_w0,
            score_base_U,
            B_w0,
            B_U,
            spike_count_total,
            spike_count_h0,
            rate_int,
            spike_log_rho_sum,
        ) = _simulate_trial_exact_grad_nb(
            w0,
            U,
            int(cfg.N),
            float(cfg.dt),
            float(cfg.T),
            float(cfg.A),
            float(cfg.omega),
            float(cfg.theta_c),
            float(cfg.h_bg),
            float(cfg.beta),
            float(cfg.g_c),
            float(cfg.u_c),
            float(cfg.g_max),
            float(cfg.g_m),
            int(g_type_code),
            float(cfg.tau_s),
            float(cfg.tau_d),
            bool(cfg.stp_enabled),
            float(cfg.bin_dt),
            rand,
            bool(record_spikes),
        )
        score_w0 = None
        score_U = None
        grad_w0 = None
        grad_U = None
        if baseline_b is not None:
            score_w0 = (J_hat - float(baseline_b)) * score_base_w0
            score_U = (J_hat - float(baseline_b)) * score_base_U
            invN = 1.0 / float(cfg.N)
            grad_w0 = (A_w0 + score_w0) * invN
            grad_U = (A_U + score_U) * invN
        spikes_out = spikes_bin if record_spikes else None
        return TrialOutputs(
            J_hat=float(J_hat),
            grad_w0=grad_w0,
            grad_U=grad_U,
            spikes_bin=spikes_out,
            A_w0=A_w0,
            B_w0=B_w0,
            score_w0=score_w0,
            score_base_w0=score_base_w0,
            A_U=A_U,
            B_U=B_U,
            score_U=score_U,
            score_base_U=score_base_U,
            spike_count_total=float(spike_count_total),
            spike_count_h0=float(spike_count_h0),
            rate_int=float(rate_int),
            spike_log_rho_sum=float(spike_log_rho_sum),
        )
    # Non-numba (pure-numpy) implementation follows.
    # Returns exact-gradient components for shared parameters w0[k], U[k].
    # The simulation produces A and score_base = (C - B); if a baseline is supplied, we also
    # assemble score and grad via grad(b) = (A + (J_hat - b) * score_base) / N.
    # Implements Appendix eligibility updates for STD-only synapses:
    #   d(t^+) = d(t^-)*(1-U)
    #   sU(t^+) = (1-U) sU(t^-) - d(t^-)
    #   eligibility jump at presyn spike:
    #     e_w0 += dw/dw0 = U d(t^-)
    #     e_U  += dw/dU  = w0 ( d(t^-) + U sU(t^-) )
    #   and exponential decay between spikes (discretized with dt).
    N = cfg.N
    dt = cfg.dt
    n_steps = int(np.round(cfg.T / dt))
    z = ring_positions(N)

    # state variables
    S = np.zeros(N)  # recurrent synaptic input current (EPSP-filtered sum)
    # synapse states indexed by (postsyn i, offset k): presyn j = i+k mod N (offset = pre - post)
    # If cfg.stp_enabled=False, we use static synapses with d≡1 and sU≡0.
    D = np.ones((N, N)) if cfg.stp_enabled else None
    sU = np.zeros((N, N)) if cfg.stp_enabled else None
    E_w0 = np.zeros((N, N))      # eligibility trace e_{ij}^{w0} aligned to postsyn
    E_U = np.zeros((N, N))       # eligibility trace e_{ij}^{U} aligned to postsyn

    # decay factors
    dec_s = np.exp(-dt / cfg.tau_s)
    dec_d = np.exp(-dt / cfg.tau_d)

    # accumulators for gradient components
    A_w0 = np.zeros(N)  # ∫ rho_i * eta_i * e dt summed over i
    B_w0 = np.zeros(N)  # ∫ g'(u_i) * e dt summed over i
    C_w0 = np.zeros(N)  # Σ_{spikes i} (g'/g)_i * e(t_i)

    A_U = np.zeros(N)
    B_U = np.zeros(N)
    C_U = np.zeros(N)

    # FI functional
    J_hat = 0.0
    rate_int = 0.0
    spike_count_total = 0.0
    spike_count_h0 = 0.0
    spike_log_rho_sum = 0.0

    # spike recording (binned)
    spikes_bin = None
    if record_spikes:
        bin_dt = cfg.bin_dt
        n_bins = int(np.round(cfg.T / bin_dt))
        spikes_bin = np.zeros((n_bins, N), dtype=np.int32)
        bin_acc = 0.0
        bin_idx = 0

    # precompute offsets
    k_vec = np.arange(N)

    for step in range(n_steps):
        t = step * dt

        # external input
        h = traveling_wave_input(z, t, cfg.A, cfg.omega, cfg.theta_c) + cfg.h_bg
        h_p = traveling_wave_input_derivative_theta_c(z, t, cfg.A, cfg.omega, cfg.theta_c)

        # membrane potential and nonlinearity terms
        u = h + S
        # NOTE: For non-exp g_type, nonlinearity_terms may return placeholders
        # until the estimator-specific derivatives are implemented.
        rho, dlogg, gprime, eta = nonlinearity_terms(u, h_p, cfg)

        # Fisher information integrand:
        # J = ∫ Σ_i [h'(t) * dlogg(u_i)]^2 rho dt
        J_hat += np.sum((h_p * dlogg) ** 2 * rho) * dt
        rate_int += float(np.sum(rho)) * dt

        # eta for pathwise term (nonlinearity-dependent)
        post_weight = rho * eta  # rho_i * eta_i

        # accumulate integrals (vectorized over k)
        # A_k += Σ_i post_weight[i] * E[i,k] dt
        A_w0 += np.sum(post_weight[:, None] * E_w0, axis=0) * dt
        B_w0 += np.sum(gprime[:, None] * E_w0, axis=0) * dt
        A_U  += np.sum(post_weight[:, None] * E_U, axis=0) * dt
        B_U  += np.sum(gprime[:, None] * E_U, axis=0) * dt

        # generate spikes in this dt (Poisson increments)
        p = 1.0 - np.exp(-rho * dt)
        spk = (rng.random(N) < p).astype(np.int32)  # 0/1
        spike_count_total += float(spk.sum())
        if spk.any():
            spike_count_h0 += float(spk[h_p == 0.0].sum())
            rho_spk = np.maximum(rho[spk != 0], 1e-12)
            spike_log_rho_sum += float(np.sum(np.log(rho_spk)))

        # score-term spike sums: C_k += Σ_{i spikes} (dlogg_i * E[i,k])
        if spk.any():
            spk_w = np.where(spk != 0, dlogg, 0.0)
            C_w0 += np.sum(spk_w[:, None] * E_w0, axis=0)
            C_U  += np.sum(spk_w[:, None] * E_U, axis=0)

        # record spikes
        if record_spikes:
            bin_dt = cfg.bin_dt
            bin_acc += dt
            if spk.any():
                spikes_bin[bin_idx, :] += spk
            if bin_acc + 1e-12 >= bin_dt:
                bin_acc -= bin_dt
                bin_idx = min(bin_idx + 1, spikes_bin.shape[0] - 1)

        # decay continuous-time states
        S *= dec_s
        E_w0 *= dec_s
        E_U *= dec_s
        if cfg.stp_enabled:
            sU *= dec_d
            D = 1.0 - (1.0 - D) * dec_d

        # presynaptic-spike-triggered updates (loop over spiking neurons)
        if spk.any():
            presyn_idx = np.flatnonzero(spk)
            for j in presyn_idx:
                # for each offset k, postsyn is i=(j-k) mod N (offset = pre - post)
                i_vec = (j - k_vec) % N  # permutation of 0..N-1
                # gather pre-spike synapse states for these synapses (t^-)
                if cfg.stp_enabled:
                    d_pre = D[i_vec, k_vec].copy()
                    sU_pre = sU[i_vec, k_vec].copy()
                else:
                    d_pre = np.ones_like(k_vec, dtype=float)
                    sU_pre = np.zeros_like(k_vec, dtype=float)

                # effective synaptic weight at this presyn spike
                w_eff = w0[k_vec] * U[k_vec] * d_pre
                # add to synaptic current
                S[i_vec] += w_eff

                # eligibility jumps (use t^- values)
                E_w0[i_vec, k_vec] += U[k_vec] * d_pre
                E_U[i_vec, k_vec]  += w0[k_vec] * (d_pre + U[k_vec] * sU_pre)

                # update sU and D at spike (only if STP enabled)
                if cfg.stp_enabled:
                    sU[i_vec, k_vec] = (1.0 - U[k_vec]) * sU_pre - d_pre
                    D[i_vec, k_vec] = d_pre * (1.0 - U[k_vec])

    # exact gradient components for shared parameters:
    # score_base = (C - B), grad(b) = (A + (J_hat - b) * score_base) / N
    # Normalize by N because each parameter affects N synapses (one per postsyn), matching the 1/(2π) average.
    score_base_w0 = C_w0 - B_w0
    score_base_U = C_U - B_U
    score_w0 = None
    score_U = None
    grad_w0 = None
    grad_U = None
    if baseline_b is not None:
        score_w0 = (J_hat - float(baseline_b)) * score_base_w0
        score_U = (J_hat - float(baseline_b)) * score_base_U
        invN = 1.0 / float(N)
        grad_w0 = (A_w0 + score_w0) * invN
        grad_U = (A_U + score_U) * invN

    return TrialOutputs(
        J_hat=float(J_hat),
        grad_w0=grad_w0,
        grad_U=grad_U,
        spikes_bin=spikes_bin,
        A_w0=A_w0,
        B_w0=B_w0,
        score_w0=score_w0,
        score_base_w0=score_base_w0,
        A_U=A_U,
        B_U=B_U,
        score_U=score_U,
        score_base_U=score_base_U,
        spike_count_total=spike_count_total,
        spike_count_h0=spike_count_h0,
        rate_int=float(rate_int),
        spike_log_rho_sum=float(spike_log_rho_sum),
    )


def _default_worker_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _mp_context() -> mp.context.BaseContext:
    if sys.platform == "win32":
        return mp.get_context("spawn")
    return mp.get_context("fork")


def _run_trial_worker(
    args: Tuple[np.ndarray, np.ndarray, RingConfig, int]
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    w0, U, cfg, seed = args
    rng = np.random.default_rng(seed)
    out = simulate_trial_exact_grad(w0, U, cfg, rng, record_spikes=False)
    assert out.A_w0 is not None and out.score_base_w0 is not None
    assert out.A_U is not None and out.score_base_U is not None
    assert out.B_w0 is not None and out.B_U is not None
    assert out.spike_count_total is not None and out.spike_count_h0 is not None
    assert out.rate_int is not None
    assert out.spike_log_rho_sum is not None
    return (
        out.J_hat,
        out.A_w0,
        out.score_base_w0,
        out.A_U,
        out.score_base_U,
        out.B_w0,
        out.B_U,
        out.spike_count_total,
        out.spike_count_h0,
        out.rate_int,
        out.spike_log_rho_sum,
    )


# --------------------------
# Metrics: cross-correlation vs distance
# --------------------------
def _next_pow_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


def _cross_correlogram_by_distance_fft(
    spikes_bin_x: np.ndarray,
    spikes_bin_y: np.ndarray,
    max_lag_bins: int = 50,
    eps: float = 1e-9,
) -> np.ndarray:
    X = spikes_bin_x.astype(float)
    Y = spikes_bin_y.astype(float)
    if X.shape != Y.shape:
        raise ValueError(f"spikes_bin_x and spikes_bin_y must match, got {X.shape} vs {Y.shape}")
    T, N = X.shape
    mu_x = X.mean(axis=0, keepdims=True)
    mu_y = Y.mean(axis=0, keepdims=True)
    Xc = X - mu_x
    Yc = Y - mu_y
    var_x = (Xc * Xc).mean(axis=0) + eps
    var_y = (Yc * Yc).mean(axis=0) + eps

    L = max_lag_bins
    lags = np.arange(-L, L + 1, dtype=int)
    lengths = (T - np.abs(lags)).astype(float)
    valid = lengths > 0
    lengths_safe = lengths.copy()
    lengths_safe[~valid] = 1.0

    pad = _next_pow_two(2 * T - 1)
    F_X = np.fft.rfft(Xc, n=pad, axis=0)
    F_Y = np.fft.rfft(Yc, n=pad, axis=0)
    lag_idx = np.where(lags >= 0, lags, pad + lags)

    C = np.empty((N, 2 * L + 1), dtype=float)
    idx = np.arange(N)
    for d in range(N):
        y_idx = (idx - d) % N
        denom = np.sqrt(var_x * var_y[y_idx])
        cross_spec = F_X * np.conj(F_Y[:, y_idx])
        corr_full = np.fft.irfft(cross_spec, n=pad, axis=0)
        corr_lags = corr_full[lag_idx, :]
        mean_normed = np.mean(corr_lags / denom[None, :], axis=1)
        C[d, :] = mean_normed / lengths_safe
        C[d, ~valid] = np.nan
    return C


@njit(cache=True)
def _cross_correlogram_by_distance_nb(
    X: np.ndarray,
    Y: np.ndarray,
    max_lag_bins: int,
    eps: float,
) -> np.ndarray:
    T, N = X.shape
    mu_x = np.zeros(N, dtype=np.float64)
    mu_y = np.zeros(N, dtype=np.float64)
    for i in range(N):
        sx = 0.0
        sy = 0.0
        for t in range(T):
            sx += X[t, i]
            sy += Y[t, i]
        mu_x[i] = sx / T
        mu_y[i] = sy / T

    Xc = np.empty_like(X)
    Yc = np.empty_like(Y)
    for t in range(T):
        for i in range(N):
            Xc[t, i] = X[t, i] - mu_x[i]
            Yc[t, i] = Y[t, i] - mu_y[i]

    var_x = np.zeros(N, dtype=np.float64)
    var_y = np.zeros(N, dtype=np.float64)
    for i in range(N):
        sx = 0.0
        sy = 0.0
        for t in range(T):
            sx += Xc[t, i] * Xc[t, i]
            sy += Yc[t, i] * Yc[t, i]
        var_x[i] = sx / T + eps
        var_y[i] = sy / T + eps

    L = max_lag_bins
    C = np.zeros((N, 2 * L + 1), dtype=np.float64)
    for d in range(N):
        for ell in range(2 * L + 1):
            lag = ell - L
            sum_cov = 0.0
            if lag < 0:
                length = T + lag
                for i in range(N):
                    j = i - d
                    if j < 0:
                        j += N
                    s = 0.0
                    for t in range(length):
                        s += Xc[t - lag, i] * Yc[t, j]
                    cov = s / length
                    sum_cov += cov / np.sqrt(var_x[i] * var_y[j])
            elif lag > 0:
                length = T - lag
                for i in range(N):
                    j = i - d
                    if j < 0:
                        j += N
                    s = 0.0
                    for t in range(length):
                        s += Xc[t, i] * Yc[t + lag, j]
                    cov = s / length
                    sum_cov += cov / np.sqrt(var_x[i] * var_y[j])
            else:
                length = T
                for i in range(N):
                    j = i - d
                    if j < 0:
                        j += N
                    s = 0.0
                    for t in range(length):
                        s += Xc[t, i] * Yc[t, j]
                    cov = s / length
                    sum_cov += cov / np.sqrt(var_x[i] * var_y[j])
            C[d, ell] = sum_cov / N
    return C


def cross_correlogram_by_distance(
    spikes_bin_x: np.ndarray,
    spikes_bin_y: np.ndarray,
    max_lag_bins: int = 50,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Cross-correlogram C[d, lag] averaged over i for each ring offset d (Pearson-normalized).

    C[d, lag] = mean_i mean_t X[t,i] * Y[t+lag, i-d]  (for lag>=0)
    Returns shape (N, 2*max_lag_bins+1) with lags from -L..L.
    """
    if _USE_CORR_FFT:
        return _cross_correlogram_by_distance_fft(
            spikes_bin_x, spikes_bin_y, max_lag_bins=max_lag_bins, eps=eps
        )
    if _USE_NUMBA:
        X = spikes_bin_x.astype(np.float64)
        Y = spikes_bin_y.astype(np.float64)
        return _cross_correlogram_by_distance_nb(X, Y, int(max_lag_bins), float(eps))

    X = spikes_bin_x.astype(float)
    Y = spikes_bin_y.astype(float)
    if X.shape != Y.shape:
        raise ValueError(f"spikes_bin_x and spikes_bin_y must match, got {X.shape} vs {Y.shape}")
    T, N = X.shape
    mu_x = X.mean(axis=0, keepdims=True)
    mu_y = Y.mean(axis=0, keepdims=True)
    Xc = X - mu_x
    Yc = Y - mu_y
    var_x = (Xc * Xc).mean(axis=0) + eps
    var_y = (Yc * Yc).mean(axis=0) + eps
    L = max_lag_bins
    C = np.zeros((N, 2*L + 1))

    for d in range(N):
        Yc_d = np.roll(Yc, shift=d, axis=1)  # centered, shifted
        var_y_d = np.roll(var_y, shift=d)
        for ell, lag in enumerate(range(-L, L+1)):
            if lag < 0:
                # X[t] with Y[t-lag]
                a = Xc[-lag:, :]
                b = Yc_d[:T+lag, :]
            elif lag > 0:
                a = Xc[:T-lag, :]
                b = Yc_d[lag:, :]
            else:
                a = Xc
                b = Yc_d
            cov = (a * b).mean(axis=0)  # per neuron i
            C[d, ell] = np.mean(cov / np.sqrt(var_x * var_y_d))
    return C


def correlogram_by_distance(spikes_bin: np.ndarray, max_lag_bins: int = 50, eps: float = 1e-9) -> np.ndarray:
    """
    Cross-correlogram C[d, lag] averaged over i for each ring offset d (Pearson-normalized).

    C[d, lag] = mean_i mean_t X[t,i] * X[t+lag, i-d]  (for lag>=0)
    Returns shape (N, 2*max_lag_bins+1) with lags from -L..L.
    """
    return cross_correlogram_by_distance(
        spikes_bin,
        spikes_bin,
        max_lag_bins=max_lag_bins,
        eps=eps,
    )


def correlogram_by_distance_trial_shuffle(
    spikes_trials: np.ndarray,
    max_lag_bins: int = 50,
    eps: float = 1e-9,
    shuffle_mode: str = "roll",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute within-trial correlogram, trial-shuffled (shift predictor), and corrected (within - shuffle).

    spikes_trials: shape (n_trials, T_bins, N)
    shuffle_mode:
      - "roll": pair trial t with t+1 (cyclic).
      - "all": average across all trial pairs i != j (more expensive).
    Returns (C_within, C_shuffle, C_corrected).
    """
    trials = np.asarray(spikes_trials)
    if trials.ndim != 3:
        raise ValueError("spikes_trials must have shape (n_trials, T_bins, N)")
    n_trials, _, N = trials.shape
    if n_trials < 2:
        raise ValueError("spikes_trials must include at least 2 trials for shuffling.")

    L = max_lag_bins
    C_within = np.zeros((N, 2*L + 1))
    for tr in range(n_trials):
        C_within += correlogram_by_distance(trials[tr], max_lag_bins=L, eps=eps)
    C_within /= n_trials

    if shuffle_mode == "roll":
        C_shuffle = np.zeros_like(C_within)
        for tr in range(n_trials):
            X = trials[tr]
            Y = trials[(tr + 1) % n_trials]
            C_shuffle += cross_correlogram_by_distance(X, Y, max_lag_bins=L, eps=eps)
        C_shuffle /= n_trials
    elif shuffle_mode == "all":
        C_shuffle = np.zeros_like(C_within)
        n_pairs = 0
        for i in range(n_trials):
            for j in range(n_trials):
                if i == j:
                    continue
                C_shuffle += cross_correlogram_by_distance(trials[i], trials[j], max_lag_bins=L, eps=eps)
                n_pairs += 1
        C_shuffle /= max(1, n_pairs)
    else:
        raise ValueError(f"unknown shuffle_mode={shuffle_mode!r}")

    C_corrected = C_within - C_shuffle
    return C_within, C_shuffle, C_corrected


def correlogram_window_summary(
    correlogram: np.ndarray,
    lags: np.ndarray,
    tau0: float,
) -> Dict[str, np.ndarray]:
    """
    Summarize correlogram within |tau| < tau0 per offset.
    Returns dict with keys: peak, area, asym.
    """
    if tau0 <= 0:
        raise ValueError("tau0 must be positive")
    if correlogram.shape[1] != lags.shape[0]:
        raise ValueError("lags length must match correlogram axis=1")

    mask = np.abs(lags) < tau0
    if not np.any(mask):
        idx = int(np.argmin(np.abs(lags)))
        mask = np.zeros_like(lags, dtype=bool)
        mask[idx] = True

    dt = float(np.mean(np.diff(lags)))
    pos_mask = (lags > 0) & mask
    neg_mask = (lags < 0) & mask

    peak = np.max(correlogram[:, mask], axis=1)
    area = np.sum(correlogram[:, mask], axis=1) * dt
    asym = (np.sum(correlogram[:, pos_mask], axis=1) - np.sum(correlogram[:, neg_mask], axis=1)) * dt
    return {"peak": peak, "area": area, "asym": asym}


def correlogram_epsp_weighted_positive_lag(
    correlogram: np.ndarray,
    lags: np.ndarray,
    tau_s: float,
    tau_max: Optional[float] = None,
) -> np.ndarray:
    """
    EPSP-kernel weighted positive-lag average per offset.

    C_+^kappa(d) = sum_{0 < tau <= tau_max} exp(-tau/tau_s) C[d,tau] / sum weights.
    The default tau_max is 5*tau_s, matching the correlogram summary window.
    """
    C = np.asarray(correlogram, dtype=float)
    lag_arr = np.asarray(lags, dtype=float)
    if C.ndim != 2:
        raise ValueError("correlogram must have shape (N, n_lags)")
    if C.shape[1] != lag_arr.shape[0]:
        raise ValueError("lags length must match correlogram axis=1")

    tau_s = float(tau_s)
    if tau_s <= 0.0:
        raise ValueError("tau_s must be positive")
    if tau_max is None:
        tau_max = 5.0 * tau_s
    tau_max = float(tau_max)
    if tau_max <= 0.0:
        raise ValueError("tau_max must be positive")

    pos_idx = np.flatnonzero(lag_arr > 0.0)
    if pos_idx.size == 0:
        raise ValueError("lags must include at least one positive lag")

    mask = (lag_arr > 0.0) & (lag_arr <= tau_max)
    if not np.any(mask):
        mask = np.zeros_like(lag_arr, dtype=bool)
        mask[pos_idx[0]] = True

    tau = lag_arr[mask]
    weights = np.exp(-tau / tau_s)
    values = C[:, mask]
    valid = np.isfinite(values)
    weighted_sum = np.sum(np.where(valid, values, 0.0) * weights[None, :], axis=1)
    weight_sum = np.sum(valid * weights[None, :], axis=1)
    out = np.full(C.shape[0], np.nan, dtype=float)
    np.divide(weighted_sum, weight_sum, out=out, where=weight_sum > 0.0)
    return out


def phase_aligned_rate(
    spikes_trials: np.ndarray,
    cfg: RingConfig,
    n_phase_bins: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate firing rate vs input phase phi = omega t - z using all neurons/time.
    Returns (phase_centers, rate).
    """
    trials = np.asarray(spikes_trials)
    if trials.ndim != 3:
        raise ValueError("spikes_trials must have shape (n_trials, T_bins, N)")
    n_trials, T_bins, N = trials.shape
    if n_trials < 1:
        raise ValueError("spikes_trials must include at least 1 trial")

    z = ring_positions(N)
    t = (np.arange(T_bins) + 0.5) * cfg.bin_dt
    phi = cfg.omega * t[:, None] - z[None, :]
    phi_wrapped = (phi + np.pi) % (2 * np.pi) - np.pi
    phi_flat = phi_wrapped.ravel()

    edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    counts = np.histogram(phi_flat, bins=edges)[0]
    spikes_sum = trials.sum(axis=0).astype(float)
    spikes_flat = spikes_sum.ravel()
    spikes_binned = np.histogram(phi_flat, bins=edges, weights=spikes_flat)[0]

    denom = counts * n_trials * cfg.bin_dt
    rate = np.full_like(spikes_binned, np.nan, dtype=float)
    valid = denom > 0
    rate[valid] = spikes_binned[valid] / denom[valid]
    phase_centers = 0.5 * (edges[:-1] + edges[1:])
    return phase_centers, rate


def _empty_phase_asymmetry_metrics() -> Dict[str, float]:
    return {key: float("nan") for key in _PHASE_ASYMMETRY_METRIC_KEYS}


def phase_rate_asymmetry_metrics(phase: np.ndarray, rate: np.ndarray) -> Dict[str, float]:
    """
    Compute left-right asymmetry metrics for a phase-aligned rate profile.
    """
    metrics = _empty_phase_asymmetry_metrics()

    phase_arr = np.asarray(phase, dtype=float).ravel()
    rate_arr = np.asarray(rate, dtype=float).ravel()
    n = min(phase_arr.size, rate_arr.size)
    if n <= 0:
        return metrics

    phase_arr = phase_arr[:n]
    rate_arr = rate_arr[:n]
    finite = np.isfinite(phase_arr) & np.isfinite(rate_arr)
    if int(np.sum(finite)) < 3:
        return metrics

    phase_arr = phase_arr[finite]
    rate_arr = rate_arr[finite]
    order = np.argsort(phase_arr)
    phase_arr = phase_arr[order]
    rate_arr = rate_arr[order]
    if phase_arr.size <= 0:
        return metrics

    metrics["peak_phase"] = float(phase_arr[int(np.argmax(rate_arr))])

    weights = np.clip(rate_arr, 0.0, None)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return metrics

    centroid = float(np.sum(weights * phase_arr) / weight_sum)
    metrics["centroid_phase"] = centroid

    centered = phase_arr - centroid
    m2 = float(np.sum(weights * (centered ** 2)) / weight_sum)
    if np.isfinite(m2) and m2 > 0.0:
        m3 = float(np.sum(weights * (centered ** 3)) / weight_sum)
        metrics["skewness"] = float(m3 / (m2 ** 1.5))

    pos_mass = float(np.sum(weights[phase_arr > 0.0]))
    neg_mass = float(np.sum(weights[phase_arr < 0.0]))
    metrics["area_index"] = float((pos_mass - neg_mass) / weight_sum)

    if phase_arr.size >= 2:
        if np.allclose(phase_arr, -phase_arr[::-1], rtol=1e-5, atol=1e-8):
            mirrored_rate = rate_arr[::-1]
        else:
            mirrored_rate = np.interp(-phase_arr, phase_arr, rate_arr, left=np.nan, right=np.nan)
        mirror_valid = np.isfinite(mirrored_rate)
        if np.any(mirror_valid):
            odd_part = 0.5 * (rate_arr[mirror_valid] - mirrored_rate[mirror_valid])
            denom = float(np.linalg.norm(rate_arr[mirror_valid]))
            if np.isfinite(denom) and denom > 0.0:
                metrics["odd_ratio"] = float(np.linalg.norm(odd_part) / denom)
    return metrics


def _phase_asymmetry_rate_required(cfg: RingConfig) -> bool:
    try:
        n_trials = int(cfg.asymmetry_rate_trials)
    except Exception:
        return False
    try:
        n_bins = int(cfg.asymmetry_phase_bins)
    except Exception:
        return False
    return (n_trials > 0) and (n_bins > 0)


def _phase_asymmetry_history_ok(
    history: Optional[Dict[str, np.ndarray]],
    cfg: Optional[RingConfig] = None,
) -> bool:
    if history is None:
        return False
    required_arrays = (
        "asymmetry_phase_init",
        "asymmetry_rate_init",
        "asymmetry_phase_final",
        "asymmetry_rate_final",
    )
    for key in required_arrays:
        if key not in history:
            return False
    try:
        phase_init = np.asarray(history["asymmetry_phase_init"], dtype=float).ravel()
        rate_init = np.asarray(history["asymmetry_rate_init"], dtype=float).ravel()
        phase_final = np.asarray(history["asymmetry_phase_final"], dtype=float).ravel()
        rate_final = np.asarray(history["asymmetry_rate_final"], dtype=float).ravel()
    except Exception:
        return False
    if phase_init.size == 0 or rate_init.size == 0 or phase_final.size == 0 or rate_final.size == 0:
        return False
    if phase_init.size != rate_init.size or phase_final.size != rate_final.size:
        return False

    for metric_key in _PHASE_ASYMMETRY_METRIC_KEYS:
        if f"asymmetry_{metric_key}_init" not in history:
            return False
        if f"asymmetry_{metric_key}_final" not in history:
            return False
    if cfg is not None and _phase_asymmetry_rate_required(cfg):
        expected_trials = int(cfg.asymmetry_rate_trials)
        expected_bins = int(cfg.asymmetry_phase_bins)
        expected_seed = _phase_asymmetry_eval_seed(cfg)
        trials_saved = _history_int_scalar(history, "asymmetry_trials")
        bins_saved = _history_int_scalar(history, "asymmetry_phase_bins")
        seed_saved = _history_int_scalar(history, "asymmetry_seed")
        if (trials_saved != expected_trials) or (bins_saved != expected_bins) or (seed_saved != expected_seed):
            return False
    return True


def _phase_asymmetry_eval_seed(cfg: RingConfig) -> int:
    seed = cfg.asymmetry_rate_seed
    if seed is None:
        seed = cfg.seed
    try:
        return int(seed)
    except Exception:
        return int(cfg.seed)


def _compute_phase_asymmetry_history(
    cfg: RingConfig,
    w0_final: np.ndarray,
    U_final: np.ndarray,
) -> Dict[str, np.ndarray]:
    if not _phase_asymmetry_rate_required(cfg):
        return {}

    n_trials = int(cfg.asymmetry_rate_trials)
    n_bins = int(cfg.asymmetry_phase_bins)
    seed_base = _phase_asymmetry_eval_seed(cfg)

    w0_init, U_init = init_params(cfg)
    spikes_init = sample_spikes_trials(cfg, w0_init, U_init, n_trials=n_trials, seed=seed_base + 11)
    spikes_final = sample_spikes_trials(
        cfg,
        np.asarray(w0_final, dtype=float),
        np.asarray(U_final, dtype=float),
        n_trials=n_trials,
        seed=seed_base + 12,
    )

    phase_init, rate_init = phase_aligned_rate(spikes_init, cfg, n_phase_bins=n_bins)
    phase_final, rate_final = phase_aligned_rate(spikes_final, cfg, n_phase_bins=n_bins)
    metrics_init = phase_rate_asymmetry_metrics(phase_init, rate_init)
    metrics_final = phase_rate_asymmetry_metrics(phase_final, rate_final)

    out: Dict[str, np.ndarray] = {
        "asymmetry_phase_init": np.asarray(phase_init, dtype=float),
        "asymmetry_rate_init": np.asarray(rate_init, dtype=float),
        "asymmetry_phase_final": np.asarray(phase_final, dtype=float),
        "asymmetry_rate_final": np.asarray(rate_final, dtype=float),
        "asymmetry_trials": np.array(n_trials, dtype=int),
        "asymmetry_phase_bins": np.array(n_bins, dtype=int),
        "asymmetry_seed": np.array(seed_base, dtype=int),
    }
    for metric_key in _PHASE_ASYMMETRY_METRIC_KEYS:
        out[f"asymmetry_{metric_key}_init"] = np.array(metrics_init[metric_key], dtype=float)
        out[f"asymmetry_{metric_key}_final"] = np.array(metrics_final[metric_key], dtype=float)
    return out


def _accum_phase_offset(
    accum: np.ndarray,
    values: np.ndarray,
    phase_bins: np.ndarray,
) -> None:
    """
    Add values[i,k] into accum[k, phase_bins[(i+k) % N]] for all i,k.
    accum has shape (N, n_phase_bins), values (N, N), phase_bins (N,).
    This aligns variables to presynaptic input phase.
    """
    N = values.shape[0]
    idx = np.arange(N)
    n_phase_bins = accum.shape[1]
    for k in range(N):
        bins = phase_bins[(idx + k) % N]
        accum[k] += np.bincount(bins, weights=values[:, k], minlength=n_phase_bins)


@njit(cache=True)
def _accum_phase_offset_nb(
    accum: np.ndarray,
    values: np.ndarray,
    phase_bins: np.ndarray,
) -> None:
    N = values.shape[0]
    for k in range(N):
        for i in range(N):
            bin_idx = phase_bins[(i + k) % N]
            accum[k, bin_idx] += values[i, k]


@njit(cache=True)
def _phase_aligned_synapse_vars_nb(
    w0: np.ndarray,
    U: np.ndarray,
    N: int,
    dt: float,
    T: float,
    A: float,
    omega: float,
    theta_c: float,
    h_bg: float,
    beta: float,
    g_c: float,
    u_c: float,
    g_max: float,
    g_m: float,
    g_type_code: int,
    tau_s: float,
    tau_d: float,
    stp_enabled: bool,
    bin_dt: float,
    phase_bins: np.ndarray,
    rand: np.ndarray,
    sum_d: np.ndarray,
    sum_E_w0: np.ndarray,
    sum_E_U: np.ndarray,
    sum_sU: np.ndarray,
) -> None:
    n_steps = int(np.round(T / dt))
    n_bins = phase_bins.shape[0]
    n_trials = rand.shape[0]

    z = 2.0 * np.pi * np.arange(N) / N
    cos_theta_c = np.cos(theta_c)
    dec_s = np.exp(-dt / tau_s)
    dec_d = np.exp(-dt / tau_d)

    spk = np.zeros(N, dtype=np.int32)

    for tr in range(n_trials):
        S = np.zeros(N, dtype=np.float64)
        D = np.ones((N, N), dtype=np.float64)
        sU = np.zeros((N, N), dtype=np.float64)
        E_w0 = np.zeros((N, N), dtype=np.float64)
        E_U = np.zeros((N, N), dtype=np.float64)

        bin_acc = 0.0
        bin_idx = 0

        for step in range(n_steps):
            t = step * dt

            for i in range(N):
                x = np.cos(omega * t - z[i]) - cos_theta_c
                if x > 0.0:
                    h = A * x + h_bg
                else:
                    h = h_bg
                u = h + S[i]
                if g_type_code == 0:
                    rho = _rho_exp_nb(u, beta, g_c, u_c, g_max)
                else:
                    rho = _rho_sigmoid_nb(u, beta, g_m, u_c)
                p = 1.0 - np.exp(-rho * dt)
                spk[i] = 1 if rand[tr, step, i] < p else 0

            S *= dec_s
            E_w0 *= dec_s
            E_U *= dec_s
            if stp_enabled:
                sU *= dec_d
                D = 1.0 - (1.0 - D) * dec_d

            for j in range(N):
                if spk[j] != 0:
                    for k in range(N):
                        i = j - k
                        if i < 0:
                            i += N
                        if stp_enabled:
                            d_pre = D[i, k]
                            sU_pre = sU[i, k]
                        else:
                            d_pre = 1.0
                            sU_pre = 0.0

                        w_eff = w0[k] * U[k] * d_pre
                        S[i] += w_eff
                        E_w0[i, k] += U[k] * d_pre
                        E_U[i, k] += w0[k] * (d_pre + U[k] * sU_pre)

                        if stp_enabled:
                            sU[i, k] = (1.0 - U[k]) * sU_pre - d_pre
                            D[i, k] = d_pre * (1.0 - U[k])

            bin_acc += dt
            if bin_acc + 1e-12 >= bin_dt:
                bin_acc -= bin_dt
                if bin_idx < n_bins:
                    bins_i = phase_bins[bin_idx]
                    _accum_phase_offset_nb(sum_d, D, bins_i)
                    _accum_phase_offset_nb(sum_E_w0, E_w0, bins_i)
                    _accum_phase_offset_nb(sum_E_U, E_U, bins_i)
                    _accum_phase_offset_nb(sum_sU, sU, bins_i)
                if bin_idx < n_bins - 1:
                    bin_idx += 1

def phase_aligned_synapse_vars(
    w0: np.ndarray,
    U: np.ndarray,
    cfg: RingConfig,
    n_trials: int = 8,
    seed: int = 0,
    n_phase_bins: int = 60,
) -> Dict[str, np.ndarray]:
    """
    Phase-aligned synapse variables vs presynaptic input phase, per offset k.
    Returns dict with keys: phase, d, w_eff, E_w0, E_U, sU.
    Each variable has shape (N, n_phase_bins).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    N = cfg.N
    dt = cfg.dt
    n_steps = int(np.round(cfg.T / dt))
    bin_dt = cfg.bin_dt
    n_bins = int(np.round(cfg.T / bin_dt))
    if n_bins < 1:
        raise ValueError("bin_dt is too large for the simulation duration.")

    z = ring_positions(N)
    t_bins = (np.arange(n_bins) + 0.5) * bin_dt
    phi = cfg.omega * t_bins[:, None] - z[None, :]
    phi_wrapped = (phi + np.pi) % (2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    phase_centers = 0.5 * (edges[:-1] + edges[1:])
    phase_bins = np.digitize(phi_wrapped, edges) - 1
    phase_bins = np.clip(phase_bins, 0, n_phase_bins - 1)
    counts = np.bincount(phase_bins.ravel(), minlength=n_phase_bins).astype(float)
    counts_total = counts * float(n_trials)

    sum_d = np.zeros((N, n_phase_bins), dtype=float)
    sum_E_w0 = np.zeros((N, n_phase_bins), dtype=float)
    sum_E_U = np.zeros((N, n_phase_bins), dtype=float)
    sum_sU = np.zeros((N, n_phase_bins), dtype=float)

    rng = np.random.default_rng(seed)

    use_numba = _USE_NUMBA and _normalize_g_type(cfg.g_type) in ("exp", "exponential", "sigmoid", "logistic")
    if use_numba:
        rand = rng.random((n_trials, n_steps, N))
        g_type_code = _g_type_code(cfg.g_type)
        _phase_aligned_synapse_vars_nb(
            w0,
            U,
            int(N),
            float(dt),
            float(cfg.T),
            float(cfg.A),
            float(cfg.omega),
            float(cfg.theta_c),
            float(cfg.h_bg),
            float(cfg.beta),
            float(cfg.g_c),
            float(cfg.u_c),
            float(cfg.g_max),
            float(cfg.g_m),
            int(g_type_code),
            float(cfg.tau_s),
            float(cfg.tau_d),
            bool(cfg.stp_enabled),
            float(bin_dt),
            phase_bins,
            rand,
            sum_d,
            sum_E_w0,
            sum_E_U,
            sum_sU,
        )
    else:
        k_vec = np.arange(N)
        dec_s = np.exp(-dt / cfg.tau_s)
        dec_d = np.exp(-dt / cfg.tau_d)

        for _ in range(n_trials):
            S = np.zeros(N, dtype=float)
            D = np.ones((N, N), dtype=float)
            sU = np.zeros((N, N), dtype=float)
            E_w0 = np.zeros((N, N), dtype=float)
            E_U = np.zeros((N, N), dtype=float)

            bin_acc = 0.0
            bin_idx = 0

            for step in range(n_steps):
                t = step * dt
                h = traveling_wave_input(z, t, cfg.A, cfg.omega, cfg.theta_c) + cfg.h_bg
                u = h + S
                rho = rate_from_cfg(u, cfg)

                p = 1.0 - np.exp(-rho * dt)
                spk = (rng.random(N) < p).astype(np.int32)

                # decay continuous-time states
                S *= dec_s
                E_w0 *= dec_s
                E_U *= dec_s
                if cfg.stp_enabled:
                    sU *= dec_d
                    D = 1.0 - (1.0 - D) * dec_d

                # presynaptic-spike-triggered updates
                if spk.any():
                    presyn_idx = np.flatnonzero(spk)
                    for j in presyn_idx:
                        i_vec = (j - k_vec) % N
                        if cfg.stp_enabled:
                            d_pre = D[i_vec, k_vec].copy()
                            sU_pre = sU[i_vec, k_vec].copy()
                        else:
                            d_pre = np.ones_like(k_vec, dtype=float)
                            sU_pre = np.zeros_like(k_vec, dtype=float)

                        w_eff = w0[k_vec] * U[k_vec] * d_pre
                        S[i_vec] += w_eff
                        E_w0[i_vec, k_vec] += U[k_vec] * d_pre
                        E_U[i_vec, k_vec] += w0[k_vec] * (d_pre + U[k_vec] * sU_pre)

                        if cfg.stp_enabled:
                            sU[i_vec, k_vec] = (1.0 - U[k_vec]) * sU_pre - d_pre
                            D[i_vec, k_vec] = d_pre * (1.0 - U[k_vec])

                bin_acc += dt
                if bin_acc + 1e-12 >= bin_dt:
                    bin_acc -= bin_dt
                    if bin_idx < n_bins:
                        bins_i = phase_bins[bin_idx]
                        _accum_phase_offset(sum_d, D, bins_i)
                        _accum_phase_offset(sum_E_w0, E_w0, bins_i)
                        _accum_phase_offset(sum_E_U, E_U, bins_i)
                        _accum_phase_offset(sum_sU, sU, bins_i)
                    bin_idx = min(bin_idx + 1, n_bins - 1)

    out = {"phase": phase_centers}
    valid = counts_total > 0
    for key, val in (
        ("d", sum_d),
        ("E_w0", sum_E_w0),
        ("E_U", sum_E_U),
        ("sU", sum_sU),
    ):
        avg = np.full_like(val, np.nan, dtype=float)
        avg[:, valid] = val[:, valid] / counts_total[valid]
        out[key] = avg

    out["w_eff"] = (w0[:, None] * U[:, None]) * out["d"]
    return out


# --------------------------
# Replay helpers (background-only)
# --------------------------

def _bin_spikes(
    spikes: np.ndarray,
    dt: float,
    bin_dt: float,
    t_start: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin spike array (T_steps, N) into counts with bin size bin_dt.
    Returns (counts, bin_times_center).
    """
    T_steps, _ = spikes.shape
    nbin = max(1, int(np.round(bin_dt / dt)))
    T_trim = (T_steps // nbin) * nbin
    counts = spikes[:T_trim].reshape(T_trim // nbin, nbin, -1).sum(axis=1).astype(float)
    times = (np.arange(counts.shape[0]) + 0.5) * (nbin * dt) + float(t_start)
    return counts, times


def _phase_trajectory(activity: np.ndarray, z: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Circular phase from activity(t,i) via arg(sum_i a_i e^{i z_i}).
    Returns (unwrapped_phase, concentration R).
    """
    Z = np.exp(1j * z)
    Tbins = activity.shape[0]
    ph = np.full(Tbins, np.nan, dtype=float)
    R = np.zeros(Tbins, dtype=float)
    for t in range(Tbins):
        a = activity[t]
        s = float(np.sum(a))
        if s <= eps:
            continue
        c = np.sum(a * Z)
        ph[t] = np.angle(c)
        R[t] = np.abs(c) / (s + eps)
    idx = np.flatnonzero(~np.isnan(ph))
    ph_u = np.full_like(ph, np.nan)
    if idx.size >= 2:
        ph_u[idx] = np.unwrap(ph[idx])
    elif idx.size == 1:
        ph_u[idx] = ph[idx]
    return ph_u, R


def _phase_velocity(ph_unwrapped: np.ndarray, t: np.ndarray, t_min: float, R: Optional[np.ndarray] = None) -> float:
    """
    Linear regression slope of unwrapped phase after t_min.
    Optionally weight by R (concentration) to downweight low-activity bins.
    """
    mask = (~np.isnan(ph_unwrapped)) & (t >= t_min)
    if R is not None:
        mask &= (R > 0.05)
    idx = np.flatnonzero(mask)
    if idx.size < 3:
        return float("nan")
    x = t[idx]
    y = ph_unwrapped[idx]
    w = np.ones_like(x) if R is None else R[idx]
    x0 = np.average(x, weights=w)
    y0 = np.average(y, weights=w)
    num = np.sum(w * (x - x0) * (y - y0))
    den = np.sum(w * (x - x0) ** 2) + 1e-12
    return float(num / den)


# --------------------------
# Training loop
# --------------------------

@dataclass
class FitResult:
    w0: np.ndarray
    U: np.ndarray
    history: Dict[str, np.ndarray]


# --------------------------
# Persistence helpers
# --------------------------

DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "saved_runs"


def _configs_match(cfg: RingConfig, cfg_saved: Dict[str, Any], tol: float = 1e-12) -> bool:
    """
    Compare a live cfg against one loaded from disk; tolerate tiny float noise.
    """
    current = asdict(cfg)
    # Analysis/visualization-only knobs should not invalidate learned-weight cache.
    ignored_keys = {
        "freq",
        "snapshot_iters",
        "snapshot_rate_trials",
        "snapshot_phase_bins",
        "snapshot_rate_seed",
        "asymmetry_rate_trials",
        "asymmetry_phase_bins",
        "asymmetry_rate_seed",
    }
    if set(cfg_saved.keys()) - set(current.keys()) - ignored_keys:
        return False

    def _rate_target_compatible(saved_rate_target: Any) -> bool:
        """
        Backward compatibility for legacy caches:
        older code persisted inferred rate_target into cfg.json even when
        user config left rate_target=None.
        """
        if current.get("rate_target") is not None:
            return False
        try:
            rate_reg_lambda = float(current.get("rate_reg_lambda", 0.0))
        except Exception:
            rate_reg_lambda = 0.0
        try:
            rate_reg_type = _normalize_rate_reg_type(current.get("rate_reg_type", "l2_mean"))
        except Exception:
            rate_reg_type = "l2_mean"
        if rate_reg_lambda <= 0.0:
            return saved_rate_target is None
        if rate_reg_type == "kl":
            if saved_rate_target is None:
                return True
            try:
                return np.isclose(float(saved_rate_target), _DEFAULT_KL_RATE_TARGET_HZ, atol=tol, rtol=0.0)
            except Exception:
                return False
        if saved_rate_target is None:
            return True
        try:
            return np.isfinite(float(saved_rate_target))
        except Exception:
            return False

    def _values_match(val: Any, saved: Any) -> bool:
        if isinstance(val, bool):
            return bool(saved) == val
        if isinstance(val, (int, np.integer)):
            try:
                return int(saved) == int(val)
            except Exception:
                return False
        if isinstance(val, (float, np.floating)):
            try:
                return np.isclose(float(saved), float(val), atol=tol, rtol=0.0)
            except Exception:
                return False
        return val == saved

    missing_keys = set(current.keys()) - set(cfg_saved.keys()) - ignored_keys
    if missing_keys:
        defaults = asdict(RingConfig())
        for key in missing_keys:
            if not _values_match(current[key], defaults[key]):
                return False

    for key, val in current.items():
        if key in ignored_keys:
            continue
        if key not in cfg_saved:
            continue
        if key == "rate_target" and _rate_target_compatible(cfg_saved[key]):
            continue
        if not _values_match(val, cfg_saved[key]):
            return False
    return True


def _run_dir(cfg_name: str, save_dir: Path = DEFAULT_SAVE_DIR) -> Path:
    return Path(save_dir) / cfg_name


def _run_path(cfg_name: str, save_dir: Path = DEFAULT_SAVE_DIR) -> Path:
    return _run_dir(cfg_name, save_dir) / "fit.npz"


def save_fit_result(cfg_name: str, cfg: RingConfig, res: FitResult, save_dir: Path = DEFAULT_SAVE_DIR) -> Path:
    """
    Persist learned parameters + training history with the config used.
    """
    path = _run_path(cfg_name, save_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    cfg_json = json.dumps(asdict(cfg), sort_keys=True, indent=2)
    history_keys = np.array(list(res.history.keys()), dtype=str)
    history_data = {f"hist_{k}": v for k, v in res.history.items()}

    cfg_path = path.parent / "cfg.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        f.write(cfg_json)
        f.write("\n")

    np.savez_compressed(
        path,
        w0=res.w0,
        U=res.U,
        cfg_json=cfg_json,
        history_keys=history_keys,
        **history_data,
    )
    return path


def save_replay_config(cfg_name: str, rp: ReplayConfig, save_dir: Path = DEFAULT_SAVE_DIR) -> Path:
    """
    Persist replay config alongside the cached run for compatibility.
    """
    path = _run_dir(cfg_name, save_dir) / "replay_cfg.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    rp_json = json.dumps(asdict(rp), sort_keys=True, indent=2)
    with path.open("w", encoding="utf-8") as f:
        f.write(rp_json)
        f.write("\n")
    return path


def try_load_fit_result(
    cfg_name: str,
    cfg: RingConfig,
    save_dir: Path = DEFAULT_SAVE_DIR,
) -> Tuple[Optional[FitResult], str, Path]:
    """
    Attempt to load a cached run. Returns (result or None, status, path).
    status in {"ok", "missing", "config_mismatch", "read_error", "incomplete"}.
    """
    path = _run_path(cfg_name, save_dir)
    if not path.exists():
        return None, "missing", path

    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return None, "read_error", path

    try:
        cfg_saved = json.loads(str(data["cfg_json"]))
    except Exception:
        return None, "read_error", path

    if not _configs_match(cfg, cfg_saved):
        return None, "config_mismatch", path

    if "history_keys" not in data:
        return None, "incomplete", path
    history_keys = [str(k) for k in data["history_keys"]]

    history = {}
    try:
        for k in history_keys:
            history[k] = data[f"hist_{k}"]
    except KeyError:
        return None, "incomplete", path

    res = FitResult(w0=data["w0"], U=data["U"], history=history)
    return res, "ok", path


def ensure_fit_result_asymmetry_history(
    cfg_name: str,
    cfg: RingConfig,
    res: FitResult,
    save_dir: Path = DEFAULT_SAVE_DIR,
    verbose: bool = True,
) -> Tuple[FitResult, bool, Path]:
    """
    Ensure phase-asymmetry history fields are present for a cached/new fit result.
    If missing, compute them from (cfg, init params, learned params) and resave fit.npz.
    """
    path = _run_path(cfg_name, save_dir)
    if not _phase_asymmetry_rate_required(cfg):
        return res, False, path
    if _phase_asymmetry_history_ok(res.history, cfg=cfg):
        return res, False, path

    add_history = _compute_phase_asymmetry_history(cfg, res.w0, res.U)
    if not add_history:
        return res, False, path

    merged_history = dict(res.history) if res.history is not None else {}
    merged_history.update(add_history)
    updated = FitResult(
        w0=np.asarray(res.w0, dtype=float).copy(),
        U=np.asarray(res.U, dtype=float).copy(),
        history=merged_history,
    )
    save_path = save_fit_result(cfg_name, cfg, updated, save_dir=save_dir)
    if verbose:
        print(f"Backfilled phase-asymmetry history and updated cache: {save_path}")
    return updated, True, save_path


def init_params(cfg: RingConfig, U_init: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    N = cfg.N

    # w0 initial: constant c/N
    w0_val = float(cfg.w0_init) / max(N, 1)
    w0 = np.full(N, w0_val)

    # U initial
    if U_init is None:
        U_init = cfg.U_init
    U = np.full(N, float(U_init))
    # small jitter can help break symmetry in practice
    if cfg.U_init_jitter and cfg.U_init_jitter > 0:
        U += float(cfg.U_init_jitter) * rng.normal(size=N)
    U = np.clip(U, cfg.U_lo, cfg.U_hi)

    return w0, U


def _clip_score_trials(score_trials: np.ndarray, pct: Optional[float]) -> np.ndarray:
    if pct is None:
        return score_trials
    pct = float(pct)
    if pct >= 100.0:
        return score_trials
    norms = np.linalg.norm(score_trials, axis=1)
    if norms.size == 0:
        return score_trials
    clip_norm = float(np.percentile(norms, pct))
    scale = np.ones_like(norms)
    mask = norms > clip_norm
    scale[mask] = clip_norm / (norms[mask] + 1e-12)
    return score_trials * scale[:, None]


def _clip_score_trials_by_norm(score_trials: np.ndarray, clip_norm: Optional[float]) -> np.ndarray:
    if clip_norm is None:
        return score_trials
    try:
        clip_norm = float(clip_norm)
    except Exception:
        return score_trials
    if clip_norm <= 0.0:
        return score_trials
    norms = np.linalg.norm(score_trials, axis=1)
    if norms.size == 0:
        return score_trials
    scale = np.ones_like(norms)
    mask = norms > clip_norm
    scale[mask] = clip_norm / (norms[mask] + 1e-12)
    return score_trials * scale[:, None]


def _score_norm_median(score_trials: np.ndarray) -> Optional[float]:
    norms = np.linalg.norm(score_trials, axis=1)
    if norms.size == 0:
        return None
    return float(np.median(norms))


def _score_median_clip_norm(history: Optional[deque], mult: Optional[float]) -> Optional[float]:
    if history is None or len(history) == 0:
        return None
    try:
        mult = float(mult)
    except Exception:
        return None
    if mult <= 0.0:
        return None
    median = float(np.median(np.array(history, dtype=float)))
    if not np.isfinite(median) or median <= 0.0:
        return None
    return median * mult


def _normalize_snapshot_iters(snapshot_iters: Optional[List[int]], n_iter: int) -> List[int]:
    if not snapshot_iters:
        return []
    out = []
    for val in snapshot_iters:
        try:
            step = int(val)
        except Exception:
            continue
        if 1 <= step <= n_iter:
            out.append(step)
    return sorted(set(out))


def _snapshot_rate_required(cfg: RingConfig) -> bool:
    try:
        n_trials = int(cfg.snapshot_rate_trials)
    except Exception:
        return False
    try:
        n_bins = int(cfg.snapshot_phase_bins)
    except Exception:
        return False
    return (n_trials > 0) and (n_bins > 0)


def _snapshot_rate_history_ok(history: Optional[Dict[str, np.ndarray]], n_snapshots: int) -> bool:
    if history is None:
        return False
    snap_rate = history.get("snapshot_rate")
    snap_phase = history.get("snapshot_phase")
    if snap_rate is None or snap_phase is None:
        return False
    try:
        sr = np.asarray(snap_rate)
        sp = np.asarray(snap_phase)
    except Exception:
        return False
    if sr.ndim != 2 or sp.ndim != 1:
        return False
    if sr.shape[0] != n_snapshots:
        return False
    if sr.shape[1] != sp.shape[0]:
        return False
    return True


def _history_int_scalar(history: Optional[Dict[str, np.ndarray]], key: str) -> Optional[int]:
    if history is None:
        return None
    val = history.get(key)
    if val is None:
        return None
    try:
        arr = np.asarray(val).ravel()
        if arr.size == 0:
            return None
        return int(arr[0])
    except Exception:
        return None


def _snapshot_rate_eval_seed(cfg: RingConfig) -> int:
    seed = cfg.snapshot_rate_seed
    if seed is None:
        seed = cfg.seed
    try:
        return int(seed)
    except Exception:
        return int(cfg.seed)


def _snapshot_rate_history_matches_cfg(
    history: Optional[Dict[str, np.ndarray]],
    cfg: RingConfig,
    n_snapshots: int,
) -> bool:
    if not _snapshot_rate_history_ok(history, n_snapshots):
        return False
    expected_trials = int(cfg.snapshot_rate_trials)
    expected_bins = int(cfg.snapshot_phase_bins)
    expected_seed = _snapshot_rate_eval_seed(cfg)
    trials_saved = _history_int_scalar(history, "snapshot_rate_trials")
    bins_saved = _history_int_scalar(history, "snapshot_phase_bins")
    seed_saved = _history_int_scalar(history, "snapshot_rate_seed")
    return (
        (trials_saved == expected_trials)
        and (bins_saved == expected_bins)
        and (seed_saved == expected_seed)
    )


def _recompute_snapshot_rate_history(
    cfg: RingConfig,
    history: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    if not _snapshot_rate_required(cfg):
        return {}
    snap_iters_raw = history.get("snapshot_iters")
    snap_w0_raw = history.get("snapshot_w0")
    snap_U_raw = history.get("snapshot_U")
    if snap_iters_raw is None or snap_w0_raw is None or snap_U_raw is None:
        return {}

    try:
        snap_iters = np.asarray(snap_iters_raw, dtype=int).ravel()
        snap_w0 = np.asarray(snap_w0_raw, dtype=float)
        snap_U = np.asarray(snap_U_raw, dtype=float)
    except Exception:
        return {}
    if snap_iters.size == 0:
        return {}
    if snap_w0.ndim != 2 or snap_U.ndim != 2:
        return {}
    if snap_w0.shape[0] != snap_iters.size or snap_U.shape[0] != snap_iters.size:
        return {}

    n_trials = int(cfg.snapshot_rate_trials)
    n_bins = int(cfg.snapshot_phase_bins)
    seed_base = _snapshot_rate_eval_seed(cfg)
    snapshot_rate: List[np.ndarray] = []
    snapshot_phase: Optional[np.ndarray] = None
    for row, it_val in enumerate(snap_iters):
        spikes_trials = sample_spikes_trials(
            cfg,
            snap_w0[row],
            snap_U[row],
            n_trials=n_trials,
            seed=seed_base + int(it_val),
        )
        phase_snap, rate_snap = phase_aligned_rate(
            spikes_trials,
            cfg,
            n_phase_bins=n_bins,
        )
        snapshot_rate.append(np.asarray(rate_snap, dtype=float))
        if snapshot_phase is None:
            snapshot_phase = np.asarray(phase_snap, dtype=float)

    if snapshot_phase is None or len(snapshot_rate) != snap_iters.size:
        return {}
    return {
        "snapshot_phase": np.asarray(snapshot_phase, dtype=float),
        "snapshot_rate": np.stack(snapshot_rate, axis=0),
        "snapshot_rate_trials": np.array(n_trials, dtype=int),
        "snapshot_phase_bins": np.array(n_bins, dtype=int),
        "snapshot_rate_seed": np.array(seed_base, dtype=int),
    }


def ensure_fit_result_snapshot_rate_history(
    cfg_name: str,
    cfg: RingConfig,
    res: FitResult,
    save_dir: Path = DEFAULT_SAVE_DIR,
    verbose: bool = True,
) -> Tuple[FitResult, bool, Path]:
    """
    Ensure snapshot phase-aligned rate history is present and matches current analysis params.
    Missing/mismatched analysis history is recomputed from cached snapshot_w0/U without retraining.
    """
    path = _run_path(cfg_name, save_dir)
    if not _snapshot_rate_required(cfg):
        return res, False, path
    snap_iters = res.history.get("snapshot_iters") if res.history else None
    if snap_iters is None:
        return res, False, path
    try:
        n_snapshots = int(np.asarray(snap_iters).size)
    except Exception:
        return res, False, path
    if n_snapshots <= 0:
        return res, False, path
    if _snapshot_rate_history_matches_cfg(res.history, cfg, n_snapshots):
        return res, False, path

    add_history = _recompute_snapshot_rate_history(cfg, res.history)
    if not add_history:
        return res, False, path

    merged_history = dict(res.history)
    merged_history.update(add_history)
    updated = FitResult(
        w0=np.asarray(res.w0, dtype=float).copy(),
        U=np.asarray(res.U, dtype=float).copy(),
        history=merged_history,
    )
    save_path = save_fit_result(cfg_name, cfg, updated, save_dir=save_dir)
    if verbose:
        print(f"Updated snapshot-rate history for analysis params and saved cache: {save_path}")
    return updated, True, save_path


def fit_exact_grad(cfg: RingConfig, snapshot_iters: Optional[List[int]] = None) -> FitResult:
    rng = np.random.default_rng(cfg.seed)
    w0, U = init_params(cfg)
    w0_sum_target: Optional[float] = None
    w0_l2_radius: Optional[float] = None
    U_sum_target: Optional[float] = None
    if cfg.constrain_w0_sum:
        w0_sum_target = float(np.sum(w0)) if cfg.w0_sum_target is None else float(cfg.w0_sum_target)
    if cfg.constrain_w0_l2:
        n_scale = np.sqrt(float(max(int(w0.size), 1)))
        if cfg.w0_l2_rms_max is None:
            w0_l2_rms_max = float(np.linalg.norm(w0)) / n_scale
        else:
            w0_l2_rms_max = float(cfg.w0_l2_rms_max)
        if (not np.isfinite(w0_l2_rms_max)) or (w0_l2_rms_max < 0.0):
            raise ValueError(f"w0_l2_rms_max must be finite and >= 0, got {cfg.w0_l2_rms_max}")
        w0_l2_radius = w0_l2_rms_max * n_scale
    if cfg.constrain_w0_sum and cfg.constrain_w0_l2:
        assert w0_sum_target is not None and w0_l2_radius is not None
        w0 = project_sum_l2_ball(w0, w0_sum_target, w0_l2_radius)
    elif cfg.constrain_w0_sum:
        assert w0_sum_target is not None
        w0 = project_sum(w0, w0_sum_target)
    elif cfg.constrain_w0_l2:
        assert w0_l2_radius is not None
        w0 = project_l2_ball(w0, w0_l2_radius)
    if cfg.constrain_U_sum:
        U_sum_target = float(np.sum(U)) if cfg.U_sum_target is None else float(cfg.U_sum_target)
        U = project_box_sum(U, cfg.U_lo, cfg.U_hi, U_sum_target)
    invN = 1.0 / float(cfg.N)
    n_steps = int(np.round(cfg.T / cfg.dt))
    duration = n_steps * cfg.dt
    norm = float(cfg.N) * duration
    # Normalize objective to per-neuron per-time so it matches mean-rate scale.
    inv_norm = 1.0 / norm if norm > 0.0 else 0.0
    learn_U = bool(cfg.stp_enabled)  # no-STP runs keep U fixed
    momentum = float(cfg.momentum)
    if not np.isfinite(momentum) or momentum < 0.0 or momentum >= 1.0:
        raise ValueError(f"momentum must satisfy 0 <= momentum < 1, got {cfg.momentum}")
    use_nesterov = bool(cfg.nesterov)
    lr_w0_base = float(cfg.lr_w0)
    lr_U_base = float(cfg.lr_U)
    adaptive_lr = bool(cfg.adaptive_lr_blockwise)
    if adaptive_lr:
        lr_beta = float(cfg.adaptive_lr_beta)
        lr_eps = float(cfg.adaptive_lr_eps)
        lr_w0_scale_min = float(cfg.adaptive_lr_w0_scale_min)
        lr_w0_scale_max = float(cfg.adaptive_lr_w0_scale_max)
        lr_U_scale_min = float(cfg.adaptive_lr_U_scale_min)
        lr_U_scale_max = float(cfg.adaptive_lr_U_scale_max)
        if (not np.isfinite(lr_beta)) or lr_beta < 0.0 or lr_beta >= 1.0:
            raise ValueError(f"adaptive_lr_beta must satisfy 0 <= beta < 1, got {cfg.adaptive_lr_beta}")
        if (not np.isfinite(lr_eps)) or lr_eps <= 0.0:
            raise ValueError(f"adaptive_lr_eps must be positive finite, got {cfg.adaptive_lr_eps}")
        if (not np.isfinite(lr_w0_scale_min)) or (not np.isfinite(lr_w0_scale_max)) or lr_w0_scale_min <= 0.0 or lr_w0_scale_min > lr_w0_scale_max:
            raise ValueError(
                "adaptive_lr_w0_scale_min/max must be finite with 0 < min <= max, "
                f"got ({cfg.adaptive_lr_w0_scale_min}, {cfg.adaptive_lr_w0_scale_max})"
            )
        if (not np.isfinite(lr_U_scale_min)) or (not np.isfinite(lr_U_scale_max)) or lr_U_scale_min <= 0.0 or lr_U_scale_min > lr_U_scale_max:
            raise ValueError(
                "adaptive_lr_U_scale_min/max must be finite with 0 < min <= max, "
                f"got ({cfg.adaptive_lr_U_scale_min}, {cfg.adaptive_lr_U_scale_max})"
            )
    else:
        lr_beta = 0.99
        lr_eps = 1e-8
        lr_w0_scale_min = 1.0
        lr_w0_scale_max = 1.0
        lr_U_scale_min = 1.0
        lr_U_scale_max = 1.0
    ema_g2_w0 = 0.0
    ema_g2_U = 0.0
    vel_w0 = np.zeros_like(w0)
    vel_U = np.zeros_like(U)

    rate_reg_lambda = float(cfg.rate_reg_lambda)
    rate_reg_type = _normalize_rate_reg_type(cfg.rate_reg_type)
    rate_target = cfg.rate_target
    if rate_reg_lambda > 0.0 and rate_target is None:
        if rate_reg_type == "kl":
            rate_target = _DEFAULT_KL_RATE_TARGET_HZ
        else:
            try:
                n_target_trials = int(cfg.rate_target_trials)
            except Exception:
                n_target_trials = 1
            n_target_trials = max(1, n_target_trials)
            rate_int_acc = 0.0
            rate_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 92831]))
            for _ in range(n_target_trials):
                out = simulate_trial_exact_grad(w0, U, cfg, rate_rng, record_spikes=False)
                assert out.rate_int is not None
                rate_int_acc += out.rate_int
            rate_target = (rate_int_acc / float(n_target_trials)) * inv_norm

    if rate_reg_lambda > 0.0:
        if rate_target is None:
            raise ValueError("rate_target must be set or inferred when rate_reg_lambda > 0")
        rate_target = float(rate_target)
        if rate_reg_type == "kl":
            if (not np.isfinite(rate_target)) or rate_target <= 0.0:
                raise ValueError(f"KL regularization requires positive finite rate_target, got {rate_target}")

    hist_J = []
    hist_gnorm_w0 = []
    hist_gnorm_U = []
    hist_gnorm_w0_A = []
    hist_gnorm_w0_score = []
    hist_gnorm_U_A = []
    hist_gnorm_U_score = []
    hist_gnorm_w0_A_fi = []
    hist_gnorm_w0_A_rate = []
    hist_gnorm_U_A_fi = []
    hist_gnorm_U_A_rate = []
    hist_gnorm_w0_score_fi = []
    hist_gnorm_w0_score_rate = []
    hist_gnorm_U_score_fi = []
    hist_gnorm_U_score_rate = []
    hist_fi_rate = []
    hist_spike_rate = []
    hist_spike_rate_h0 = []
    hist_rate_mean = []
    hist_rate_error = []
    hist_rate_penalty = []
    hist_total_loss = []
    hist_w0_mean = []
    hist_w0_std = []
    hist_w0_l2 = []
    hist_U_mean = []
    hist_U_std = []
    hist_lr_w0 = []
    hist_lr_U = []
    hist_lr_w0_scale = []
    hist_lr_U_scale = []

    try:
        score_median_window = int(cfg.score_median_clip_window)
    except Exception:
        score_median_window = 0
    if score_median_window <= 0:
        score_median_window = 0
    score_hist_w0 = deque(maxlen=score_median_window) if score_median_window > 0 else None
    score_hist_U = deque(maxlen=score_median_window) if score_median_window > 0 else None

    snapshot_iters_norm = _normalize_snapshot_iters(snapshot_iters, cfg.n_iter)
    snapshot_iters_set = set(snapshot_iters_norm)
    snapshot_w0 = []
    snapshot_U = []
    snapshot_gw0 = []
    snapshot_gU = []
    snapshot_iter = []
    snapshot_rate = []
    snapshot_phase = None
    try:
        snapshot_rate_trials = int(cfg.snapshot_rate_trials)
    except Exception:
        snapshot_rate_trials = 0
    try:
        snapshot_phase_bins = int(cfg.snapshot_phase_bins)
    except Exception:
        snapshot_phase_bins = 0
    snapshot_rate_seed = _snapshot_rate_eval_seed(cfg)

    use_parallel = bool(cfg.parallel_batch and cfg.batch_size > 1)
    pool = None
    if use_parallel:
        ctx = _mp_context()
        n_workers = cfg.num_workers or _default_worker_count()
        n_workers = max(1, min(int(n_workers), cfg.batch_size))
        pool = ctx.Pool(processes=n_workers)
        print(f"Parallel batch enabled: {n_workers} worker(s) for batch_size={cfg.batch_size}")
    if momentum > 0.0:
        accel = "Nesterov" if use_nesterov else "classical momentum"
        print(f"Optimizer: SGD + {accel} (momentum={momentum:.3f})")
    if adaptive_lr:
        print(
            "Adaptive LR: block-wise scalar RMSProp "
            f"(beta={lr_beta:.3f}, eps={lr_eps:.1e}, "
            f"w0_scale=[{lr_w0_scale_min:.3g},{lr_w0_scale_max:.3g}], "
            f"U_scale=[{lr_U_scale_min:.3g},{lr_U_scale_max:.3g}])"
        )

    try:
        for it in range(cfg.n_iter):
            # Monte Carlo batch
            J_trials: List[float] = []
            A_w0_trials: List[np.ndarray] = []
            score_base_w0_trials: List[np.ndarray] = []
            A_U_trials: List[np.ndarray] = []
            score_base_U_trials: List[np.ndarray] = []
            B_w0_trials: List[np.ndarray] = []
            B_U_trials: List[np.ndarray] = []
            rate_int_trials: List[float] = []
            spike_total_trials: List[float] = []
            spike_log_rho_trials: List[float] = []
            spike_total_batch = 0.0
            spike_h0_batch = 0.0

            if use_parallel and pool is not None:
                seed_seq = np.random.SeedSequence([cfg.seed, it])
                child_seeds = seed_seq.generate_state(cfg.batch_size, dtype=np.uint32)
                args: List[Tuple[np.ndarray, np.ndarray, RingConfig, int]] = [
                    (w0, U, cfg, int(child_seeds[b])) for b in range(cfg.batch_size)
                ]
                results = pool.map(_run_trial_worker, args)
                for J_hat, A_w0, score_base_w0, A_U, score_base_U, B_w0, B_U, spike_count_total, spike_count_h0, rate_int, spike_log_rho_sum in results:
                    J_trials.append(float(J_hat))
                    A_w0_trials.append(A_w0)
                    score_base_w0_trials.append(score_base_w0)
                    A_U_trials.append(A_U)
                    score_base_U_trials.append(score_base_U)
                    B_w0_trials.append(B_w0)
                    B_U_trials.append(B_U)
                    rate_int_trials.append(float(rate_int))
                    spike_total_trials.append(float(spike_count_total))
                    spike_log_rho_trials.append(float(spike_log_rho_sum))
                    spike_total_batch += spike_count_total
                    spike_h0_batch += spike_count_h0
            else:
                for _ in range(cfg.batch_size):
                    out = simulate_trial_exact_grad(w0, U, cfg, rng, record_spikes=False)
                    J_trials.append(out.J_hat)
                    assert out.A_w0 is not None and out.score_base_w0 is not None
                    assert out.A_U is not None and out.score_base_U is not None
                    assert out.B_w0 is not None and out.B_U is not None
                    assert out.rate_int is not None
                    assert out.spike_count_total is not None and out.spike_count_h0 is not None
                    assert out.spike_log_rho_sum is not None
                    A_w0_trials.append(out.A_w0)
                    score_base_w0_trials.append(out.score_base_w0)
                    A_U_trials.append(out.A_U)
                    score_base_U_trials.append(out.score_base_U)
                    B_w0_trials.append(out.B_w0)
                    B_U_trials.append(out.B_U)
                    rate_int_trials.append(out.rate_int)
                    spike_total_trials.append(out.spike_count_total)
                    spike_log_rho_trials.append(out.spike_log_rho_sum)
                    spike_total_batch += out.spike_count_total
                    spike_h0_batch += out.spike_count_h0

            J_trials = np.array(J_trials, dtype=float)
            if J_trials.size == 0:
                continue
            batch_size_eff = int(J_trials.size)
            J_trials *= inv_norm
            J_batch = float(J_trials.mean())
            spike_total_batch /= batch_size_eff
            spike_h0_batch /= batch_size_eff

            A_w0_trials = np.stack(A_w0_trials, axis=0)
            score_base_w0_trials = np.stack(score_base_w0_trials, axis=0)
            A_U_trials = np.stack(A_U_trials, axis=0)
            score_base_U_trials = np.stack(score_base_U_trials, axis=0)
            B_w0_trials = np.stack(B_w0_trials, axis=0)
            B_U_trials = np.stack(B_U_trials, axis=0)
            rate_int_trials = np.array(rate_int_trials, dtype=float)
            spike_total_trials = np.array(spike_total_trials, dtype=float)
            spike_log_rho_trials = np.array(spike_log_rho_trials, dtype=float)

            A_w0_trials *= inv_norm
            A_U_trials *= inv_norm

            # Leave-one-out (LOO) batch baseline for FI score-term control variate
            # (fallback to zero baseline when batch_size=1).
            if batch_size_eff > 1:
                J_sum = float(J_trials.sum())
                b_trials = (J_sum - J_trials) / float(batch_size_eff - 1)
            else:
                b_trials = np.zeros_like(J_trials)
            fi_score_w0_trials = (J_trials - b_trials)[:, None] * score_base_w0_trials
            fi_score_U_trials = (J_trials - b_trials)[:, None] * score_base_U_trials

            reg_w0_trials = np.zeros_like(A_w0_trials)
            reg_U_trials = np.zeros_like(A_U_trials)
            reg_score_w0_trials = np.zeros_like(score_base_w0_trials)
            reg_score_U_trials = np.zeros_like(score_base_U_trials)
            kl_div_batch = float("nan")

            if rate_reg_lambda > 0.0:
                rate_bar_trials = rate_int_trials * inv_norm
                target_rate = float(rate_target)
                if rate_reg_type == "kl":
                    log_target = np.log(target_rate)
                    kl_div_trials = (
                        spike_log_rho_trials
                        - spike_total_trials * log_target
                        - rate_int_trials
                        + norm * target_rate
                    ) * inv_norm
                    kl_div_batch = float(np.mean(kl_div_trials))
                    # KL regularization contributes only through the score term:
                    # d/dZ E[F_KL] = E[(F_KL-b) * d/dZ log P(X|Z)] when target rate is Z-independent.
                    reg_obj_trials = -rate_reg_lambda * kl_div_trials
                else:
                    rate_err_trials = rate_bar_trials - target_rate
                    reg_w0_trials = -(rate_reg_lambda * rate_err_trials)[:, None] * (B_w0_trials * inv_norm)
                    reg_U_trials = -(rate_reg_lambda * rate_err_trials)[:, None] * (B_U_trials * inv_norm)
                    # Exact score contribution for the L2 mean-rate penalty term:
                    #   L_rate = -0.5 * lambda * (rate - target)^2
                    reg_obj_trials = -0.5 * rate_reg_lambda * (rate_err_trials ** 2)
                if batch_size_eff > 1:
                    reg_sum = float(reg_obj_trials.sum())
                    reg_b_trials = (reg_sum - reg_obj_trials) / float(batch_size_eff - 1)
                else:
                    reg_b_trials = np.zeros_like(reg_obj_trials)
                reg_score_w0_trials = (reg_obj_trials - reg_b_trials)[:, None] * score_base_w0_trials
                reg_score_U_trials = (reg_obj_trials - reg_b_trials)[:, None] * score_base_U_trials

            direct_w0_trials = A_w0_trials + reg_w0_trials
            direct_U_trials = A_U_trials + reg_U_trials
            score_w0_trials = fi_score_w0_trials + reg_score_w0_trials
            score_U_trials = fi_score_U_trials + reg_score_U_trials

            # Per-trial score-term norm clipping based on batch percentile.
            score_w0_trials = _clip_score_trials(score_w0_trials, cfg.score_clip_percentile)
            score_U_trials = _clip_score_trials(score_U_trials, cfg.score_clip_percentile)
            if score_hist_w0 is not None:
                median_w0 = _score_norm_median(score_w0_trials)
                if median_w0 is not None and np.isfinite(median_w0):
                    score_hist_w0.append(median_w0)
                clip_norm_w0 = _score_median_clip_norm(score_hist_w0, cfg.score_median_clip_mult)
                score_w0_trials = _clip_score_trials_by_norm(score_w0_trials, clip_norm_w0)
            if score_hist_U is not None:
                median_U = _score_norm_median(score_U_trials)
                if median_U is not None and np.isfinite(median_U):
                    score_hist_U.append(median_U)
                clip_norm_U = _score_median_clip_norm(score_hist_U, cfg.score_median_clip_mult)
                score_U_trials = _clip_score_trials_by_norm(score_U_trials, clip_norm_U)

            A_w0_batch = A_w0_trials.mean(axis=0)
            A_U_batch = A_U_trials.mean(axis=0)
            reg_w0_batch = reg_w0_trials.mean(axis=0)
            reg_U_batch = reg_U_trials.mean(axis=0)
            direct_w0_batch = direct_w0_trials.mean(axis=0)
            direct_U_batch = direct_U_trials.mean(axis=0)
            fi_score_w0_batch = fi_score_w0_trials.mean(axis=0)
            fi_score_U_batch = fi_score_U_trials.mean(axis=0)
            reg_score_w0_batch = reg_score_w0_trials.mean(axis=0)
            reg_score_U_batch = reg_score_U_trials.mean(axis=0)
            score_w0_batch = score_w0_trials.mean(axis=0)
            score_U_batch = score_U_trials.mean(axis=0)

            gw0 = (direct_w0_batch + score_w0_batch) * invN
            gU  = (direct_U_batch + score_U_batch) * invN

            if norm > 0.0:
                fi_rate = J_batch
                spike_rate = spike_total_batch / norm
                spike_rate_h0 = spike_h0_batch / norm
            else:
                fi_rate = float("nan")
                spike_rate = float("nan")
                spike_rate_h0 = float("nan")

            w0_mean = float(np.mean(w0))
            w0_std = float(np.std(w0))
            w0_l2 = float(np.linalg.norm(w0))
            U_mean = float(np.mean(U))
            U_std = float(np.std(U))

            if adaptive_lr:
                lr_w0_scale, ema_g2_w0 = blockwise_rmsprop_scale(
                    gw0,
                    ema_g2_prev=ema_g2_w0,
                    beta=lr_beta,
                    eps=lr_eps,
                    scale_min=lr_w0_scale_min,
                    scale_max=lr_w0_scale_max,
                )
                lr_w0_t = lr_w0_base * lr_w0_scale
                if learn_U:
                    lr_U_scale, ema_g2_U = blockwise_rmsprop_scale(
                        gU,
                        ema_g2_prev=ema_g2_U,
                        beta=lr_beta,
                        eps=lr_eps,
                        scale_min=lr_U_scale_min,
                        scale_max=lr_U_scale_max,
                    )
                else:
                    lr_U_scale = 1.0
                lr_U_t = lr_U_base * lr_U_scale
            else:
                lr_w0_t = lr_w0_base
                lr_U_t = lr_U_base
                lr_w0_scale = 1.0
                lr_U_scale = 1.0

            # SGD (+ optional momentum / Nesterov) update
            w0, vel_w0 = sgd_momentum_step(
                w0,
                gw0,
                velocity=vel_w0,
                lr=lr_w0_t,
                momentum=momentum,
                nesterov=use_nesterov,
                update_clip=cfg.update_clip,
            )
            if cfg.constrain_w0_sum and cfg.constrain_w0_l2:
                assert w0_sum_target is not None and w0_l2_radius is not None
                w0 = project_sum_l2_ball(w0, w0_sum_target, w0_l2_radius)
            elif cfg.constrain_w0_sum:
                assert w0_sum_target is not None
                w0 = project_sum(w0, w0_sum_target)
            elif cfg.constrain_w0_l2:
                assert w0_l2_radius is not None
                w0 = project_l2_ball(w0, w0_l2_radius)
            if learn_U:
                U, vel_U = sgd_momentum_step(
                    U,
                    gU,
                    velocity=vel_U,
                    lr=lr_U_t,
                    momentum=momentum,
                    nesterov=use_nesterov,
                    update_clip=cfg.update_clip,
                )
            if cfg.constrain_U_sum:
                U = project_box_sum(U, cfg.U_lo, cfg.U_hi, U_sum_target)
            elif learn_U:
                U = np.clip(U, cfg.U_lo, cfg.U_hi)

            if snapshot_iters_set and (it + 1) in snapshot_iters_set:
                snapshot_w0.append(w0.copy())
                snapshot_U.append(U.copy())
                # Grad snapshots correspond to the batch gradient used for this update.
                snapshot_gw0.append(gw0.copy())
                snapshot_gU.append(gU.copy())
                snapshot_iter.append(it + 1)
                if (snapshot_rate_trials > 0) and (snapshot_phase_bins > 0):
                    snap_seed = snapshot_rate_seed + int(it + 1)
                    spikes_trials = sample_spikes_trials(
                        cfg,
                        w0,
                        U,
                        n_trials=snapshot_rate_trials,
                        seed=snap_seed,
                    )
                    phase_snap, rate_snap = phase_aligned_rate(
                        spikes_trials,
                        cfg,
                        n_phase_bins=snapshot_phase_bins,
                    )
                    snapshot_rate.append(rate_snap)
                    if snapshot_phase is None:
                        snapshot_phase = phase_snap

            hist_J.append(J_batch)
            hist_gnorm_w0.append(float(np.linalg.norm(gw0)))
            hist_gnorm_U.append(float(np.linalg.norm(gU)))
            hist_gnorm_w0_A.append(float(np.linalg.norm(direct_w0_batch) * invN))
            hist_gnorm_w0_score.append(float(np.linalg.norm(score_w0_batch) * invN))
            hist_gnorm_U_A.append(float(np.linalg.norm(direct_U_batch) * invN))
            hist_gnorm_U_score.append(float(np.linalg.norm(score_U_batch) * invN))
            hist_gnorm_w0_A_fi.append(float(np.linalg.norm(A_w0_batch) * invN))
            hist_gnorm_w0_A_rate.append(float(np.linalg.norm(reg_w0_batch) * invN))
            hist_gnorm_U_A_fi.append(float(np.linalg.norm(A_U_batch) * invN))
            hist_gnorm_U_A_rate.append(float(np.linalg.norm(reg_U_batch) * invN))
            hist_gnorm_w0_score_fi.append(float(np.linalg.norm(fi_score_w0_batch) * invN))
            hist_gnorm_w0_score_rate.append(float(np.linalg.norm(reg_score_w0_batch) * invN))
            hist_gnorm_U_score_fi.append(float(np.linalg.norm(fi_score_U_batch) * invN))
            hist_gnorm_U_score_rate.append(float(np.linalg.norm(reg_score_U_batch) * invN))
            hist_fi_rate.append(float(fi_rate))
            hist_spike_rate.append(float(spike_rate))
            hist_spike_rate_h0.append(float(spike_rate_h0))
            if rate_int_trials.size > 0 and norm > 0.0:
                rate_mean = float(rate_int_trials.mean() * inv_norm)
            else:
                rate_mean = float("nan")
            if rate_target is None or not np.isfinite(rate_mean):
                rate_error = float("nan")
            else:
                rate_error = float(rate_mean - float(rate_target))
            if rate_reg_lambda > 0.0:
                if rate_reg_type == "kl":
                    rate_penalty = rate_reg_lambda * kl_div_batch if np.isfinite(kl_div_batch) else float("nan")
                elif np.isfinite(rate_error):
                    rate_penalty = 0.5 * rate_reg_lambda * (rate_error ** 2)
                else:
                    rate_penalty = float("nan")
            else:
                rate_penalty = float("nan") if not np.isfinite(rate_error) else 0.0
            if np.isfinite(fi_rate) and np.isfinite(rate_penalty):
                total_loss = float(fi_rate - rate_penalty)
            else:
                total_loss = float("nan")
            hist_rate_mean.append(rate_mean)
            hist_rate_error.append(rate_error)
            hist_rate_penalty.append(rate_penalty)
            hist_total_loss.append(total_loss)
            hist_w0_mean.append(w0_mean)
            hist_w0_std.append(w0_std)
            hist_w0_l2.append(w0_l2)
            hist_U_mean.append(U_mean)
            hist_U_std.append(U_std)
            hist_lr_w0.append(float(lr_w0_t))
            hist_lr_U.append(float(lr_U_t))
            hist_lr_w0_scale.append(float(lr_w0_scale))
            hist_lr_U_scale.append(float(lr_U_scale))

            if (it + 1) % max(1, cfg.n_iter // 10) == 0:
                print(f"[{it+1:4d}/{cfg.n_iter}]  J~{J_batch:.3e}  ||g_w0||={hist_gnorm_w0[-1]:.2e}  ||g_U||={hist_gnorm_U[-1]:.2e}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    history = {
        "J": np.array(hist_J, dtype=float),
        "gnorm_w0": np.array(hist_gnorm_w0, dtype=float),
        "gnorm_U": np.array(hist_gnorm_U, dtype=float),
        "gnorm_w0_A": np.array(hist_gnorm_w0_A, dtype=float),
        "gnorm_w0_score": np.array(hist_gnorm_w0_score, dtype=float),
        "gnorm_U_A": np.array(hist_gnorm_U_A, dtype=float),
        "gnorm_U_score": np.array(hist_gnorm_U_score, dtype=float),
        "gnorm_w0_A_fi": np.array(hist_gnorm_w0_A_fi, dtype=float),
        "gnorm_w0_A_rate": np.array(hist_gnorm_w0_A_rate, dtype=float),
        "gnorm_U_A_fi": np.array(hist_gnorm_U_A_fi, dtype=float),
        "gnorm_U_A_rate": np.array(hist_gnorm_U_A_rate, dtype=float),
        "gnorm_w0_score_fi": np.array(hist_gnorm_w0_score_fi, dtype=float),
        "gnorm_w0_score_rate": np.array(hist_gnorm_w0_score_rate, dtype=float),
        "gnorm_U_score_fi": np.array(hist_gnorm_U_score_fi, dtype=float),
        "gnorm_U_score_rate": np.array(hist_gnorm_U_score_rate, dtype=float),
        "fi_rate": np.array(hist_fi_rate, dtype=float),
        "spike_rate": np.array(hist_spike_rate, dtype=float),
        "spike_rate_h0": np.array(hist_spike_rate_h0, dtype=float),
        "rate_mean": np.array(hist_rate_mean, dtype=float),
        "rate_error": np.array(hist_rate_error, dtype=float),
        "rate_penalty": np.array(hist_rate_penalty, dtype=float),
        "total_loss": np.array(hist_total_loss, dtype=float),
        "w0_mean": np.array(hist_w0_mean, dtype=float),
        "w0_std": np.array(hist_w0_std, dtype=float),
        "w0_l2": np.array(hist_w0_l2, dtype=float),
        "U_mean": np.array(hist_U_mean, dtype=float),
        "U_std": np.array(hist_U_std, dtype=float),
        "lr_w0": np.array(hist_lr_w0, dtype=float),
        "lr_U": np.array(hist_lr_U, dtype=float),
        "lr_w0_scale": np.array(hist_lr_w0_scale, dtype=float),
        "lr_U_scale": np.array(hist_lr_U_scale, dtype=float),
    }
    if snapshot_iters_set and snapshot_iter:
        history["snapshot_iters"] = np.array(snapshot_iter, dtype=int)
        history["snapshot_w0"] = np.stack(snapshot_w0, axis=0)
        history["snapshot_U"] = np.stack(snapshot_U, axis=0)
        history["snapshot_gw0"] = np.stack(snapshot_gw0, axis=0)
        history["snapshot_gU"] = np.stack(snapshot_gU, axis=0)
        if snapshot_rate and snapshot_phase is not None and len(snapshot_rate) == len(snapshot_iter):
            history["snapshot_phase"] = np.array(snapshot_phase, dtype=float)
            history["snapshot_rate"] = np.stack(snapshot_rate, axis=0)
            history["snapshot_rate_trials"] = np.array(snapshot_rate_trials, dtype=int)
            history["snapshot_phase_bins"] = np.array(snapshot_phase_bins, dtype=int)
            history["snapshot_rate_seed"] = np.array(snapshot_rate_seed, dtype=int)
    phase_asymmetry_history = _compute_phase_asymmetry_history(cfg, w0, U)
    if phase_asymmetry_history:
        history.update(phase_asymmetry_history)
    return FitResult(w0=w0, U=U, history=history)


# --------------------------
# Training driver + evaluation helpers
# --------------------------

def get_fit_result(
    cfg: RingConfig,
    cfg_name: str = "default",
    force_retrain: bool = False,
    snapshot_iters: Optional[List[int]] = None,
    require_snapshots: bool = False,
) -> FitResult:
    cached, status, path = try_load_fit_result(cfg_name, cfg)
    if cached is not None and not force_retrain:
        cached, _, path = ensure_fit_result_asymmetry_history(
            cfg_name,
            cfg,
            cached,
            save_dir=DEFAULT_SAVE_DIR,
            verbose=True,
        )
        retrain_for_snapshots = False
        if snapshot_iters and require_snapshots:
            requested = _normalize_snapshot_iters(snapshot_iters, cfg.n_iter)
            cached_iters = cached.history.get("snapshot_iters") if cached.history else None
            if requested:
                if cached_iters is None:
                    retrain_for_snapshots = True
                    status = "missing_snapshots"
                else:
                    have = set()
                    for v in np.asarray(cached_iters).tolist():
                        try:
                            have.add(int(v))
                        except Exception:
                            continue
                    missing = [step for step in requested if step not in have]
                    if missing:
                        retrain_for_snapshots = True
                        status = "missing_snapshots"
                    else:
                        if _snapshot_rate_required(cfg):
                            n_snapshots = len(cached_iters)
                            if not _snapshot_rate_history_matches_cfg(cached.history, cfg, n_snapshots):
                                cached, _, path = ensure_fit_result_snapshot_rate_history(
                                    cfg_name,
                                    cfg,
                                    cached,
                                    save_dir=DEFAULT_SAVE_DIR,
                                    verbose=True,
                                )
                            if not _snapshot_rate_history_matches_cfg(cached.history, cfg, n_snapshots):
                                retrain_for_snapshots = True
                                status = "missing_snapshots"
                            else:
                                print(f"Loaded cached run '{cfg_name}' from {path}")
                                return cached
                        else:
                            print(f"Loaded cached run '{cfg_name}' from {path}")
                            return cached
            else:
                print(f"Loaded cached run '{cfg_name}' from {path}")
                return cached
        else:
            print(f"Loaded cached run '{cfg_name}' from {path}")
            return cached
        if retrain_for_snapshots:
            cached = None

    if cached is None:
        if status == "missing_snapshots":
            print(f"Cached run '{cfg_name}' missing requested snapshots; retraining.")
        elif status == "missing":
            print(f"No cached run named '{cfg_name}' at {path}; training from scratch.")
        elif status == "config_mismatch":
            print(f"Cached run at {path} uses a different config; retraining.")
        else:
            print(f"Could not use cache at {path} (status={status}); retraining.")
    else:
        print(f"force_retrain=True -> ignoring cached run at {path}.")

    print("Fitting (joint w0 & U)...")
    res = fit_exact_grad(cfg, snapshot_iters=snapshot_iters)
    save_path = save_fit_result(cfg_name, cfg, res)
    print(f"Saved run to {save_path}")
    return res


def evaluate_FI(cfg: RingConfig, w0: np.ndarray, U: np.ndarray, n_trials: int = 8, seed: int = 1234) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_steps = int(np.round(cfg.T / cfg.dt))
    duration = n_steps * cfg.dt
    norm = float(cfg.N) * duration
    if norm <= 0.0:
        raise ValueError("Fisher information normalization requires positive duration and N.")
    Js = []
    for _ in range(n_trials):
        out = simulate_trial_exact_grad(w0, U, cfg, rng, record_spikes=False)
        Js.append(out.J_hat / norm)
    Js = np.array(Js, dtype=float)
    return float(Js.mean()), float(Js.std(ddof=1) / np.sqrt(len(Js)))


def sample_spikes(cfg: RingConfig, w0: np.ndarray, U: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = simulate_trial_exact_grad(w0, U, cfg, rng, record_spikes=True)
    assert out.spikes_bin is not None
    return out.spikes_bin


def sample_spikes_trials(
    cfg: RingConfig,
    w0: np.ndarray,
    U: np.ndarray,
    n_trials: int,
    seed: int = 0,
) -> np.ndarray:
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_trials):
        out = simulate_trial_exact_grad(w0, U, cfg, rng, record_spikes=True)
        assert out.spikes_bin is not None
        trials.append(out.spikes_bin)
    return np.stack(trials, axis=0)


# --------------------------
# Replay simulation (background-only input)
# --------------------------

_REPLAY_H_BG_ATTR_BY_STP = {
    "with_stp": "h_bg_with_stp",
    "pre_stp": "h_bg_pre_stp",
    "no_stp": "h_bg_no_stp",
}


def _infer_stp_type_for_replay(cfg: RingConfig) -> str:
    if not bool(cfg.stp_enabled):
        return "no_stp"
    if np.isclose(float(cfg.lr_U), 0.0, atol=1e-12, rtol=0.0) and np.isclose(
        float(cfg.U_init_jitter), 0.0, atol=1e-12, rtol=0.0
    ):
        return "pre_stp"
    return "with_stp"


def _resolve_replay_h_bg(rp: ReplayConfig, stp_type: Optional[str]) -> float:
    if stp_type is not None:
        attr = _REPLAY_H_BG_ATTR_BY_STP.get(str(stp_type))
        if attr is not None:
            val = getattr(rp, attr, None)
            if val is not None:
                h_bg = float(val)
                if not np.isfinite(h_bg):
                    raise ValueError(f"{attr} must be finite, got {val!r}")
                return h_bg

    h_bg = float(rp.h_bg)
    if not np.isfinite(h_bg):
        raise ValueError(f"h_bg must be finite, got {rp.h_bg!r}")
    return h_bg


def simulate_replay_activity(
    cfg: RingConfig,
    w0: np.ndarray,
    U: np.ndarray,
    rp: ReplayConfig,
    seed: Optional[int] = None,
    stp_type: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulate activity with no traveling-wave drive (background + brief cue) and track replay direction.

    Returns:
      dict with binned spikes, phase trajectories, concentration, and per-trial phase velocities.
    """
    N = cfg.N
    dt = cfg.dt
    n_steps = int(np.round(rp.T / dt))
    t_sim_start = float(rp.t_start)
    cue_start_abs = t_sim_start + float(rp.cue_start)
    cue_end_abs = cue_start_abs + float(rp.cue_duration)
    z = ring_positions(N)
    k_vec = np.arange(N)
    stp_type = _infer_stp_type_for_replay(cfg) if stp_type is None else str(stp_type)
    replay_h_bg = _resolve_replay_h_bg(rp, stp_type)

    dec_s = np.exp(-dt / cfg.tau_s)
    dec_d = np.exp(-dt / cfg.tau_d)

    trials_spk = []
    trials_phase = []
    trials_R = []
    trials_vel = []
    t_bins_out = None

    base_seed = rp.seed if seed is None else seed

    # Normalize the cue bump to unit amplitude at z = 0.
    bump = (2.0 / rp.cue_theta**2) * cosine_bump(z, rp.cue_center, rp.cue_theta) if rp.cue_A != 0.0 else None

    for tr in range(rp.n_trials):
        rng = np.random.default_rng(base_seed + tr)

        S = np.zeros(N)
        D = np.ones((N, N)) if cfg.stp_enabled else None
        sU = np.zeros((N, N)) if cfg.stp_enabled else None
        spikes = np.zeros((n_steps, N), dtype=np.int8)

        for step in range(n_steps):
            t = t_sim_start + step * dt

            # background + optional brief cue
            h = np.full(N, replay_h_bg, dtype=float)
            if (bump is not None) and (cue_start_abs <= t < cue_end_abs):
                h += rp.cue_A * bump

            u = h + S
            rho = rate_from_cfg(u, cfg)

            p = 1.0 - np.exp(-rho * dt)
            spk = (rng.random(N) < p).astype(np.int8)
            spikes[step] = spk

            # decay continuous-time states
            S *= dec_s
            if cfg.stp_enabled:
                sU *= dec_d
                D = 1.0 - (1.0 - D) * dec_d

            # apply presynaptic spikes
            if spk.any():
                presyn_idx = np.flatnonzero(spk)
                for j in presyn_idx:
                    i_vec = (j - k_vec) % N
                    if cfg.stp_enabled:
                        d_pre = D[i_vec, k_vec].copy()
                        sU_pre = sU[i_vec, k_vec].copy()
                    else:
                        d_pre = np.ones_like(k_vec, dtype=float)
                        sU_pre = np.zeros_like(k_vec, dtype=float)

                    w_eff = w0[k_vec] * U[k_vec] * d_pre
                    S[i_vec] += w_eff

                    if cfg.stp_enabled:
                        sU[i_vec, k_vec] = (1.0 - U[k_vec]) * sU_pre - d_pre
                        D[i_vec, k_vec] = d_pre * (1.0 - U[k_vec])

        binned, t_bins = _bin_spikes(spikes, dt=dt, bin_dt=rp.bin_dt, t_start=t_sim_start)
        if t_bins_out is None:
            t_bins_out = t_bins

        ph_u, R = _phase_trajectory(binned, z)
        vel = _phase_velocity(ph_u, t_bins, t_min=cue_end_abs, R=R)

        trials_spk.append(binned)
        trials_phase.append(ph_u)
        trials_R.append(R)
        trials_vel.append(vel)

    return {
        "spikes_binned": np.array(trials_spk, dtype=float),  # (n_trials, Tbins, N)
        "t_binned": t_bins_out,
        "z": z,
        "phase_unwrapped": np.array(trials_phase, dtype=float),
        "concentration": np.array(trials_R, dtype=float),
        "phase_velocity": np.array(trials_vel, dtype=float),
        "bin_dt": float(rp.bin_dt),
        "t_start": t_sim_start,
        "t_end": t_sim_start + n_steps * dt,
        "cue_window": np.array([cue_start_abs, cue_end_abs], dtype=float),
    }
