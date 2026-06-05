"""
fit_scaling_law.py
==================
Fit a Data Mixing Scaling Law (Extended Linear Regression Model) to
SlimPajama LLM data using Basin-Hopping + L-BFGS-B.

GPU acceleration via JAX:
  - Inner KKT bisection is vectorised over the full batch with jax.lax.scan + vmap
  - Exact gradients via jax.value_and_grad (replaces finite-difference Jacobian)
  - JIT compilation fuses the entire forward + backward pass into a single kernel
"""

import warnings
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize

import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad

# ── GPU setup ────────────────────────────────────────────────────────────────
try:
    jax.devices("gpu")
    print("[INFO] JAX backend: GPU —", jax.devices("gpu")[0])
except RuntimeError:
    print("[WARN] No GPU found — falling back to CPU")
    print("[HINT] On this machine run via:  ./run.sh  (sets LD_PRELOAD for libcusparse)")

# Use 64-bit floats (matches scipy's L-BFGS-B expectations)
jax.config.update("jax_enable_x64", True)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Configuration
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH   = "dmsl_llm_slimpajama.csv"
K          = 7

DOMAIN_NAMES = ["arxiv", "book", "c4", "github", "commoncrawl", "stackexchange", "wikipedia"]
REDPAJAMA_NAMES = [
    "RedPajamaArXiv",          # 0: 对应 "arxiv"
    "RedPajamaBook",           # 1: 对应 "book"
    "RedPajamaC4",             # 2: 对应 "c4"
    "RedPajamaGithub",         # 3: 对应 "github"
    "RedPajamaCommonCrawl",    # 4: 对应 "commoncrawl"
    "RedPajamaStackExchange",  # 5: 对应 "stackexchange"
    "RedPajamaWikipedia",      # 6: 对应 "wikipedia"
]
WEIGHT_COLS  = [f"weight_{d}" for d in DOMAIN_NAMES]
LOSS_COLS    = [f"{d}_loss"   for d in DOMAIN_NAMES]

MODEL_SIZE_FILTER =  122000000
N_TOKENS_FILTER   =9998000000

TRAIN_FRAC        = 0.80
RANDOM_SEED       = 11514
H_EPS             = 0.1    # minimum h_j to count as "active" in metrics
H_MIN             = 1e-4   # floor for mixture-search h (avoids singular noise term)

BISECT_ITERS      = 100   # fixed bisection depth; 2^-100 ≈ 1e-30 precision

BH_NITER          = 10
BH_STEPSIZE       = 0.25
BH_T              = 1.0
BH_NITER_SUCCESS  = 60

LBFGSB_MAXITER    = 1000


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data Loading & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str):
    df   = pd.read_csv(csv_path)
    mask = (df["model_size"] == MODEL_SIZE_FILTER) & (df["n_tokens"] == N_TOKENS_FILTER)
    df_f = df[mask].reset_index(drop=True)

    if len(df_f) == 0:
        combo = df[["model_size", "n_tokens"]].drop_duplicates().iloc[0]
        mask  = (df["model_size"] == combo["model_size"]) & (df["n_tokens"] == combo["n_tokens"])
        df_f  = df[mask].reset_index(drop=True)
        print(f"[WARN] Hardcoded filter matched 0 rows. "
              f"Falling back to model_size={combo['model_size']}, n_tokens={combo['n_tokens']}")
    else:
        print(f"[INFO] Filtered to model_size={MODEL_SIZE_FILTER:.0f}, "
              f"n_tokens={N_TOKENS_FILTER:.0f}  →  {len(df_f)} rows")

    h_data = df_f[WEIGHT_COLS].values.astype(np.float64)
    L_data = df_f[LOSS_COLS].values.astype(np.float64)
    N = float(df_f["model_size"].iloc[0]) / 1e6
    D = float(df_f["n_tokens"].iloc[0])  / 1e9
    return h_data, L_data, N, D


def load_data_for_pair(csv_path: str, model_size: float, n_tokens: float):
    """
    Load and split data for a specific (model_size, n_tokens) pair.
    Returns (h_train, L_train, h_val, L_val, N, D) on the JAX device.
    """
    df   = pd.read_csv(csv_path)
    mask = (df["model_size"] == model_size) & (df["n_tokens"] == n_tokens)
    df_f = df[mask].reset_index(drop=True)
    if len(df_f) == 0:
        raise ValueError(f"No data for model_size={model_size}, n_tokens={n_tokens}")
    h_data = df_f[WEIGHT_COLS].values.astype(np.float64)
    L_data = df_f[LOSS_COLS].values.astype(np.float64)
    N = float(df_f["model_size"].iloc[0]) / 1e6
    D = float(df_f["n_tokens"].iloc[0])  / 1e9

    rng  = np.random.default_rng(RANDOM_SEED)
    idx  = rng.permutation(len(h_data))
    n_tr = int(len(h_data) * TRAIN_FRAC)
    tr, va = idx[:n_tr], idx[n_tr:]

    h_tr = jnp.array(h_data[tr]);  L_tr = jnp.array(L_data[tr])
    h_va = jnp.array(h_data[va]);  L_va = jnp.array(L_data[va])
    return h_tr, L_tr, h_va, L_va, N, D


def train_val_split(h_data, L_data, train_frac=TRAIN_FRAC, seed=RANDOM_SEED):
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(h_data))
    n_tr = int(len(h_data) * train_frac)
    tr, va = idx[:n_tr], idx[n_tr:]
    return h_data[tr], L_data[tr], h_data[va], L_data[va]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Inner Optimisation — JAX batched bisection
# ─────────────────────────────────────────────────────────────────────────────

def _solve_inner_single(h, b, c, H, N):
    """
    KKT-based inner solver for ONE mixture vector h.
    Uses a fixed-iteration bisection (jax.lax.scan) so it is JIT-compilable
    and differentiable.

    x_i(λ) = max(H, (h_i b_i c_i / λ)^{1/(b_i+1)})
    Find λ* s.t. Σ x_i(λ*) = N + (K-1)·H
    """
    capacity   = N + (K - 1) * H
    numerators = h * b * c   # (K,)

    def x_of_lam(lam):
        safe_num      = jnp.where(numerators > 0, numerators, 1.0)
        unconstrained = jnp.where(
            numerators > 0,
            (safe_num / lam) ** (1.0 / (b + 1.0)),
            0.0,
        )
        return jnp.maximum(H, unconstrained)

    def bisect_step(carry, _):
        lo, hi  = carry
        mid     = (lo + hi) * 0.5
        f_mid   = x_of_lam(mid).sum() - capacity
        lo      = jnp.where(f_mid > 0.0, mid, lo)
        hi      = jnp.where(f_mid < 0.0, mid, hi)
        return (lo, hi), None

    lo0 = jnp.float64(1e-30)
    hi0 = jnp.float64(1e30)
    (lo, hi), _ = jax.lax.scan(bisect_step, (lo0, hi0), None, length=BISECT_ITERS)
    x_star = x_of_lam((lo + hi) * 0.5)
    return jnp.where(K * H >= capacity, jnp.full(K, H), x_star)


_solve_inner_batch = vmap(_solve_inner_single, in_axes=(0, None, None, None, None))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Forward Model — batched over (M, K)
# ─────────────────────────────────────────────────────────────────────────────

def _predict_batch(h_batch, b, c, A, a, E, H, N, D):
    """
    h_batch : (M, K)
    Returns  : (M, K) predicted losses
    """
    x_star = _solve_inner_batch(h_batch, b, c, H, N)          # (M, K)
    capacity_term = c * (x_star ** (-b))                       # (M, K)
    # Use H_MIN floor: avoids singularity when h_j → 0, and correctly
    # suppresses the noise term for near-zero weights rather than substituting 1.0
    safe_h     = jnp.maximum(h_batch, H_MIN)
    noise_term = A * (D * safe_h) ** (-a)                      # (M, K)
    return capacity_term + noise_term + E                      # (M, K)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Parameter Packing / Unpacking  &  Bounds
# ─────────────────────────────────────────────────────────────────────────────
# θ layout (length = 5·K + 1 = 36):
#   [ b×K | c×K | A×K | a×K | E×K | H ]

def pack(b, c, A, a, E, H):
    return np.concatenate([b, c, A, a, E, [H]])

def _unpack_jax(theta):
    b = theta[0*K : 1*K]
    c = theta[1*K : 2*K]
    A = theta[2*K : 3*K]
    a = theta[3*K : 4*K]
    E = theta[4*K : 5*K]
    H = theta[5*K]
    return b, c, A, a, E, H

def unpack(theta):
    b, c, A, a, E, H = _unpack_jax(theta)
    return b, c, A, a, E, float(H)

def get_bounds():
    lo = 1e-5
    return (
        [(lo, None)] * K   +   # b
        [(lo, None)] * K   +   # c
        [(lo, None)] * K   +   # A
        [(lo, None)] * K   +   # a
        [(lo, None)] * K +   # E
        [(lo, None)]           # H
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Objective (MSE + exact gradient) and Metrics (MRE, MAE, RMSE) — all in JAX
# ─────────────────────────────────────────────────────────────────────────────

def _mse_jax(theta, h_train, L_train, N, D):
    """Pure JAX MSE — differentiable end-to-end."""
    b, c, A, a, E, H = _unpack_jax(theta)
    L_hat  = _predict_batch(h_train, b, c, A, a, E, H, N, D)   # (M, K)
    active = h_train >= H_EPS                                    # (M, K) bool
    sq_err = jnp.where(active, (L_hat - L_train) ** 2, 0.0)
    return sq_err.sum() / jnp.maximum(active.sum(), 1)

# JIT-compile value + gradient together (single GPU kernel)
_mse_val_and_grad = jit(value_and_grad(_mse_jax))


def objective_scipy(theta_np, h_tr_jax, L_tr_jax, N, D):
    """
    Wrapper for scipy: accepts/returns numpy arrays.
    Returns (scalar_loss, gradient_array) so L-BFGS-B gets exact gradients.
    """
    theta_jax = jnp.array(theta_np)
    val, grad = _mse_val_and_grad(theta_jax, h_tr_jax, L_tr_jax, N, D)
    return float(val), np.array(grad, dtype=np.float64)


@jit
def _mre_jax(theta, h_set, L_set, N, D):
    b, c, A, a, E, H = _unpack_jax(theta)
    L_hat  = _predict_batch(h_set, b, c, A, a, E, H, N, D)
    active = (h_set >= H_EPS) & (jnp.abs(L_set) > 1e-12)
    rel_err = jnp.where(active, jnp.abs(L_hat - L_set) / jnp.abs(L_set), 0.0)
    return rel_err.sum() / jnp.maximum(active.sum(), 1)


@jit
def _metrics_jax(theta, h_set, L_set, N, D):
    """Compute MRE, MAE, and RMSE in a single pass."""
    b, c, A, a, E, H = _unpack_jax(theta)
    L_hat  = _predict_batch(h_set, b, c, A, a, E, H, N, D)
    active = (h_set >= H_EPS) & (jnp.abs(L_set) > 1e-12)
    n      = jnp.maximum(active.sum(), 1)

    abs_err = jnp.where(active, jnp.abs(L_hat - L_set), 0.0)
    rel_err = jnp.where(active, abs_err / jnp.abs(L_set), 0.0)
    sq_err  = jnp.where(active, (L_hat - L_set) ** 2, 0.0)

    mre  = rel_err.sum() / n
    mae  = abs_err.sum() / n
    mse  = sq_err.sum()  / n
    rmse = jnp.sqrt(mse)
    return mre, mae, rmse


def mean_relative_error(theta_np, h_jax, L_jax, N, D):
    return float(_mre_jax(jnp.array(theta_np), h_jax, L_jax, N, D))


def compute_metrics(theta_np, h_jax, L_jax, N, D):
    """Return (MRE, MAE, RMSE) as floats."""
    mre, mae, rmse = _metrics_jax(jnp.array(theta_np), h_jax, L_jax, N, D)
    return float(mre), float(mae), float(rmse)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Initial Parameter Guess
# ─────────────────────────────────────────────────────────────────────────────

def initial_theta(N: float, seed: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = rng.uniform(0.10, 0.60, K)
    c = rng.uniform(1.00, 8.00, K)
    A = rng.uniform(1.00, 8.00, K)
    a = rng.uniform(0.10, 0.60, K)
    E = rng.uniform(0.10, 1.00, K)
    H = N / (10.0 * K)
    return pack(b, c, A, a, E, H)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Optimal Mixture Extraction (softmax-parametrised L-BFGS-B)
# ─────────────────────────────────────────────────────────────────────────────

@jit
def _loss_for_mixture(h, b, c, A, a, E, H, N, D):
    """Total predicted loss for a given mixture h (used for optimal h* search)."""
    L_hat = _predict_batch(h[jnp.newaxis, :], b, c, A, a, E, H, N, D)  # (1, K)
    return L_hat.sum()


_obj_h_val_grad = None   # lazily initialised


def _get_obj_h():
    """Lazily create the JIT-compiled objective over mixture h."""
    global _obj_h_val_grad
    if _obj_h_val_grad is None:
        _obj_h_val_grad = jit(value_and_grad(_loss_for_mixture))
    return _obj_h_val_grad


def compute_optimal_mixture(theta_np, N=None, D=None, n_restarts: int = 20,
                             N_opt: float = 1300.0, D_opt: float = 30.0):
    """
    Find h* = argmin_h  (1/K) Σ_i L_i(h)  s.t.  sum(h)=1, h_j >= H_MIN.

    By default N_opt=1300M, D_opt=30B (fixed reference scale) so that
    all rows in results.jsonl use a consistent mixture for fair comparison.

    Parametrisation: h = H_MIN + (1-K·H_MIN)·softmax(z), z∈R^K —
    unconstrained L-BFGS-B over z.
    Returns optimal DOMAIN_PROPORTIONS dict (RedPajama naming, values sum to 1).
    """
    b, c, A, a, E, H = unpack(theta_np)
    b_j = jnp.array(b); c_j = jnp.array(c)
    A_j = jnp.array(A); a_j = jnp.array(a)
    E_j = jnp.array(E)

    obj_fn = _get_obj_h()

    def obj_z(z_np):
        z = jnp.array(z_np)
        h_raw = jax.nn.softmax(z)
        h = H_MIN + (1.0 - K * H_MIN) * h_raw
        raw_val, raw_grad = obj_fn(h, b_j, c_j, A_j, a_j, E_j, float(H), N_opt, D_opt)
        val  = float(raw_val) / float(K)
        grad = np.array(raw_grad, dtype=np.float64) / float(K)
        return val, grad

    def logits_to_h(z_np):
        h_raw = jax.nn.softmax(jnp.array(z_np))
        return np.array(H_MIN + (1.0 - K * H_MIN) * h_raw)

    rng      = np.random.default_rng(RANDOM_SEED)
    best_val = np.inf
    best_h   = None

    for _ in range(n_restarts):
        z0 = rng.standard_normal(K)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                obj_z, z0, method="L-BFGS-B", jac=True,
                options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10},
            )
        if res.fun < best_val:
            best_val = res.fun
            best_h   = logits_to_h(res.x)

    proportions = {}
    for i, rp_name in enumerate(REDPAJAMA_NAMES):
        proportions[rp_name] = round(float(best_h[i]), 4)

    return proportions, best_h


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Fit for a single (N, D) pair — used by run_experiments.py
# ─────────────────────────────────────────────────────────────────────────────

def fit_for_pair(model_size, n_tokens, csv_path=CSV_PATH,
                 n_trials=1, verbose=False):
    """
    Fit the scaling law for a specific (model_size, n_tokens) pair.

    Parameters
    ----------
    model_size : float
    n_tokens   : float
    csv_path   : str
    n_trials   : int   (default 1; up to 20 for hyperparameter search)
    verbose    : bool

    Returns
    -------
    dict with keys:
        theta_best, mre_val, mae_val, rmse_val,
        proportions, fit_time, n_trials_run
    """
    t0 = time.time()

    h_tr, L_tr, h_va, L_va, N, D = load_data_for_pair(
        csv_path, model_size, n_tokens)

    if verbose:
        print(f"[INFO] N={N:.3f}M  D={D:.3f}B  train={len(h_tr)}  val={len(h_va)}")

    # JIT warm-up
    theta0 = initial_theta(N)
    _ = objective_scipy(theta0, h_tr, L_tr, N, D)

    best_theta = None
    best_mre   = np.inf

    for trial in range(n_trials):
        seed = RANDOM_SEED + trial
        theta0 = initial_theta(N, seed=seed)

        # Vary BH parameters across trials for diversity
        bh_t       = BH_T       * (0.5 + 0.5 * (trial % 5) / 4)   # [0.5, 1.0]×BH_T
        bh_stepsize = BH_STEPSIZE * (0.5 + 0.5 * (trial % 7) / 6)  # [0.5, 1.0]×BH_STEPSIZE

        minimizer_kwargs = {
            "method":   "L-BFGS-B",
            "jac":      True,
            "args":     (h_tr, L_tr, N, D),
            "bounds":   get_bounds(),
            "options":  {
                "maxiter": LBFGSB_MAXITER,
                "ftol":    1e-14,
                "gtol":    1e-9,
            },
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = basinhopping(
                objective_scipy,
                theta0,
                minimizer_kwargs=minimizer_kwargs,
                niter=BH_NITER,
                stepsize=bh_stepsize,
                T=bh_t,
                niter_success=BH_NITER_SUCCESS,
                seed=seed,
            )

        mre_trial = mean_relative_error(result.x, h_va, L_va, N, D)
        if verbose:
            print(f"  Trial {trial+1}/{n_trials}: val MRE={mre_trial:.4%}")

        if mre_trial < best_mre:
            best_mre   = mre_trial
            best_theta = result.x

    fit_time = time.time() - t0
    mre, mae, rmse = compute_metrics(best_theta, h_va, L_va, N, D)
    # Optimal mixture is always computed at the fixed reference scale (1.3B, 30B)
    proportions, _ = compute_optimal_mixture(best_theta, N_opt=1300.0, D_opt=30.0)

    return {
        "theta_best":    best_theta,
        "mre_val":       mre,
        "mae_val":       mae,
        "rmse_val":      rmse,
        "proportions":   proportions,
        "fit_time":      fit_time,
        "n_trials_run":  n_trials,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Data Mixing Scaling Law — Basin-Hopping + L-BFGS-B (JAX/GPU)")
    print("=" * 65)

    # ── Load & split ─────────────────────────────────────────────────────────
    h_data, L_data, N, D = load_data(CSV_PATH)
    print(f"[INFO] N = {N:.3f} M params   D = {D:.3f} B tokens")
    print(f"[INFO] Dataset size = {len(h_data)} rows")

    h_tr_np, L_tr_np, h_va_np, L_va_np = train_val_split(h_data, L_data)
    print(f"[INFO] Train / Val  = {len(h_tr_np)} / {len(h_va_np)}")

    # Move data to JAX (GPU) once — stays there for the whole run
    h_tr  = jnp.array(h_tr_np);  L_tr  = jnp.array(L_tr_np)
    h_va  = jnp.array(h_va_np);  L_va  = jnp.array(L_va_np)

    # ── Warm-up JIT (first call compiles the kernel) ──────────────────────────
    print("[INFO] Compiling JAX kernels …", end=" ", flush=True)
    theta0 = initial_theta(N)
    _ = objective_scipy(theta0, h_tr, L_tr, N, D)
    print("done")

    mse0 = objective_scipy(theta0, h_tr, L_tr, N, D)[0]
    print(f"[INFO] Initial MSE = {mse0:.6e}\n")

    # ── Progress callbacks ────────────────────────────────────────────────────
    state = {"bh_step": 0, "lbfgs_iter": 0}

    def lbfgsb_callback(x):
        state["lbfgs_iter"] += 1
        mre_tr = mean_relative_error(x, h_tr, L_tr, N, D)
        mre_va = mean_relative_error(x, h_va, L_va, N, D)
        print(f"  [BH {state['bh_step']:3d} | LBFGS {state['lbfgs_iter']:4d}]"
              f"  train MRE = {mre_tr:.4%}   val MRE = {mre_va:.4%}")

    def bh_callback(x, f, accepted):
        state["bh_step"]   += 1
        state["lbfgs_iter"] = 0
        print(f"  ── BH step {state['bh_step']:3d} done"
              f"  MSE = {f:.6e}  {'[accepted]' if accepted else '[rejected]'}")

    # ── Basin-Hopping ─────────────────────────────────────────────────────────
    minimizer_kwargs = {
        "method":   "L-BFGS-B",
        "jac":      True,          # objective_scipy returns (value, grad)
        "args":     (h_tr, L_tr, N, D),
        "bounds":   get_bounds(),
        "callback": lbfgsb_callback,
        "options":  {
            "maxiter": LBFGSB_MAXITER,
            "ftol":    1e-14,

            "gtol":    1e-9,
        },
    }

    print(f"[INFO] Running Basin-Hopping  (niter={BH_NITER}, "
          f"stepsize={BH_STEPSIZE}, T={BH_T}) …\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = basinhopping(
            objective_scipy,
            theta0,
            minimizer_kwargs=minimizer_kwargs,
            niter=BH_NITER,
            stepsize=BH_STEPSIZE,
            T=BH_T,
            niter_success=BH_NITER_SUCCESS,
            seed=RANDOM_SEED,
            callback=bh_callback,
        )

    theta_best = result.x
    print(f"\n[INFO] Optimisation finished after {state['bh_step']} BH steps.")
    print(f"[INFO] Final train MSE = {result.fun:.6e}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    mre_tr, mae_tr, rmse_tr = compute_metrics(theta_best, h_tr, L_tr, N, D)
    mre_va, mae_va, rmse_va = compute_metrics(theta_best, h_va, L_va, N, D)

    print("\n" + "=" * 65)
    print("  Evaluation")
    print("=" * 65)
    print(f"  {'Metric':<10}  {'Train':>10}  {'Val':>10}")
    print(f"  {'MRE':10}  {mre_tr:>10.4%}  {mre_va:>10.4%}")
    print(f"  {'MAE':10}  {mae_tr:>10.4e}  {mae_va:>10.4e}")
    print(f"  {'RMSE':10}  {rmse_tr:>10.4e}  {rmse_va:>10.4e}")

    # ── Fitted parameters ─────────────────────────────────────────────────────
    b, c, A, a, E, H = unpack(theta_best)
    print("\n  Fitted per-domain parameters:")
    header = f"  {'Domain':<16} {'b':>8} {'c':>8} {'A':>8} {'a':>8} {'E':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(DOMAIN_NAMES):
        print(f"  {name:<16} {b[i]:>8.4f} {c[i]:>8.4f} "
              f"{A[i]:>8.4f} {a[i]:>8.4f} {E[i]:>8.4f}")
    print(f"\n  H (global capacity slack) = {H:.6f} M params")

    # ── Optimal mixture h* at the current (N, D) ──────────────────────────────
    print(f"\n[INFO] Searching for h* at N={N:.3f}M, D={D:.3f}B …")
    b_j = jnp.array(b); c_j = jnp.array(c)
    A_j = jnp.array(A); a_j = jnp.array(a)
    E_j = jnp.array(E)

    proportions_star, h_star = compute_optimal_mixture(theta_best, N_opt=N, D_opt=D)
    L_star = _loss_for_mixture(jnp.array(h_star), b_j, c_j, A_j, a_j, E_j, float(H), N, D)
    avg_loss = float(L_star) / K
    print(f"  h* = {np.round(h_star, 4)}")
    print(f"  (1/K) Σ L_i(h*) = {avg_loss:.6f}")

    print("\n  Optimal DOMAIN_PROPORTIONS (h*):")
    max_len = max(len(n) for n in REDPAJAMA_NAMES)
    print("  DOMAIN_PROPORTIONS = {")
    for i, rp_name in enumerate(REDPAJAMA_NAMES):
        comma = "," if i < K - 1 else ""
        print(f'    "{rp_name}":{" " * (max_len - len(rp_name) + 3)}{proportions_star[rp_name]:.4f}{comma}')
    print("  }")

    return theta_best


# Guard: only run when executed as a script, not when imported as a module.
if __name__ == "__main__":
    main()
