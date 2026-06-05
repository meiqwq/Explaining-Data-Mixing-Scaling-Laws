"""
optimal_mixture.py
==================
Given fitted scaling-law parameters (b, c, A, a, E, H, N, D),
find the mixture weight vector h* that minimises the mean loss
averaged over all K domains:

    min_{h}  (1/K) Σ_i  L_i(h, N, D)
    s.t.     Σ_i h_i = 1
             h_i ≥ 0  ∀ i

where L_i is the forward model from fit_scaling_law.py:

    L_i(h) = c_i · (x_i*(h))^{-b_i}  +  A_i · (D·h_i)^{-a_i}  +  E_i

and x*(h) is the inner KKT solution (capacity allocation).

Because h_i = 0 makes the noise term A_i·(D·h_i)^{-a_i} blow up,
we enforce h_i ≥ h_min (default 1e-4) instead of a hard zero bound.

Optimisation strategy
---------------------
The outer problem is smooth and low-dimensional (K=7), so we use
L-BFGS-B with exact JAX gradients, wrapped in Basin-Hopping for
global search.  The simplex constraint Σ h_i = 1 is handled by
reparameterising h via a softmax over K unconstrained logits z:

    h_i = softmax(z)_i  =  exp(z_i) / Σ_j exp(z_j)

This makes the search space unconstrained (z ∈ R^K) and the
gradient flows cleanly through the softmax.
"""

import warnings
import numpy as np
from scipy.optimize import minimize, basinhopping

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

# ── reuse the inner solver from fit_scaling_law ───────────────────────────────
from fitting_7domains import _solve_inner_single, K, DOMAIN_NAMES

jax.config.update("jax_enable_x64", True)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
H_MIN        = 1e-4   # minimum weight per domain (avoids noise-term singularity)
BH_NITER     = 200
BH_STEPSIZE  = 0.5
BH_T         = 1.0
BH_NITER_SUCCESS = 60
LBFGSB_MAXITER   = 1000
RANDOM_SEED  = 42


# ─────────────────────────────────────────────────────────────────────────────
# Forward model for a single h vector (JAX, differentiable)
# ─────────────────────────────────────────────────────────────────────────────

def _predict_single(h, b, c, A, a, E, H, N, D):
    """
    Returns the K-dim loss vector for mixture h.
    All domains are assumed active (h_i ≥ H_MIN enforced by the softmax
    parameterisation + H_MIN floor).
    """
    x_star        = _solve_inner_single(h, b, c, H, N)       # (K,)
    capacity_term = c * (x_star ** (-b))                      # (K,)
    noise_term    = A * (D * h) ** (-a)                       # (K,)  safe: h≥H_MIN
    return capacity_term + noise_term + E                     # (K,)


# ─────────────────────────────────────────────────────────────────────────────
# Objective: mean loss over domains, parameterised via softmax logits
# ─────────────────────────────────────────────────────────────────────────────

def _mean_loss_from_logits(z, b, c, A, a, E, H, N, D):
    """
    z : (K,) unconstrained logits
    Returns scalar mean loss (1/K) Σ L_i(h(z)).
    """
    # Softmax with H_MIN floor and renormalisation
    h_raw = jax.nn.softmax(z)                                 # sums to 1, all > 0
    h     = H_MIN + (1.0 - K * H_MIN) * h_raw                # floor at H_MIN, still sums to 1

    L = _predict_single(h, b, c, A, a, E, H, N, D)
    return jnp.mean(L)


_obj_val_and_grad = jit(value_and_grad(_mean_loss_from_logits))


def _objective_scipy(z_np, b, c, A, a, E, H, N, D):
    """Scipy-compatible wrapper: numpy in, (float, numpy grad) out."""
    z   = jnp.array(z_np)
    val, grad = _obj_val_and_grad(z, b, c, A, a, E, H, N, D)
    return float(val), np.array(grad, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_mixture(b, c, A, a, E, H, N, D,
                         bh_niter=BH_NITER,
                         seed=RANDOM_SEED,
                         verbose=True):
    """
    Find h* = argmin_{h∈Δ_K, h_i≥H_MIN} (1/K) Σ_i L_i(h, N, D).

    Parameters
    ----------
    b, c, A, a, E : array-like, shape (K,)  — fitted domain parameters
    H             : float                   — capacity slack
    N             : float                   — model size in millions
    D             : float                   — training tokens in billions
    bh_niter      : int                     — Basin-Hopping iterations
    seed          : int
    verbose       : bool

    Returns
    -------
    h_star   : (K,) optimal mixture weights (sum to 1, each ≥ H_MIN)
    loss_star: float, mean domain loss at h*
    """
    b = jnp.array(b, dtype=jnp.float64)
    c = jnp.array(c, dtype=jnp.float64)
    A = jnp.array(A, dtype=jnp.float64)
    a = jnp.array(a, dtype=jnp.float64)
    E = jnp.array(E, dtype=jnp.float64)
    H = jnp.float64(H)

    args = (b, c, A, a, E, H, N, D)

    # Warm-up JIT
    _ = _objective_scipy(np.zeros(K), *args)

    rng   = np.random.default_rng(seed)
    z0    = rng.standard_normal(K)          # random starting logits

    state = {"bh_step": 0, "lbfgs_iter": 0, "best": np.inf}

    def lbfgsb_callback(z):
        state["lbfgs_iter"] += 1
        val = _objective_scipy(z, *args)[0]
        if verbose:
            h_cur = _logits_to_h(z)
            print(f"  [BH {state['bh_step']:3d} | LBFGS {state['lbfgs_iter']:4d}]"
                  f"  mean loss = {val:.6f}"
                  f"  h = [{', '.join(f'{v:.3f}' for v in h_cur)}]")

    def bh_callback(z, f, accepted):
        state["bh_step"]   += 1
        state["lbfgs_iter"] = 0
        if f < state["best"]:
            state["best"] = f
        if verbose:
            print(f"  ── BH step {state['bh_step']:3d} done"
                  f"  best mean loss = {state['best']:.6f}"
                  f"  {'[accepted]' if accepted else '[rejected]'}")

    minimizer_kwargs = {
        "method":   "L-BFGS-B",
        "jac":      True,
        "args":     args,
        "options":  {"maxiter": LBFGSB_MAXITER, "ftol": 1e-15, "gtol": 1e-10},
        "callback": lbfgsb_callback,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = basinhopping(
            _objective_scipy,
            z0,
            minimizer_kwargs=minimizer_kwargs,
            niter=bh_niter,
            stepsize=BH_STEPSIZE,
            T=BH_T,
            niter_success=BH_NITER_SUCCESS,
            seed=seed,
            callback=bh_callback,
        )

    h_star    = _logits_to_h(result.x)
    loss_star = float(result.fun)
    return h_star, loss_star


def _logits_to_h(z_np):
    """Convert logits → mixture weights (numpy output for display)."""
    z     = jnp.array(z_np)
    h_raw = jax.nn.softmax(z)
    h     = H_MIN + (1.0 - K * H_MIN) * h_raw
    return np.array(h)


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo — plug in your own fitted parameters
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Optimal Mixture Search")
    print("=" * 65)

    #        arxiv   book    c4      github  cc      se      wiki
    b_fit = np.array([1.2931, 0.2504, 0.7759, 4.2533, 0.8209, 2.1420, 0.9245])
    c_fit = np.array([6.1217, 3.3571, 7.8612, 7.3098, 2.3083, 3.8489, 2.3059])
    A_fit = np.array([0.3931, 0.4080, 0.7319, 1.2868, 0.7130, 2.0230, 0.8802])
    a_fit = np.array([0.9685, 1.6999, 0.3088, 0.0827, 1.0596, 0.0777, 0.5132])
    E_fit = np.array([0.9448, 0.8770, 0.5612, 0.0000, 2.4074, 0.0001, 1.3406])
    H_fit = 4.877181
    N     = 1300   # model size in millions
    D     = 30    # training tokens in billions
    # ─────────────────────────────────────────────────────────────────────────

    print(f"[INFO] N = {N} M params,  D = {D} B tokens,  H = {H_fit}\n")

    h_star, loss_star = find_optimal_mixture(
        b_fit, c_fit, A_fit, a_fit, E_fit, H_fit, N, D,
        bh_niter=BH_NITER, verbose=True,
    )

    print("\n" + "=" * 65)
    print("  Result")
    print("=" * 65)
    print(f"  Optimal mean loss : {loss_star:.6f}")
    print(f"  {'Domain':<16} {'h*':>8}")
    print("  " + "-" * 26)
    for name, hi in zip(DOMAIN_NAMES, h_star):
        print(f"  {name:<16} {hi:>8.4f}")
    print(f"\n  sum(h*) = {h_star.sum():.6f}  (should be 1.0)")
