import numpy as np
from scipy.optimize import minimize, root_scalar, least_squares, basinhopping
import time

def law2(h_data, L_data, N_data, M_data, MIN_X=0.001, seed=42, N_BASINS=5, BH_TEMPERATURE=5e-5, STEP_SIZE=0.5):
    """
    Fitting with per-sample N[i] and M[i].
    Uses Basin Hopping + L-BFGS-B for global optimization.
    Constraint: No constraint between alpha and b. They are independent.

    Inputs:
    - h_data: (n_samples, K) feature/intensity per dimension
    - L_data: (n_samples, K) observed targets
    - N_data: (n_samples,) per-sample scaling factor N[i]
    - M_data: (n_samples,) per-sample total resource M[i]
    - MIN_X: lower bound for each allocation dimension
    - seed: random seed for reproducibility
    """
    n_samples, K = h_data.shape
    print(f"Detected K={K} from data.")

    # Train/Val split 7:3
    rng_split = np.random.default_rng(20)
    perm = rng_split.permutation(n_samples)
    split_idx = int( n_samples*0.7)
    train_idx = perm[:split_idx]
    val_idx = perm[split_idx:]
    #val_idx = perm
    h_train = h_data[train_idx]
    L_train = L_data[train_idx]
    N_train = N_data[train_idx]
    M_train = M_data[train_idx]

    h_val = h_data[val_idx]
    L_val = L_data[val_idx]
    N_val = N_data[val_idx]
    M_val = M_data[val_idx]

    n_train = len(train_idx)
    n_val = len(val_idx)
    print(f"Train samples: {n_train}, Val samples: {n_val}")

    # ==========================================
    # FAST ANALYTIC INNER SOLVER (uses per-sample M)
    # ==========================================
    def solve_x_star_batch(h_batch, c, b, M_batch):
        """
        Vectorized batch solver for x*.
        Solves for all samples simultaneously using parallel binary search.
        Note: Inner solver only depends on resource parameters (c, b), not alpha.
        
        Inputs:
            h_batch: (N, K) - batch of h vectors
            c: (K,) - coefficients
            b: (K,) - exponents (resource)
            M_batch: (N,) - per-sample resource limits
        Returns:
            x_opt: (N, K) - optimal allocations
        """
        N_samples, K = h_batch.shape
        
        # Broadcast c, b to (1, K)
        c_safe = np.maximum(c, 1e-9)[None, :]
        b_safe = np.maximum(b, 1e-9)[None, :]
        M_batch = M_batch.reshape(-1, 1)  # (N, 1)
        
        # Precompute: numerators (N, K), exponents (1, K)
        numerators = b_safe * h_batch * c_safe
        exponents = 1.0 / (b_safe + 1.0)
        
        # Binary search bounds (N, 1)
        lb = np.full((N_samples, 1), 1e-30, dtype=np.float64)
        ub = np.full((N_samples, 1), 1e25, dtype=np.float64)
        
        # 60 iterations for high precision
        for _ in range(100):
            mid = 0.5 * (lb + ub)
            
            # Compute x for current lambda (mid)
            # mid: (N, 1), numerators: (N, K) -> broadcast
            x_raw = np.power(numerators / mid, exponents)
            x_curr = np.maximum(x_raw, MIN_X)
            
            # Sum along K dimension: (N, 1)
            total_usage = np.sum(x_curr, axis=1, keepdims=True)
            
            # Update bounds: if usage > M, need larger lambda (lb = mid)
            mask_too_high = (total_usage > M_batch)
            lb = np.where(mask_too_high, mid, lb)
            ub = np.where(~mask_too_high, mid, ub)
        
        # Final calculation with converged lambda
        lam_final = ub
        x_opt = np.power(numerators / lam_final, exponents)
        x_opt = np.maximum(x_opt, MIN_X)
        
        # Optional: Explicit renormalization
        current_sum = np.sum(x_opt, axis=1, keepdims=True)
        scale = M_batch / (current_sum + 1e-12)
        x_opt = x_opt * scale
        
        return x_opt
    
    def solve_x_star(h_vec, c, b, M_total):
        """
        Single-sample wrapper for compatibility.
        Calls vectorized batch solver.
        """
        h_batch = h_vec.reshape(1, -1)
        M_batch = np.array([M_total])
        x_batch = solve_x_star_batch(h_batch, c, b, M_batch)
        return x_batch[0]

    # Optimized Alpha (constraint released)
    # FIXED_ALPHA = np.array([0.146349, 0.493726, 0.099988, 0.668617])
    # ... (Truncating/Padding alpha logic removed or commented out) ...

    def objective_function(params):
        """
        Global objective: Minimize MSE on training data.
        params: [c, b, alpha, E, A] (Size 5*K)
        """
        c_curr = params[:K]
        b_curr = params[K:2*K]
        alpha_curr = params[2*K:3*K] # Now optimized
        E_curr = params[3*K:4*K]
        A_curr = params[4*K:]
        
        # Vectorized batch prediction (Resource allocation logic uses b)
        x_opt_batch = solve_x_star_batch(h_train, c_curr, b_curr, M_train)
        
        # Resource term: (N, K) (uses b)
        term_resource = c_curr * np.power(x_opt_batch, -b_curr)
        
        # Scaling term: (N, K) (uses alpha)
        N_mat = N_train.reshape(-1, 1)
        scaling_base = N_mat * h_train + 1e-12
        term_scaling = A_curr * np.power(scaling_base, -alpha_curr)
        
        # Prediction: (N, K)
        L_pred = term_resource + E_curr + term_scaling

        # Mask and compute MSE
        mask = h_train > 0
        diff = L_pred[mask] - L_train[mask]
        
        loss_sum = np.sum(diff**2)
                
        return loss_sum / n_train

    def compute_metrics(params, h_set, L_set, N_set, M_set):
        c_curr = params[:K]
        b_curr = params[K:2*K]
        alpha_curr = params[2*K:3*K] # Now optimized
        E_curr = params[3*K:4*K]
        A_curr = params[4*K:]

        # Vectorized batch prediction
        x_opt_batch = solve_x_star_batch(h_set, c_curr, b_curr, M_set)
        
        # Resource term
        term_resource = c_curr * np.power(x_opt_batch, -b_curr)
        
        # Scaling term
        N_mat = N_set.reshape(-1, 1)
        scaling_base = N_mat * h_set + 1e-12
        term_scaling = A_curr * np.power(scaling_base, -alpha_curr)
        
        # Prediction
        L_pred = term_resource + E_curr + term_scaling
        
        # Metrics
        mask = h_set > 0
        obs = L_set[mask]
        pred = L_pred[mask]
        
        rel_err = np.abs((pred - obs) / (obs + 1e-9))
        mre_sum = np.mean(rel_err)
        mae_sum = np.mean(np.abs(pred - obs))
        rmse_sum = np.sqrt(np.mean((pred - obs)**2))
        
        overall_mre = mre_sum
        overall_mae = mae_sum
        overall_rmse = rmse_sum
        
        # Per-dimension metrics
        mre_per_dim = []
        counts_per_dim = []
        
        for k in range(K):
            mask_k = h_set[:, k] > 0
            count_k = np.sum(mask_k)
            counts_per_dim.append(count_k)
            
            if count_k > 0:
                obs_k = L_set[mask_k, k]
                pred_k = L_pred[mask_k, k]
                rel_err_k = np.abs((pred_k - obs_k) / (obs_k + 1e-9))
                # User requested: multiply by count_k (weighted contribution)
                mre_per_dim.append(np.mean(rel_err_k) * count_k)
            else:
                mre_per_dim.append(0.0)
        
        return overall_mre, overall_mae, overall_rmse, L_pred, np.array(mre_per_dim), np.array(counts_per_dim)

    # --- Setup Optimization ---
    
    np.random.seed(seed)
    
    # Needs 5 groups of parameters now: c, b, alpha, E, A
    
    # Bounds:
    bounds_c = [(0.0, 20.0)] * K
    bounds_b = [(0.01, 0.8)] * K
    bounds_alpha = [(0.01, 0.8)] * K # User specified range
    bounds_E = [(0.0, 5.0)] * K
    bounds_A = [(0.0, 500.0)] * K
    
    bounds = bounds_c + bounds_b + bounds_alpha + bounds_E + bounds_A
    
    # Initial Guess
    c_init = np.array([1.254, 1.372, 2.46,  1.206, 1.303, 0.435, 1.401, 0.984, 1.34,  0.337, 1.614, 1.522,0.789, 2.075, 0.389, 1.237, 1.606])
    b_init = np.array([0.111, 0.134, 0.082, 0.411, 0.09,  0.5,   0.121, 0.179, 0.073, 0.028, 0.088, 0.446, 0.412, 0.271, 0.5,   0.057, 0.062])
    #alpha_init = np.array([.000361, 0.001065, 1.251461, 0.000868])  # Use fixed initial alpha
    E_init = np.array([0.888, 0.979, 0.01,  1.473, 1.36,  1.056, 0.038, 1.95,  0.673, 2.475, 1.227, 2.691,1.706, 0.761, 2.548, 1.26,  0.872])
    #A_init = np.array([1.204, 3.252, 1.495, 3.486])
    alpha_init=np.random.uniform(0.01,0.8,K)
    A_init = np.zeros(K)
    
    '''
    c_init=np.array([2.328 ,0.326, 2.374 ,0.736])
    b_init=np.array([0.258, 0.452 , 0.217, 0.379])
    # alpha_init determined by FIXED_ALPHA
    E_init=np.array([0.   , 2.645, 0.356, 2.339])
    A_init=np.array([1.143 ,4.017, 1.523, 3.716])
    ''' 
    x0 = np.concatenate([c_init, b_init, alpha_init, E_init, A_init])
    
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds,
        "tol": 1e-15,
        "options": {"gtol": 1e-15}
    }

    print(f"Starting Basin-Hopping Optimization (K={K})...")
    print(f"Independent: alpha, b")
    print("-" * 60)
    
    start_time = time.time()
    t0 = start_time
    
    basin_counter = 0
    best_fun = np.inf

    def bh_callback(x, f, accepted):
        nonlocal basin_counter, best_fun
        basin_counter += 1
        elapsed = time.time() - t0
        status = "ACCEPTED" if accepted else "REJECTED"
        
        if f < best_fun:
            best_fun = f
            
        val_mre, _, _, _, val_mre_per_dim, val_counts = compute_metrics(x, h_val, L_val, N_val, M_val)
        
        print(f"Basin {basin_counter}/{N_BASINS}: obj={f:.6e} [{status}] | Val MRE={val_mre*100:.4f}% | Time={elapsed:.1f}s", flush=True)
        print(f"  > Val per-dim Weighted Error Sum: {np.array2string(val_mre_per_dim, precision=4, suppress_small=True)}")
        
        # Print params
        c_curr = x[:K]
        b_curr = x[K:2*K]
        alpha_curr = x[2*K:3*K]
        E_curr = x[3*K:4*K]
        A_curr = x[4*K:]
        
        print(f"  > b    : {np.array2string(b_curr, precision=3, suppress_small=True)}")
        print(f"  > alpha: {np.array2string(alpha_curr, precision=3, suppress_small=True)}")
        print(f"  > c    : {np.array2string(c_curr, precision=3, suppress_small=True)}")
        print(f"  > E    : {np.array2string(E_curr, precision=3, suppress_small=True)}")
        print(f"  > A    : {np.array2string(A_curr, precision=3, suppress_small=True)}")
        
        return False

    res = basinhopping(
        objective_function,
        x0,
        niter=N_BASINS,
        T=BH_TEMPERATURE,
        stepsize=STEP_SIZE,
        minimizer_kwargs=minimizer_kwargs,
        callback=bh_callback,
        disp=False
    )
    
    total_time = time.time() - t0
    best_params = res.x
    
    c_final = best_params[:K]
    b_final = best_params[K:2*K]
    alpha_final = best_params[2*K:3*K]
    E_final = best_params[3*K:4*K]
    A_final = best_params[4*K:] # Fixed indexing for A_final
    
    # Final Validation
    train_mre, _, _, _, _, _ = compute_metrics(best_params, h_train, L_train, N_train, M_train)
    val_mre, val_mae, val_rmse, val_preds, val_mre_per_dim, val_counts = compute_metrics(best_params, h_val, L_val, N_val, M_val)
    
    print("-" * 60)
    print(f"Optimization finished in {total_time:.2f}s")
    print(f"Best Objective: {res.fun:.6e}")
    print("\nFinal Results:")
    print(f"Train MRE: {train_mre*100:.4f}%")
    print(f"Val MRE (Weighted):   {val_mre*100:.4f}%")
    print(f"Val MAE (Weighted):   {val_mae:.6f}")
    print(f"Val RMSE (Weighted):  {val_rmse:.6f}")
    
    # Calculate weighted MRE explicitly for display
    total_valid = np.sum(val_counts)
    if total_valid > 0:
        # mre_per_dim now already contains * count_k factor
        weighted_check = np.sum(val_mre_per_dim) / total_valid
        print(f"Check Weighted Avg: {weighted_check*100:.4f}% (matches Val MRE)")
    
    print(f"Val per-dim Weighted Error Sum (MRE*N): {val_mre_per_dim/total_valid*17}")
    print(f"Val dim counts:  {val_counts}")
    
    print("\nParameters:")
    print(f"  b     = {b_final}")
    print(f"  alpha = {alpha_final}")
    print(f"  c     = {c_final}")
    print(f"  E     = {E_final}")
    print(f"  A     = {A_final}")
    
    return {
        'c': c_final,
        'b': b_final,
        'alpha': alpha_final, 
        'E': E_final,
        'A': A_final,
        'val_mre': val_mre,
        'val_preds': val_preds
    }