import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping, minimize, root_scalar
import cvxpy as cp
import time
import json
def law1(h_data,L_data,K=200,N_BASINS=5,BH_TEMPERATURE=5e-5,STEP_SIZE=0.5,MIN_X=0.3,use_13train=False,seed=42):
    N=h_data.shape[1]
    valid_pos = np.arange(N)
    if use_13train:
        valid_pos = [0,1,3,4,5,6,8,10,11,12,14,15,16]
    h_=-1
    h_data=h_data.T
    L_observed_noisy=L_data.T

    # --- Train/Validation Split (70/30) ---
    num_samples = h_data.shape[1]
    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(num_samples)
    split_idx = int(0.7 * num_samples)
    train_idx = perm[:split_idx]
    val_idx = perm[split_idx:]

    h_train = h_data[:, train_idx]
    L_train = L_observed_noisy[:, train_idx]
    h_val = h_data[:, val_idx]
    L_val = L_observed_noisy[:, val_idx]

    # Progress counters
    eval_counter = 0
    start_time = 0.0
    
    # Timing counters for root_scalar
    total_solve_time = 0.0
    total_solve_count = 0

    # Basin-hopping settings


    # --- 3. The Implicit Layer (Vector Predictor) ---
    def predict_vector_fast(h_vec, params):
        # Unpack params:
        # First N are b
        # Next N are c
        # Last N are E (Vector)
        nonlocal total_solve_time, total_solve_count
        b_guess = params[:N]
        c_guess = params[N:2*N]
        E_guess = params[2*N:] # Sliced for vector E

        # Pre-calc terms for KKT (System minimizes h*c*x^-b)
        w = h_vec * c_guess
        numerator = w * b_guess
        exponents = 1.0 / (b_guess + 1.0)

        # Solve for Lambda
        def constraint_func(lam):
            #lam = max(lam, 1e-12)
            x_vals = np.maximum(np.power(numerator / lam, exponents), MIN_X)#
            return np.sum(x_vals) - K

        try:
            t_solve_start = time.time()
            sol = root_scalar(constraint_func, bracket=[1e-20, 1e12], method='brentq')

            lam_opt = sol.root
            total_solve_time += (time.time() - t_solve_start)
            total_solve_count += 1
            
        except ValueError:
            return None

        # Compute optimal x
        x_opt = np.maximum(np.power(numerator / lam_opt, exponents), MIN_X)#
        #print(x_opt)
        # Predict observed L_i = c_i * x_i^(-b_i) + E_i
        L_pred = c_guess * np.power(x_opt, -b_guess) + E_guess
        return L_pred[valid_pos]

    # --- 4. Inverse Optimization ---
    def inverse_objective_func(params, h_matrix, L_obs_matrix):
        nonlocal eval_counter, start_time
        eval_counter += 1
        # Constraints
        # b and c must be positive (> 0.001)
        if np.any(params[:2*N] <= 0.001): return 1e12
        # E must be non-negative (>= 0)
        #if np.any(params[2*N:] < 0): return 1e12

        # Vectorized Error Calculation
        total_sq_error = 0.0
        for i in range(h_matrix.shape[1]):
            pred_vec = predict_vector_fast(h_matrix[:, i], params)
            if pred_vec is None or np.any(~np.isfinite(pred_vec)):
                return 1e12
            obs_vec = L_obs_matrix[:, i]
            #total_sq_error += np.sum((pred_vec - obs_vec)**2)
            for _ in range(len(valid_pos)):
                if h_matrix[_,i]>=h_:
                    total_sq_error += (pred_vec[_]-obs_vec[_])**2

        obj = total_sq_error / (h_matrix.shape[1] * len(valid_pos))


        return obj

    def compute_metrics(params, h_matrix, L_obs_matrix):
        """Compute metrics for given params; returns (mre_overall, mae, rmse, mre_per_dim or None on failure)."""
        L_pred_matrix = []
        for i in range(h_matrix.shape[1]):
            pred_vec = predict_vector_fast(h_matrix[:, i], params)
            if pred_vec is None or np.any(~np.isfinite(pred_vec)):
                return None
            L_pred_matrix.append(pred_vec)
                
        L_pred_matrix = np.array(L_pred_matrix).T  # (len(valid_pos), num_samples)
        eps = 1e-9
        l=L_obs_matrix[valid_pos,:]
        diff = L_pred_matrix - l
        abs_diff = np.abs(diff)
        
        # Per-dimension metrics
        mre_per_dim = np.mean(abs_diff / np.maximum(np.abs(l), eps), axis=1)
        mae_per_dim = np.mean(abs_diff, axis=1)
        mse_per_dim = np.mean(diff ** 2, axis=1)

        # Overall metrics
        mre_overall = np.mean(abs_diff / np.maximum(np.abs(l), eps))
        mae = np.mean(abs_diff)
        rmse = np.sqrt(np.mean(diff ** 2))
        return mre_overall, mae, rmse, mre_per_dim, mae_per_dim, mse_per_dim

    print("Running Inverse Optimization (Recovering 3N parameters)...")
    start_time = time.time()
    t0 = start_time

    # Initial Guess: b=0.5, c=1000.0, E=0.0
    # We now have 3*N parameters total

    # Bounds:
    # b in [0.1, 1.0]
    # c in [0.1, 10.0]
    # E in [0.0, 10.0] (Vectorized)
    bounds = [(0.01,0.5)]*N + [(0.1, 20)]*N + [(0.01, 3.0)]*N
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    def project_to_bounds(p):
        return np.minimum(np.maximum(p, lower_bounds), upper_bounds)


    # Basin-hopping with L-BFGS-B local minimizer
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "args": (h_train, L_train),
        "bounds": bounds,
        "tol": 1e-15,
        "options": { "gtol": 1e-15},
    }

    # Start from mid-bound guess
    init_guess = 0.5 * (lower_bounds + upper_bounds)
    # 把init_guess 改成在lower_bound和upper_bound之间随机取
    rng_init = np.random.default_rng(seed)
    init_guess = rng_init.uniform(lower_bounds, upper_bounds)
    best_res = None
    best_fun = np.inf
    basin_counter = 0

    def bh_callback(x, f, accepted):
        """Track best solution and log metrics per basin hop."""
        nonlocal best_res, best_fun, basin_counter
        basin_counter += 1

        # ALWAYS Print status regardless of acceptance
        status = "ACCEPTED" if accepted else "REJECTED"
        
        if f < best_fun:
            best_fun = f
            best_res = minimize(
                inverse_objective_func,
                x,
                args=(h_train, L_train),
                method="L-BFGS-B",
                bounds=bounds,
                tol=1e-15,
                options={"gtol": 1e-15},
            )

        # Compute validation metrics
        val_metrics = compute_metrics(x, h_val, L_val)
        # Compute train metrics for logging
        train_metrics = compute_metrics(x, h_train, L_train)
        
        if val_metrics is not None and train_metrics is not None:
            val_mre_o, val_mae_o, val_rmse_o, val_mre_per_dim, val_mae_per_dim, val_mse_per_dim = val_metrics
            train_mre_o, train_mae_o, train_rmse_o, _, _, _ = train_metrics
            print(
                f"Basin {basin_counter}/{N_BASINS}: obj={f:.6e} [{status}]\n"
                f"  Train: MRE={train_mre_o:.4f} ({train_mre_o*100:.2f}%), MAE={train_mae_o:.6f}, RMSE={train_rmse_o:.6f}\n"
                f"  Val:   MRE={val_mre_o:.4f} ({val_mre_o*100:.2f}%), MAE={val_mae_o:.6f}, RMSE={val_rmse_o:.6f}\n"
                f"  elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
            
            # Print Parameters
            b_curr = x[:N]
            c_curr = x[N:2*N]
            E_curr = x[2*N:]
            print(f"  > b: {np.array2string(b_curr, precision=3, suppress_small=True)}")
            print(f"  > c: {np.array2string(c_curr, precision=3, suppress_small=True)}")
            print(f"  > E: {np.array2string(E_curr, precision=3, suppress_small=True)}")

            # Print Per-dim metrics
            print("  Val Per-dim MRE:", np.array2string(val_mre_per_dim, precision=4, suppress_small=True))
            print("  Val Per-dim MAE:", np.array2string(val_mae_per_dim, precision=4, suppress_small=True))
            print("  Val Per-dim MSE:", np.array2string(val_mse_per_dim, precision=4, suppress_small=True))

            if len(valid_pos)==N and False:
                print("mean 13 val:{}".format((val_mre_o*17-val_mre_per_dim[[2,7,9,13]].sum())/13))
            print(flush=True)
        else:
            print(
                f"Basin {basin_counter}/{N_BASINS}: obj={f:.6e} [{status}], Val metrics=invalid, elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
        return False

    res = basinhopping(
        inverse_objective_func,
        init_guess,
        niter=N_BASINS,
        T=BH_TEMPERATURE,
        stepsize=STEP_SIZE,
        minimizer_kwargs=minimizer_kwargs,
        callback=bh_callback,
        disp=True,
    )

    # Use the best run for reporting
    if best_res is None:
        best_res = res.lowest_optimization_result
        best_fun = best_res.fun
    res = best_res
    print(f"Best objective: {best_fun:.6e}")
    print(f"Optimization finished in {time.time()-t0:.2f}s")

    # --- 5. Validation ---
    p_opt = res.x
    b_rec = p_opt[:N]
    c_rec = p_opt[N:2*N]
    E_rec = p_opt[2*N:]
    print("c_rec:", c_rec)
    print("E_rec:", E_rec)
    print("b_rec:", b_rec)

    # --- 6. Compute MRE (Mean Relative Error) ---
    print("\n" + "="*50)
    print("Computing Train/Validation Fit Metrics...")
    print("="*50)

    # Train metrics
    train_metrics = compute_metrics(p_opt, h_train, L_train)
    if train_metrics is not None:
        train_mre, train_mae, train_rmse, train_mre_per_dim, train_mae_per_dim, train_mse_per_dim = train_metrics
        print("Train Fit Results:")
        print(f"  MRE (Mean Relative Error):  {train_mre:.6f} ({train_mre*100:.4f}%)")
        print(f"  MAE (Mean Absolute Error):  {train_mae:.6f}")
        print(f"  RMSE (Root MSE):            {train_rmse:.6f}")
    else:
        print("Train metrics: Invalid")

    print("-"*50)

    # Validation metrics
    val_metrics = compute_metrics(p_opt, h_val, L_val)
    if val_metrics is not None:
        val_mre, val_mae, val_rmse, val_mre_per_dim, val_mae_per_dim, val_mse_per_dim = val_metrics
        print("Validation Fit Results:")
        print(f"  MRE (Mean Relative Error):  {val_mre:.6f} ({val_mre*100:.4f}%)")
        print(f"  MAE (Mean Absolute Error):  {val_mae:.6f}")
        print(f"  RMSE (Root MSE):            {val_rmse:.6f}")
        print("-"*50)
        print("Validation Per-dimension Metrics:")
        for idx, (mre, mae, mse) in enumerate(zip(val_mre_per_dim, val_mae_per_dim, val_mse_per_dim)):
            print(f"  dim {idx:02d}: MRE={mre:.6f} ({mre*100:.4f}%), MAE={mae:.6f}, MSE={mse:.6f}")
    else:
        print("Validation metrics: Invalid")

    print("="*50)

    # Print recovered parameters summary
    print(f"\nRecovered Parameters:")
    print(f"  b (exponents)   - mean: {np.mean(b_rec):.6f}, std: {np.std(b_rec):.6f}")
    print(f"  c (coefficients) - mean: {np.mean(c_rec):.6f}, std: {np.std(c_rec):.6f}")
    print(f"  E (offsets)     - mean: {np.mean(E_rec):.6f}, std: {np.std(E_rec):.6f}")
    print("="*50)