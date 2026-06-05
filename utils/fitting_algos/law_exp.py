import numpy as np
import time
from scipy.optimize import minimize, curve_fit

def compute_metrics(y_true, y_pred):
    eps = 1e-9
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    
    # Per-element metrics
    # Avoid div by zero
    mre_element = abs_diff / np.maximum(np.abs(y_true), eps)
    
    mre = np.mean(mre_element)
    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(diff**2))
    
    # Per-dimension metrics (average over samples)
    mre_per_dim = np.mean(mre_element, axis=0)
    
    return mre, mae, rmse, mre_per_dim

def law_exp(h_data, L_data, seed=42, alpha=0.1):
    """
    Fits a linear model L_i(h) = E_i + sum_j(h_j * t_ij)
    Inputs:
        h_data: (N_samples, N_features) array
        L_data: (N_samples, N_targets) array
        alpha: Regularization strength for Ridge Regression
    """
    
    # Correct shape handling
    # h_data is (Samples, Dims)
    num_samples, N_dims = h_data.shape
    
    # Inputs for regression
    H = h_data
    Y = L_data
    
    # Shuffle and Split 70/30 along the sample dimension (axis 0)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    split_idx = int(0.7 * num_samples)
    
    train_idx = perm[:split_idx]
    val_idx = perm[split_idx:]
    
    H_train = H[train_idx]
    Y_train = Y[train_idx]
    H_val = H[val_idx]
    Y_val = Y[val_idx]
    
    print(f"Data Split: {len(train_idx)} Train, {len(val_idx)} Validation")
    start_time = time.time()

    # Define the non-linear model: L = E + exp(H @ T.T)
    # curve_fit requires a function f(x, *params) -> y
    # Since we have multi-output y, we need to flatten the output.
    
    def model_func_flat(H_in, *params):
        # params is a tuple of arguments
        params = np.array(params)
        
        # Unpack parameters
        # T: (N_dims, N_dims) 
        T_flat = params[:N_dims * N_dims]
        E_bias = params[N_dims * N_dims:]
        
        T_mat = T_flat.reshape(N_dims, N_dims) 
        
        # H_in: (Samples, N_dims)
        linear_term = H_in @ T_mat.T
        
        # Prediction
        pred = E_bias + np.exp(linear_term)
        
        # Return flattened array for curve_fit
        return pred.flatten()

    def predict_model(params, H_in):
        # Wrapper to return shaped output for validation
        return model_func_flat(H_in, *params).reshape(-1, N_dims)

    # Initial Guesses
    init_E = np.min(Y_train, axis=0) - 0.1 
    rng_init = np.random.default_rng(seed)
    init_T = rng_init.normal(0, 0.1, size=N_dims * N_dims)
    
    x0 = np.concatenate([init_T, init_E])

    print("\n" + "="*50)
    print("Non-Linear Fit Results (law_exp):")
    print(f"Model: L = E + exp(H @ T.T)")
    print("="*50)
    print(f"Starting optimization (curve_fit, {len(x0)} params)...")
    
    try:
        # curve_fit(f, xdata, ydata, p0=...)
        # Note: curve_fit does not support 'alpha' regularization standardly
        # so we perform unregularized non-linear least squares here.
        p_opt, pcov = curve_fit(
            model_func_flat,
            H_train, 
            Y_train.flatten(), 
            p0=x0,
            method='trf', # Trust Region Reflective, robust
            maxfev=10000
        )
        success = True
        message = "curve_fit completed"
    except Exception as e:
        success = False
        message = str(e)
        p_opt = x0

    elapsed = time.time() - start_time
    print(f"Fitting completed in {elapsed:.4f}s. Success: {success}, Message: {message}")
    
    # Extract optimized parameters
    T_matrix = p_opt[:N_dims * N_dims].reshape(N_dims, N_dims)
    E_bias = p_opt[N_dims * N_dims:]
    
    # Predict
    Y_train_pred = predict_model(p_opt, H_train)
    Y_val_pred = predict_model(p_opt, H_val)
    
    # Compute Metrics
    tr_mre, tr_mae, tr_rmse, _ = compute_metrics(Y_train, Y_train_pred)
    val_mre, val_mae, val_rmse, val_mre_per_dim = compute_metrics(Y_val, Y_val_pred)
    
    print("\n" + "="*50)
    print("Non-Linear Fit Results (law_exp):")
    print("="*50)
    print("Train Metrics:")
    print(f"  MRE:  {tr_mre:.6f} ({tr_mre*100:.4f}%)")
    print(f"  MAE:  {tr_mae:.6f}")
    print(f"  RMSE: {tr_rmse:.6f}")
    
    print("-" * 50)
    print("Validation Metrics:")
    print(f"  MRE:  {val_mre:.6f} ({val_mre*100:.4f}%)")
    print(f"  MAE:  {val_mae:.6f}")
    print(f"  RMSE: {val_rmse:.6f}")
    
    print("\nPer-dimension Validation MRE:")
    print(val_mre_per_dim)

    print("="*50)

    # Reconstruct parameters T and E
    return T_matrix, E_bias