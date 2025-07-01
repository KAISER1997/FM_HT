import torch
import numpy as np

# DTYPE and device selection will be handled in main or passed around.

# --- Alpha, Beta, K Calculation (Inputs are SCALAR constants) ---
def get_alpha_beta_K(mu_scalar, sigma_scalar, r_scalar, device, dtype): #CORRECT
    mu = torch.as_tensor(mu_scalar, dtype=dtype, device=device)
    sigma = torch.as_tensor(sigma_scalar, dtype=dtype, device=device)
    r = torch.as_tensor(r_scalar, dtype=dtype, device=device)

    nan_val = torch.tensor(float('nan'), dtype=dtype, device=device)
    if torch.abs(sigma) < 1e-12:
        return nan_val, nan_val, nan_val

    discriminant = mu**2 + 2 * sigma**2 * r
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))
    
    safe_sigma_sq = sigma**2
    if safe_sigma_sq < 1e-38: 
        safe_sigma_sq = torch.tensor(1e-38, dtype=dtype, device=device)

    alpha = (-mu + sqrt_discriminant) / safe_sigma_sq
    neg_beta_sol = (-mu - sqrt_discriminant) / safe_sigma_sq
    beta = -neg_beta_sol
    
    sum_alpha_beta = alpha + beta
    K_val = nan_val
    if torch.abs(sum_alpha_beta) > 1e-12:
        K_val = (alpha * beta) / sum_alpha_beta
    elif torch.abs(alpha) < 1e-12 and torch.abs(beta) < 1e-12:
        K_val = torch.tensor(0.0, dtype=dtype, device=device)
    return alpha, beta, K_val

# --- f_LDP and its derivative (K_s, alpha_s, beta_s, x_R_s are SCALAR constants) ---
# def f_LDP_func(x_b, K_s, alpha_s, beta_s, x_R_s, dtype):                      #CORRECT
#     exp_clamp_val = 80.0 if dtype == torch.float32 else 700.0
#     print("a",beta_s,x_b,x_R_s,exp_clamp_val)
#     term1_exp_arg = torch.clamp(beta_s * (x_b - x_R_s), max=exp_clamp_val)
#     term2_exp_arg = torch.clamp(-alpha_s * (x_b - x_R_s), max=exp_clamp_val)
#     val_le_xR = K_s * torch.exp(term1_exp_arg)
#     val_ge_xR = K_s * torch.exp(term2_exp_arg)
#     return torch.where(x_b <= x_R_s, val_le_xR, val_ge_xR)


def f_LDP_func(x_b, K_s, alpha_s, beta_s, x_R_s, dtype):  # CORRECT
    # Ensure all inputs are PyTorch tensors with correct dtype
    x_b = torch.as_tensor(x_b, dtype=dtype)
    K_s = torch.as_tensor(K_s, dtype=dtype)
    alpha_s = torch.as_tensor(alpha_s, dtype=dtype)
    beta_s = torch.as_tensor(beta_s, dtype=dtype)
    x_R_s = torch.as_tensor(x_R_s, dtype=dtype)

    # Set clamping value to avoid overflow in exponentials
    exp_clamp_val = 80.0 if dtype == torch.float32 else 700.0

    # Compute the exponent arguments safely
    term1_exp_arg = torch.clamp(beta_s * (x_b - x_R_s), max=exp_clamp_val)
    term2_exp_arg = torch.clamp(-alpha_s * (x_b - x_R_s), max=exp_clamp_val)

    # Compute function values on each side of x_R_s
    val_le_xR = K_s * torch.exp(term1_exp_arg)
    val_ge_xR = K_s * torch.exp(term2_exp_arg)

    # Return result depending on whether x_b <= x_R_s
    return torch.where(x_b <= x_R_s, val_le_xR, val_ge_xR)


def f_LDP_prime_func(x_b, K_s, alpha_s, beta_s, x_R_s, dtype):              #CORRECT
    exp_clamp_val = 80.0 if dtype == torch.float32 else 700.0
    term1_exp_arg = torch.clamp(beta_s * (x_b - x_R_s), max=exp_clamp_val)
    term2_exp_arg = torch.clamp(-alpha_s * (x_b - x_R_s), max=exp_clamp_val)
    deriv_le_xR = K_s * beta_s * torch.exp(term1_exp_arg)
    deriv_gt_xR = K_s * (-alpha_s) * torch.exp(term2_exp_arg)
    return torch.where(x_b <= x_R_s, deriv_le_xR, deriv_gt_xR)

# --- Gaussian PDF and its derivatives (mean, std_dev can be batched) ---
def gaussian_pdf(x, mean, std_dev, device, dtype):     #CORRECT
    safe_std_dev = torch.clamp(std_dev, min=1e-12) 
    var = safe_std_dev**2
    pi_tensor = torch.tensor(np.pi, dtype=dtype, device=device)
    denom = safe_std_dev * torch.sqrt(2 * pi_tensor)
    denom = torch.clamp(denom, min=1e-38)
    exp_term = torch.exp(-((x - mean)**2) / torch.clamp(2 * var, min=1e-38))
    return exp_term / denom

def d_gaussian_dx_pdf(x, mean, std_dev, device, dtype):  #CORRECT
    safe_std_dev = torch.clamp(std_dev, min=1e-12)
    var = safe_std_dev**2
    pdf_val = gaussian_pdf(x, mean, std_dev, device, dtype) 
    factor = -(x - mean) / torch.clamp(var, min=1e-38)
    is_std_dev_effectively_zero = (std_dev < 1e-9)
    return torch.where(is_std_dev_effectively_zero, torch.zeros_like(x, device=device, dtype=dtype), pdf_val * factor)

# --- f_W, F_N and their derivatives ---
# x_b:(B), x_0_b:(B), t_b:(B). mu_s, sigma_s: SCALAR constants
def f_W_func(x_b, x_0_b, mu_s, t_b, sigma_s, device, dtype,gt):      #CORRECT
    mean_b = x_0_b + mu_s * t_b # (B) + scalar * (B) -> (B)

    if gt:
        print("gt2")
        mean_b=((x_0_b*0 + mu_s * t_b))-5
        std_dev_b=torch.sqrt(sigma_s**2*t_b+1)
    else:
        std_dev_b = sigma_s * torch.sqrt(torch.clamp(t_b, min=1e-12)) # scalar * sqrt(B) -> (B)
    return gaussian_pdf(x_b, mean_b, std_dev_b, device, dtype)

def d_f_W_dx_func(x_b, x_0_b, mu_s, t_b, sigma_s, device, dtype,gt):  #CORRECT
    mean_b = x_0_b + mu_s * t_b
    if gt:
        print("gt")
        mean_b=((x_0_b*0 + mu_s * t_b))-5
        std_dev_b=torch.sqrt(sigma_s**2*t_b+1)
    else:
        std_dev_b = sigma_s * torch.sqrt(torch.clamp(t_b, min=1e-12))
    return d_gaussian_dx_pdf(x_b, mean_b, std_dev_b, device, dtype)

# For F_N used in convolution:
# x_y_grid_b: (B_sub, N_y), mu_s: SCALAR, t_sub_b: (B_sub), sigma_s: SCALAR
def F_N_func(x_y_grid_b, mu_s, t_sub_b, sigma_s, device, dtype):     #CORRECT
    mean_b_expanded = (mu_s * t_sub_b).unsqueeze(1) 
    std_dev_b_expanded = (sigma_s * torch.sqrt(torch.clamp(t_sub_b, min=1e-12))).unsqueeze(1)
    return gaussian_pdf(x_y_grid_b, mean_b_expanded, std_dev_b_expanded, device, dtype)

def F_N_prime_func(x_y_grid_b, mu_s, t_sub_b, sigma_s, device, dtype):   #CORRECT
    mean_b_expanded = (mu_s * t_sub_b).unsqueeze(1)
    std_dev_b_expanded = (sigma_s * torch.sqrt(torch.clamp(t_sub_b, min=1e-12))).unsqueeze(1)
    return d_gaussian_dx_pdf(x_y_grid_b, mean_b_expanded, std_dev_b_expanded, device, dtype)

# --- Convolutions (No change here, as x_0 does not directly enter convolution kernel F_N) ---
# x_t_b:(B), K_s,alpha_s,beta_s,x_R_s:SCALAR, mu_s,sigma_s:SCALAR, t_b:(B)
def _convolve_core(x_t_b, K_s, alpha_s, beta_s, x_R_s, mu_s, t_b, sigma_s,        #NEED TO CHECK 
                   y_grid_num_std, num_y_points_float, device, dtype,
                   fn_values_on_grid_func, 
                   fn_at_xt_minus_offset_func 
                   ):
    num_y_points = int(num_y_points_float)
    F_N_mean_b = mu_s * t_b
    F_N_std_dev_b = sigma_s * torch.sqrt(torch.clamp(t_b, min=1e-12))
    is_delta_case = F_N_std_dev_b < 1e-9
    conv_result = torch.empty_like(x_t_b, dtype=dtype, device=device)

    if torch.any(is_delta_case):
        x_t_delta = x_t_b[is_delta_case]
        F_N_mean_delta = F_N_mean_b[is_delta_case]
        conv_result[is_delta_case] = fn_at_xt_minus_offset_func(
            x_t_delta - F_N_mean_delta, K_s, alpha_s, beta_s, x_R_s, dtype)

    non_delta_mask = ~is_delta_case
    if torch.any(non_delta_mask):
        x_t_nd = x_t_b[non_delta_mask]
        t_nd = t_b[non_delta_mask]
        F_N_mean_nd = F_N_mean_b[non_delta_mask]
        F_N_std_dev_nd = F_N_std_dev_b[non_delta_mask]
        y_min_nd = F_N_mean_nd - y_grid_num_std * F_N_std_dev_nd
        y_max_nd = F_N_mean_nd + y_grid_num_std * F_N_std_dev_nd
        normalized_steps = torch.linspace(0, 1, num_y_points, dtype=dtype, device=device).unsqueeze(0)
        y_grid_nd = y_min_nd.unsqueeze(1) + (y_max_nd - y_min_nd).unsqueeze(1) * normalized_steps
        dy_nd = (y_max_nd - y_min_nd) / (num_y_points - 1) if num_y_points > 1 \
                else torch.ones_like(y_max_nd, dtype=dtype, device=device)
        fn_vals_nd = fn_values_on_grid_func(y_grid_nd, mu_s, t_nd, sigma_s, device, dtype)
        x_t_minus_y_nd = x_t_nd.unsqueeze(1) - y_grid_nd
        f_LDP_shifted_values_nd = f_LDP_func(x_t_minus_y_nd, K_s, alpha_s, beta_s, x_R_s, dtype)
        integrand_nd = fn_vals_nd * f_LDP_shifted_values_nd
        conv_sum_nd = torch.sum(integrand_nd * dy_nd.unsqueeze(1), dim=-1).double()
        conv_result[non_delta_mask] = conv_sum_nd
    return conv_result

def convolve_fN_fLDP(*args): #CORRECT
    return _convolve_core(*args, fn_values_on_grid_func=F_N_func, fn_at_xt_minus_offset_func=f_LDP_func)
def convolve_fNprime_fLDP(*args): #CORRECT
    return _convolve_core(*args, fn_values_on_grid_func=F_N_prime_func, fn_at_xt_minus_offset_func=f_LDP_prime_func)

# --- p_t and its derivative dp_t/dx_t ---
# x_t_b:(B), x_0_b:(B), mu_s,sigma_s,r_s,x_R_s:SCALAR, t_b:(B), K_s,alpha_s,beta_s:SCALAR
def p_t_calculate(x_t_b, x_0_b, mu_s, sigma_s, r_s, t_b, x_R_s, K_s, alpha_s, beta_s,
                  y_grid_num_std, num_y_points, device, dtype,gt):  #CORRECT
    f_ldp_val = f_LDP_func(x_t_b, K_s, alpha_s, beta_s, x_R_s, dtype)
    C_exp = torch.exp(-r_s * t_b) 
    # Pass batched x_0_b to f_W_func
    f_w_val = f_W_func(x_t_b, x_0_b, mu_s, t_b, sigma_s, device, dtype,gt)
    conv_args = (x_t_b, K_s, alpha_s, beta_s, x_R_s, mu_s, t_b, sigma_s,
                 y_grid_num_std, num_y_points, device, dtype)
    conv_val = convolve_fN_fLDP(*conv_args)
    p_t_val = f_ldp_val + C_exp * (f_w_val - conv_val)
    return p_t_val

def dp_dx_t_calculate(x_t_b, x_0_b, mu_s, sigma_s, r_s, t_b, x_R_s, K_s, alpha_s, beta_s,    #CORRECT
                      y_grid_num_std, num_y_points, device, dtype,gt):
    dfLDP_dx_val = f_LDP_prime_func(x_t_b, K_s, alpha_s, beta_s, x_R_s, dtype)
    C_exp = torch.exp(-r_s * t_b)
    # Pass batched x_0_b to d_f_W_dx_func
    dfW_dx_val = d_f_W_dx_func(x_t_b, x_0_b, mu_s, t_b, sigma_s, device, dtype,gt)
    conv_args = (x_t_b, K_s, alpha_s, beta_s, x_R_s, mu_s, t_b, sigma_s,
                 y_grid_num_std, num_y_points, device, dtype)
    dconv_dx_val = convolve_fNprime_fLDP(*conv_args)
    dp_dx_val = dfLDP_dx_val + C_exp * dfW_dx_val - C_exp * dconv_dx_val
    return dp_dx_val

# --- Main function ---
def get_derivative_log_pt_manual(
    x_t_val,    # (B) or scalar
    x_0_param,  # (B) or scalar
    # SCALAR parameters:
    mu_param, sigma_param, r_param, x_R_param, 
    t_param,    # (B) or scalar
    device, dtype, 
    y_grid_num_std=5.0, num_y_points=1000.0,
    log_eps=1e-38,
    gt=False
):
    # Determine if original inputs were scalar-like for return type handling
    is_x_scalar_like = isinstance(x_t_val, (float, int, np.generic)) or \
                       (isinstance(x_t_val, (np.ndarray, torch.Tensor)) and x_t_val.ndim == 0 and np.size(x_t_val) == 1)
    is_x0_scalar_like = isinstance(x_0_param, (float, int, np.generic)) or \
                       (isinstance(x_0_param, (np.ndarray, torch.Tensor)) and x_0_param.ndim == 0 and np.size(x_0_param) == 1)
    is_t_scalar_like = isinstance(t_param, (float, int, np.generic)) or \
                       (isinstance(t_param, (np.ndarray, torch.Tensor)) and t_param.ndim == 0 and np.size(t_param) == 1)


    def _to_tensor_ensure_batch(val, ref_shape_for_expand, name):
        if not isinstance(val, torch.Tensor):
            np_dtype_val = np.float64 if dtype == torch.float64 else np.float32
            tensor_val = torch.from_numpy(np.asarray(val, dtype=np_dtype_val)).to(dtype=dtype, device=device)
        else:
            tensor_val = val.to(dtype=dtype, device=device)
        
        if tensor_val.ndim == 0: 
            tensor_val = tensor_val.unsqueeze(0) # Make it (1,)
        
        # Expand if it's (1,) and ref_shape is (B>1,)
        if tensor_val.shape[0] == 1 and ref_shape_for_expand[0] > 1:
            tensor_val = tensor_val.expand(ref_shape_for_expand)
        # Check for mismatch if both are multi-element batched
        elif tensor_val.shape[0] > 1 and ref_shape_for_expand[0] > 1 and tensor_val.shape[0] != ref_shape_for_expand[0]:
            raise ValueError(f"Batch dimension mismatch for {name}: {tensor_val.shape} vs reference {ref_shape_for_expand}")
        return tensor_val

    # First, determine the target batch shape (max of x_t, x_0, t batch dims)
    # Convert all to tensor and ensure at least 1D (shape (1,) if scalar)
    initial_tensors = []
    for param_val, param_name in [(x_t_val, "x_t"), (x_0_param, "x_0"), (t_param, "t")]:
        if not isinstance(param_val, torch.Tensor):
            np_dtype_val = np.float64 if dtype == torch.float64 else np.float32
            tensor_p = torch.from_numpy(np.asarray(param_val, dtype=np_dtype_val)).to(dtype=dtype, device=device)
        else:
            tensor_p = param_val.to(dtype=dtype, device=device)
        if tensor_p.ndim == 0: tensor_p = tensor_p.unsqueeze(0)
        initial_tensors.append(tensor_p)
    
    # Determine max batch size
    max_batch_size = 1
    for tns in initial_tensors:
        if tns.shape[0] > 1:
            if max_batch_size == 1: max_batch_size = tns.shape[0]
            elif max_batch_size != tns.shape[0]:
                raise ValueError("Inconsistent batch sizes among x_t, x_0, t.")
    
    target_batch_shape = (max_batch_size,)

    x_t_tensor = _to_tensor_ensure_batch(x_t_val, target_batch_shape, "x_t_val")
    x_0_tensor = _to_tensor_ensure_batch(x_0_param, target_batch_shape, "x_0_param")
    t_tensor = _to_tensor_ensure_batch(t_param, target_batch_shape, "t_param")

    # Scalar SDE parameters to 0-dim tensors
    mu_s = torch.as_tensor(mu_param, dtype=dtype, device=device)
    sigma_s = torch.as_tensor(sigma_param, dtype=dtype, device=device)
    r_s = torch.as_tensor(r_param, dtype=dtype, device=device)
    x_R_s = torch.as_tensor(x_R_param, dtype=dtype, device=device)

    alpha_s, beta_s, K_s = get_alpha_beta_K(mu_s, sigma_s, r_s, device, dtype)
    
    if torch.isnan(alpha_s).any() or torch.isnan(beta_s).any() or torch.isnan(K_s).any():
        nan_result = torch.full_like(x_t_tensor, float('nan'), dtype=dtype, device=device)
        if is_x_scalar_like and is_x0_scalar_like and is_t_scalar_like : return nan_result.squeeze().item()
        # If original inputs were all scalar-like but became (1,), squeeze back to 0-dim tensor
        if is_x_scalar_like and is_x0_scalar_like and is_t_scalar_like and nan_result.shape == (1,): return nan_result.squeeze(0)
        return nan_result

    # Pass the batched x_0_tensor
    p_t_val = p_t_calculate(x_t_tensor, x_0_tensor, mu_s, sigma_s, r_s, t_tensor, x_R_s, K_s, alpha_s, beta_s,
                            y_grid_num_std, num_y_points, device, dtype,gt)
    dp_dx_t_val = dp_dx_t_calculate(x_t_tensor, x_0_tensor, mu_s, sigma_s, r_s, t_tensor, x_R_s, K_s, alpha_s, beta_s,
                                    y_grid_num_std, num_y_points, device, dtype,gt)

    clamped_p_t_val = torch.clamp(p_t_val, min=log_eps)
    deriv_log_pt = dp_dx_t_val / clamped_p_t_val
    
    # Handle return type
    if is_x_scalar_like and is_x0_scalar_like and is_t_scalar_like and deriv_log_pt.numel() == 1:
        return deriv_log_pt.squeeze().item() # Return Python scalar if all inputs were scalar-like
    # If all inputs were scalar-like, result was (1,), squeeze to 0-dim tensor
    if is_x_scalar_like and is_x0_scalar_like and is_t_scalar_like and deriv_log_pt.shape == (1,):
        return deriv_log_pt.squeeze(0)
        
    return deriv_log_pt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float64
    print(f"Using device: {device}, DTYPE: {DTYPE}")

    # Scalar SDE parameters (mu, sigma, r, x_R)
    mu_val = 0.05
    sigma_val = 0.2
    r_val = 0.02
    x_R_val = 0.0
    
    print(f"Scalar SDE Params: mu={mu_val}, sigma={sigma_val}, r={r_val}, x_R={x_R_val}")

    # Test 1: All inputs scalar
    print("\n--- Test 1: All inputs scalar ---")
    xt1 = 0.25
    x01 = 0.5
    t1 = 1.0
    deriv1 = get_derivative_log_pt_manual(xt1, x01, mu_val, sigma_val, r_val, x_R_val, t1, device, DTYPE, num_y_points=201)
    print(f"x_t={xt1}, x_0={x01}, t={t1}, Derivative={deriv1} (type: {type(deriv1)})")

    # Test 2: Batched x_t, x_0, t
    print("\n--- Test 2: Batched x_t, x_0, t (Matching Sizes) ---")
    xt2 = np.array([-0.1, 0.1, 0.3])
    x02 = np.array([0.4, 0.5, 0.6])
    t2 = np.array([0.8, 1.0, 1.2])
    deriv2 = get_derivative_log_pt_manual(xt2, x02, mu_val, sigma_val, r_val, x_R_val, t2, device, DTYPE, num_y_points=201)
    print(f"x_t={xt2.tolist()}, x_0={x02.tolist()}, t={t2.tolist()}, Derivatives={deriv2.cpu().numpy().tolist()}")

    # Test 3: Batched x_t, Scalar x_0, Batched t (x_0 will be expanded)
    print("\n--- Test 3: Batched x_t, Scalar x_0, Batched t ---")
    xt3 = np.array([-0.2, 0.2])
    x03 = 0.5 # Scalar x_0
    t3 = np.array([0.9, 1.1])
    deriv3 = get_derivative_log_pt_manual(xt3, x03, mu_val, sigma_val, r_val, x_R_val, t3, device, DTYPE, num_y_points=201)
    print(f"x_t={xt3.tolist()}, x_0={x03}, t={t3.tolist()}, Derivatives={deriv3.cpu().numpy().tolist()}")

    # Test 4: Scalar x_t, Batched x_0, Batched t (x_t will be expanded)
    print("\n--- Test 4: Scalar x_t, Batched x_0, Batched t ---")
    xt4 = 0.1
    x04 = np.array([0.45, 0.55])
    t4 = np.array([0.85, 1.05])
    deriv4 = get_derivative_log_pt_manual(xt4, x04, mu_val, sigma_val, r_val, x_R_val, t4, device, DTYPE, num_y_points=201)
    print(f"x_t={xt4}, x_0={x04.tolist()}, t={t4.tolist()}, Derivatives={deriv4.cpu().numpy().tolist()}")

    # Test 5: Mismatched batch sizes for x_t and x_0 (Error expected)
    print("\n--- Test 5: Mismatched batch sizes (Error expected) ---")
    xt5 = np.array([-0.1, 0.1, 0.3]) # B=3
    x05 = np.array([0.4, 0.5])       # B=2
    t5 = 1.0                         # Scalar t
    try:
        deriv5 = get_derivative_log_pt_manual(xt5, x05, mu_val, sigma_val, r_val, x_R_val, t5, device, DTYPE, num_y_points=201)
        print(f"Derivatives (should not reach here): {deriv5.cpu().numpy().tolist()}")
    except ValueError as e:
        print(f"Caught expected error: {e}")