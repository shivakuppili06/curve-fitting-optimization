# Parametric curve parameter recovery (theta, M, X)
#
# Trick: shift (x-X, y-42) and rotate by theta.
# Projecting onto direction vector (cos theta, sin theta) cancels out the sin(0.3t) wave
# and gives t directly: t = (x - X)*cos(theta) + (y - 42)*sin(theta).
# The orthogonal projection isolates the wave: b = e^(M*t) * sin(0.3*t).
# No 1D root search needed per point!

import numpy as np
from scipy.optimize import least_squares

def load_data(filepath='xy_data.csv'):
    pts = np.loadtxt(filepath, delimiter=',', skiprows=1)
    return pts[:, 0], pts[:, 1]

# --- Naive attempt (kept for reference) ---
# def solve_t_per_point(x, y, theta, M, X):
#     # Originally tried scipy.optimize.root_scalar for each point to find t.
#     # Extremely slow (~15-20 sec per residual call). Replaced with closed-form projection below.
#     pass

def calc_residuals(params, x, y):
    theta, M, X = params
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    u = x - X
    v = y - 42.0

    # closed form t recovery via projection along tangent direction
    t_rec = u * cos_t + v * sin_t

    # orthogonal component: observed vs predicted envelope
    b_obs = -u * sin_t + v * cos_t
    b_pred = np.exp(M * t_rec) * np.sin(0.3 * t_rec)

    return b_obs - b_pred

def fit_parameters(x, y):
    # Parameter bounds from assignment spec:
    # 0 < theta < 50 deg, -0.05 < M < 0.05, 0 < X < 100
    bounds = (
        [0.0, -0.05, 0.0],
        [np.deg2rad(50.0), 0.05, 100.0]
    )
    init_guess = [np.deg2rad(25.0), 0.01, 50.0]

    opt_res = least_squares(calc_residuals, init_guess, args=(x, y), bounds=bounds)
    return opt_res

def eval_curve(t, theta, M, X):
    amp = np.exp(M * np.abs(t))
    x_val = t * np.cos(theta) - amp * np.sin(0.3 * t) * np.sin(theta) + X
    y_val = 42.0 + t * np.sin(theta) + amp * np.sin(0.3 * t) * np.cos(theta)
    return x_val, y_val

def main():
    x, y = load_data('xy_data.csv')

    res = fit_parameters(x, y)
    theta_opt, M_opt, X_opt = res.x

    deg_theta = np.degrees(theta_opt)

    # Reconstruct points using calculated t to check distance errors
    ct, st = np.cos(theta_opt), np.sin(theta_opt)
    t_arr = (x - X_opt) * ct + (y - 42.0) * st
    x_rec, y_rec = eval_curve(t_arr, theta_opt, M_opt, X_opt)

    # Error metrics (L2 and L1 per point)
    err_l2 = np.sqrt((x - x_rec)**2 + (y - y_rec)**2)
    err_l1 = np.abs(x - x_rec) + np.abs(y - y_rec)

    print("Recovered parameters:")
    print(f"  theta = {deg_theta:.6f} deg  ({theta_opt:.6f} rad)")
    print(f"  M     = {M_opt:.6f}")
    print(f"  X     = {X_opt:.6f}")
    print()
    print("Reconstruction error (L2 Euclidean distance per point):")
    print(f"  mean   = {err_l2.mean():.8f}")
    print(f"  median = {np.median(err_l2):.8f}")
    print(f"  max    = {err_l2.max():.8f}")
    print(f"  p90    = {np.percentile(err_l2, 90):.8f}")
    print()
    print("Reconstruction error (L1 distance per point):")
    print(f"  mean   = {err_l1.mean():.8f}")
    print(f"  median = {np.median(err_l1):.8f}")
    print(f"  max    = {err_l1.max():.8f}")
    print()
    print(f"Recovered t range: [{t_arr.min():.3f}, {t_arr.max():.3f}]  (expected 6 < t < 60)")

    with open("results.txt", "w") as f:
        f.write("Recovered parameters\n")
        f.write(f"theta_deg = {deg_theta}\n")
        f.write(f"theta_rad = {theta_opt}\n")
        f.write(f"M = {M_opt}\n")
        f.write(f"X = {X_opt}\n\n")
        f.write("Reconstruction error (L2)\n")
        f.write(f"mean = {err_l2.mean()}\n")
        f.write(f"median = {np.median(err_l2)}\n")
        f.write(f"max = {err_l2.max()}\n")
        f.write(f"p90 = {np.percentile(err_l2, 90)}\n\n")
        f.write("Reconstruction error (L1)\n")
        f.write(f"mean = {err_l1.mean()}\n")
        f.write(f"median = {np.median(err_l1)}\n")
        f.write(f"max = {err_l1.max()}\n")

if __name__ == '__main__':
    main()
