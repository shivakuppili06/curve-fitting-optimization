# Plot fitted parametric curve against xy_data.csv scatter points

import os
import numpy as np
import matplotlib.pyplot as plt

from solve import load_data, fit_parameters, eval_curve

def main():
    pts_x, pts_y = load_data('xy_data.csv')
    opt_res = fit_parameters(pts_x, pts_y)
    theta, M, X = opt_res.x

    # Sample dense t array for smooth visualization
    t_grid = np.linspace(6, 60, 2000)
    curve_x, curve_y = eval_curve(t_grid, theta, M, X)

    os.makedirs('results', exist_ok=True)

    plt.figure(figsize=(9, 6))
    plt.scatter(pts_x, pts_y, s=8, alpha=0.5, label='Original data (xy_data.csv)', color='tab:blue')
    plt.plot(curve_x, curve_y, color='tab:red', linewidth=1.5,
             label=f'Reconstructed Curve\nθ={np.degrees(theta):.2f}°, M={M:.4f}, X={X:.2f}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Parametric Curve Recovery Fit vs Data Points')
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join('results', 'fitted_curve.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved fit plot to {out_path}")

if __name__ == '__main__':
    main()
