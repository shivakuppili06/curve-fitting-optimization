# Parametric Curve Parameter Recovery

This repo contains my solution for recovering the unknown parameters $\theta$, $M$, and $X$ from the 1,500 2D data points in `xy_data.csv`.

---

## Process & Approach

### 1. The Parametric Equations
The sampled curve follows:
$$x(t) = t \cos(\theta) - e^{M|t|} \sin(0.3t) \sin(\theta) + X$$
$$y(t) = 42 + t \sin(\theta) + e^{M|t|} \sin(0.3t) \cos(\theta)$$

Given bounds:
- $0^\circ < \theta < 50^\circ$
- $-0.05 < M < 0.05$
- $0 < X < 100$
- $6 < t < 60$

### 2. Initial Attempt (What Didn't Work)
My first instinct was to treat $t_i$ for each point $(x_i, y_i)$ as an additional unknown and solve a 1D non-linear equation per point (using `scipy.optimize.root_scalar` or a 1D grid search). 

While this works in theory, doing a 1D root search across all 1,500 points at every iteration of `least_squares` was extremely slow (taking ~15–20 seconds per residual evaluation).

### 3. The Rotation Trick (Closed-Form $t$)
Looking closer at the geometry, I realized the curve consists of a linear trajectory along angle $\theta$ plus a perpendicular sinusoidal oscillation:

Let $u = x - X$ and $v = y - 42$.
1. **Projection along $(\cos\theta, \sin\theta)$**:
   $$u \cos\theta + v \sin\theta = t \left(\cos^2\theta + \sin^2\theta\right) + e^{Mt}\sin(0.3t)\left(-\sin\theta\cos\theta + \cos\theta\sin\theta\right)$$
   Notice the $\sin(0.3t)$ oscillation cancels out completely!
   $$t_i = (x_i - X)\cos\theta + (y_i - 42)\sin\theta$$

2. **Orthogonal Projection**:
   $$-u \sin\theta + v \cos\theta = e^{M t_i} \sin(0.3 t_i)$$

This meant for any candidate triplet $(\theta, M, X)$, $t_i$ and the expected perpendicular amplitude $b_i = e^{M t_i}\sin(0.3 t_i)$ can be computed in closed form instantly without any root finding.

Plugging this into `scipy.optimize.least_squares` converges in just a few iterations.

---

## Results

### Recovered Parameters

- **$\theta$** = **$30.000^\circ$** ($0.523599\text{ rad}$)
- **$M$** = **$0.030000$**
- **$X$** = **$55.000000$**

### Reconstruction Error Metrics

| Metric | L2 Euclidean Error | L1 Distance Error |
| :--- | :--- | :--- |
| **Mean** | `0.00000256` | `0.00000350` |
| **Median** | `0.00000194` | `0.00000265` |
| **Max** | `0.00001762` | `0.00002406` |
| **90th Percentile** | `0.00000586` | — |

The recovered $t$ parameter values fall strictly within `[6.049, 59.995]`, matching the assignment bounds ($6 < t < 60$).

---

## Plot & Visualization

Running `visualize.py` generates the plot overlay of the fitted curve against the input scatter points:

![Fitted Curve](results/fitted_curve.png)

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run solver (prints recovered parameters & error metrics)
python solve.py

# Generate plot visualization (saved to results/fitted_curve.png)
python visualize.py
```

---

## Submission Details

- **Desmos Template**: [https://www.desmos.com/calculator/rfj91yrxob](https://www.desmos.com/calculator/rfj91yrxob)
- **Desmos Submission String**:
  ```latex
  \left(t*\cos(0.523599)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.523599)+55,42+t*\sin(0.523599)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.523599)\right)
  ```
