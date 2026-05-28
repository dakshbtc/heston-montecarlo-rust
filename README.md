# Heston Model Monte Carlo Pricing Engine

A high-performance European option pricer written in Rust, implementing the
Heston (1993) stochastic volatility model via Monte Carlo simulation.

---

## Table of Contents

1. [Why Not Black-Scholes?](#1-why-not-black-scholes)
2. [The Heston Model](#2-the-heston-model)
3. [Risk-Neutral Pricing](#3-risk-neutral-pricing)
4. [Discretisation — Euler-Maruyama](#4-discretisation--euler-maruyama)
5. [Correlated Brownian Motions via Cholesky](#5-correlated-brownian-motions-via-cholesky)
6. [Numerical Stability Fixes](#6-numerical-stability-fixes)
7. [Monte Carlo Estimator](#7-monte-carlo-estimator)
8. [Variance Reduction — Antithetic Variates](#8-variance-reduction--antithetic-variates)
9. [Performance Optimisations](#9-performance-optimisations)
10. [Put-Call Parity Sanity Check](#10-put-call-parity-sanity-check)
11. [Project Structure](#11-project-structure)
12. [Running](#12-running)

---

## 1. Why Not Black-Scholes?

Black-Scholes (1973) assumes that the asset's **volatility is constant** over
time. In reality:

- Implied volatility extracted from market option prices forms a **smile** (or
  skew) — deep in/out-of-the-money options trade at higher implied vol than
  at-the-money options. A constant-vol model cannot reproduce this.
- **Volatility clusters** — calm periods are followed by turbulent ones
  (GARCH-like behaviour in returns).
- **Leverage effect** — when equity prices fall, volatility tends to rise
  (negative price/vol correlation).

The Heston model addresses all three by making variance itself a stochastic
process.

---

## 2. The Heston Model

Proposed by Steven Heston (1993), the model specifies two coupled stochastic
differential equations (SDEs) under the risk-neutral measure \(\mathbb{Q}\):

### Asset Price SDE

$$dS_t = r \, S_t \, dt + \sqrt{v_t} \, S_t \, dW_t^1$$

The asset grows at the risk-free rate \(r\) and has **instantaneous volatility
\(\sqrt{v_t}\)** that changes over time.

### Variance SDE (Cox-Ingersoll-Ross process)

$$dv_t = \kappa(\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^2$$

This is the **CIR process** (Cox, Ingersoll, Ross 1985). Key properties:

| Symbol | Name | Meaning |
|--------|------|---------|
| \(\kappa\) | Mean reversion speed | How quickly \(v\) snaps back to \(\theta\) |
| \(\theta\) | Long-run variance | The equilibrium level variance is pulled toward |
| \(\sigma\) | Vol of vol | How noisy the variance process is |
| \(\rho\) | Correlation | \(dW^1 \cdot dW^2 = \rho \, dt\) |
| \(v_0\) | Initial variance | Starting value of the variance process |

### The Correlation

$$dW_t^1 \, dW_t^2 = \rho \, dt$$

A **negative \(\rho\)** (typically \(-0.5\) to \(-0.8\) for equities) captures
the **leverage effect**: when the stock price falls (negative \(dW^1\)), variance
tends to spike (positive \(dW^2\)).

### Feller Condition

For variance to remain strictly positive almost surely, the parameters must satisfy:

$$2\kappa\theta > \sigma^2$$

If this is violated, variance can touch zero and spend time there. In practice
many calibrated models violate Feller, so numerical truncation is required (see
§6).

---

## 3. Risk-Neutral Pricing

The fair price of any derivative is the **discounted expected payoff under the
risk-neutral measure \(\mathbb{Q}\)**:

$$V_0 = e^{-rT} \, \mathbb{E}^{\mathbb{Q}}\!\left[\,\Phi(S_T)\,\right]$$

For a European **call** with strike \(K\) and expiry \(T\):

$$\Phi(S_T) = \max(S_T - K,\; 0) = (S_T - K)^+$$

For a European **put**:

$$\Phi(S_T) = (K - S_T)^+$$

Because the Heston model has stochastic volatility, there is no closed-form
\(S_T\) distribution in general — we approximate the expectation by Monte Carlo.

---

## 4. Discretisation — Euler-Maruyama

We partition \([0, T]\) into \(N\) equal steps of size \(\Delta t = T/N\).

---

### 4.1 Log-Euler Scheme for \(S\) — Full Derivation

#### Step 1 — Recall the asset price SDE

$$dS_t = r\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^1$$

A naive Euler step applied directly to \(S\) gives:

$$S_{t+\Delta t} \approx S_t + r\,S_t\,\Delta t + \sqrt{v_t}\,S_t\,\Delta W_t^1$$

This is problematic because the Gaussian noise term can make \(S_{t+\Delta t}\)
negative for large \(|\Delta W_t^1|\). We fix this by working in log-space.

#### Step 2 — Apply Itô's Lemma

Itô's Lemma states: for a twice-differentiable function \(f(X_t)\) where
\(dX_t = \mu_t\,dt + \sigma_t\,dW_t\):

$$df(X_t) = f'(X_t)\,dX_t + \tfrac{1}{2}f''(X_t)\,(dX_t)^2$$

The extra \(\tfrac{1}{2}f''\) term is the **Itô correction** — it has no
analogue in ordinary calculus and arises because Brownian paths have non-zero
quadratic variation: \((dW_t)^2 = dt\).

Set \(f(S) = \ln S\), so \(f'(S) = S^{-1}\) and \(f''(S) = -S^{-2}\):

$$d(\ln S_t) = \frac{1}{S_t}\,dS_t - \frac{1}{2S_t^2}\,(dS_t)^2$$

#### Step 3 — Compute \((dS_t)^2\)

Substitute \(dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^1\) and expand, keeping
only terms of order \(dt\) (since \(dt^2 = 0\), \(dt\,dW_t = 0\), \((dW_t)^2 = dt\)):

$$\begin{aligned}
(dS_t)^2 &= \bigl(r S_t\,dt + \sqrt{v_t} S_t\,dW_t^1\bigr)^2 \\
          &= r^2 S_t^2\,dt^2
           + 2r S_t^2\sqrt{v_t}\,dt\,dW_t^1
           + v_t S_t^2\,(dW_t^1)^2 \\
          &= 0 + 0 + v_t S_t^2\,dt \\
          &= v_t S_t^2\,dt
\end{aligned}$$

#### Step 4 — Substitute back into Itô's Lemma

$$d(\ln S_t)
= \frac{1}{S_t}\bigl(r S_t\,dt + \sqrt{v_t} S_t\,dW_t^1\bigr)
  - \frac{1}{2 S_t^2}\,v_t S_t^2\,dt$$

$$= r\,dt + \sqrt{v_t}\,dW_t^1 - \frac{v_t}{2}\,dt$$

$$\boxed{d(\ln S_t) = \left(r - \tfrac{1}{2}v_t\right)dt + \sqrt{v_t}\,dW_t^1}$$

This is a **linear SDE in \(\ln S_t\)** — the drift and diffusion coefficients
no longer depend on \(S_t\) itself.

#### Step 5 — Integrate over one time step

Treat \(v_t\) as **frozen at its current value** over the interval
\([t,\, t+\Delta t]\) (this is the Euler approximation on \(v\)):

$$\ln S_{t+\Delta t} - \ln S_t
= \left(r - \tfrac{1}{2}v_t\right)\Delta t + \sqrt{v_t}\,\Delta W_t^1$$

where \(\Delta W_t^1 = Z_1\sqrt{\Delta t}\), \(Z_1 \sim \mathcal{N}(0,1)\).

#### Step 6 — Exponentiate to recover \(S\)

$$\boxed{S_{t+\Delta t} = S_t \cdot \exp\!\left[\left(r - \tfrac{1}{2}v_t\right)\Delta t + \sqrt{v_t}\,\Delta W_t^1\right]}$$

Because \(\exp(\cdot) > 0\) always, **\(S\) is guaranteed strictly positive**
on every path regardless of how large the noise term is. In code this is:

```rust
s *= ((option.r - 0.5 * v_pos) * dt + sqrt_v * dw1).exp();
//     ↑ drift correction              ↑ diffusion
//     (r - ½v)·Δt                     √v · ΔW¹
```

---

### 4.2 Euler-Maruyama Scheme for \(v\) — Full Derivation

#### Step 1 — Recall the variance SDE

$$dv_t = \underbrace{\kappa(\theta - v_t)}_{\text{drift}}\,dt
       + \underbrace{\sigma\sqrt{v_t}}_{\text{diffusion}}\,dW_t^2$$

This is the **CIR (Cox-Ingersoll-Ross)** process. Its drift is linear in
\(v_t\), pulling it toward the mean \(\theta\) with force \(\kappa\).

#### Step 2 — General Euler-Maruyama formula

For any Itô SDE \(dX_t = \mu(X_t)\,dt + \sigma(X_t)\,dW_t\), the
Euler-Maruyama discretisation at step \(t \to t + \Delta t\) is:

$$X_{t+\Delta t} = X_t + \mu(X_t)\,\Delta t + \sigma(X_t)\,\Delta W_t$$

This is the stochastic analogue of the forward Euler method for ODEs. It
achieves **strong order 0.5** and **weak order 1.0** of convergence.

#### Step 3 — Identify the coefficient functions

For the variance SDE:

$$\mu(v_t) = \kappa(\theta - v_t), \qquad \sigma(v_t) = \sigma\sqrt{v_t}$$

#### Step 4 — Write the discretised step

Substituting directly into the Euler-Maruyama formula:

$$\boxed{v_{t+\Delta t} = v_t + \kappa(\theta - v_t)\,\Delta t + \sigma\sqrt{v_t}\,\Delta W_t^2}$$

where \(\Delta W_t^2 = \bigl(\rho\,Z_1 + \sqrt{1-\rho^2}\,Z_2\bigr)\sqrt{\Delta t}\)
is the correlated Brownian increment (see §5).

In code this is:

```rust
v += heston.kappa * (heston.theta - v_pos) * dt + heston.sigma * sqrt_v * dw2;
//   ↑ mean-reversion drift                        ↑ stochastic vol-of-vol term
//   κ·(θ − v)·Δt                                  σ·√v · ΔW²
```

Note that `v_pos = v.max(0.0)` is used everywhere \(v\) appears on the
right-hand side (the full-truncation fix — see §6), but the update is
accumulated into the raw `v`. This ensures the mean-reversion drift
\(\kappa(\theta - v_t)\) always pulls toward \(\theta > 0\) even if `v`
temporarily goes negative.

#### Step 5 — Why not an exact CIR step?

The CIR process has a **known transition distribution** — the non-central
chi-squared distribution — and can be sampled exactly without discretisation
error. However, the coupling with the price SDE (through \(\rho\)) means the
joint \((S_t, v_t)\) distribution is not tractable. The exact CIR sampler for
\(v\) alone (Broadie-Kaya scheme) requires an additional integral inversion that
is expensive. The Euler scheme is standard practice for the Heston model at
fine time grids (\(\Delta t = 1/252\)).

---

## 5. Correlated Brownian Motions via Cholesky

We need two **correlated** standard normal increments at each step. The trick is
Cholesky decomposition of the 2×2 correlation matrix:

$$\Sigma = \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}
= L L^\top, \quad
L = \begin{pmatrix} 1 & 0 \\ \rho & \sqrt{1-\rho^2} \end{pmatrix}$$

Starting from two **independent** standard normals \(Z_1, Z_2 \sim \mathcal{N}(0,1)\):

$$\begin{pmatrix} \Delta W^1 \\ \Delta W^2 \end{pmatrix}
= L \begin{pmatrix} Z_1 \\ Z_2 \end{pmatrix} \sqrt{\Delta t}
= \begin{pmatrix} Z_1 \sqrt{\Delta t} \\ (\rho Z_1 + \sqrt{1-\rho^2}\,Z_2)\sqrt{\Delta t} \end{pmatrix}$$

You can verify the correlation property:

$$\mathbb{E}[\Delta W^1 \Delta W^2]
= \mathbb{E}\!\left[Z_1 \cdot (\rho Z_1 + \sqrt{1-\rho^2}\,Z_2)\right]\Delta t
= \rho\,\Delta t \checkmark$$

In code (`simulation.rs`):

```rust
let rho_perp = (1.0 - heston.rho * heston.rho).sqrt();   // √(1−ρ²)
let dw1 = z1 * sqrt_dt;
let dw2 = (heston.rho * z1 + rho_perp * z2) * sqrt_dt;
```

---

## 6. Numerical Stability Fixes

### Full-Truncation Scheme for \(v\)

The Euler step can produce negative variance, which would make \(\sqrt{v_t}\)
imaginary. The **full-truncation** scheme (Lord et al. 2010) clamps before every
use:

$$v_t^+ = \max(v_t, 0)$$

and then uses \(v_t^+\) in the drift and diffusion, but applies the update to
the raw (possibly negative) \(v_t\). This is more stable than the *reflection*
scheme \(|v_t|\) and introduces less bias.

```rust
let v_pos  = v.max(0.0);     // clamp before sqrt and drift
let sqrt_v = v_pos.sqrt();

s *= ((r - 0.5 * v_pos) * dt + sqrt_v * dw1).exp();
v += kappa * (theta - v_pos) * dt + sigma * sqrt_v * dw2;
// Note: v itself (not v_pos) is updated — raw value can go negative
```

### Why Not the Milstein Scheme?

The Milstein scheme adds a second-order correction term to the variance SDE:

$$v_{t+\Delta t}^{\text{Mil}} = v_t + \kappa(\theta - v_t)\Delta t
+ \sigma\sqrt{v_t}\,\Delta W^2 + \tfrac{1}{4}\sigma^2\!\left[(\Delta W^2)^2 - \Delta t\right]$$

It reduces the strong order of convergence from 0.5 to 1.0. However, it
complicates the truncation logic and the gain is small relative to simply using
more paths, so this implementation uses Euler for simplicity.

---

## 7. Monte Carlo Estimator

Given \(N\) simulated terminal prices \(S_T^{(1)}, \ldots, S_T^{(N)}\), the
price estimator is:

$$\hat{V}_0 = e^{-rT} \cdot \frac{1}{N} \sum_{i=1}^{N} \Phi\!\left(S_T^{(i)}\right)$$

By the Law of Large Numbers, \(\hat{V}_0 \to V_0\) as \(N \to \infty\).

### Standard Error

The Monte Carlo standard error quantifies the uncertainty of the estimate:

$$\text{SE} = e^{-rT} \cdot \frac{\hat{\sigma}_\Phi}{\sqrt{N}}$$

where \(\hat{\sigma}_\Phi^2\) is the sample variance of the payoffs:

$$\hat{\sigma}_\Phi^2 = \frac{1}{N}\sum_{i=1}^{N}\Phi_i^2 - \left(\frac{1}{N}\sum_{i=1}^{N}\Phi_i\right)^2$$

The standard error shrinks at rate \(1/\sqrt{N}\) — doubling precision requires
**4× more paths**. This is the fundamental motivation for variance reduction
techniques.

---

## 8. Variance Reduction — Antithetic Variates

### The Idea

For any path driven by normals \((Z_1, Z_2, \ldots, Z_{2N})\), we can construct
a **mirror path** driven by \((-Z_1, -Z_2, \ldots, -Z_{2N})\). Since normal
distributions are symmetric around zero, both paths are individually valid
samples. But their payoffs are negatively correlated — when one is large, the
other tends to be small.

### Variance Reduction Formula

Let \(\Phi^+\) be the payoff on the original path and \(\Phi^-\) on the
antithetic path. The antithetic estimator uses the average:

$$\Phi^{\text{AV}} = \frac{\Phi^+ + \Phi^-}{2}$$

Its variance is:

$$\text{Var}\!\left(\Phi^{\text{AV}}\right)
= \frac{\text{Var}(\Phi^+) + \text{Var}(\Phi^-) + 2\,\text{Cov}(\Phi^+, \Phi^-)}{4}$$

Since \(\Phi^+\) and \(\Phi^-\) have the same marginal distribution,
\(\text{Var}(\Phi^+) = \text{Var}(\Phi^-) \equiv \sigma^2\):

$$\text{Var}\!\left(\Phi^{\text{AV}}\right) = \frac{\sigma^2 + \text{Cov}(\Phi^+, \Phi^-)}{2}$$

Whenever \(\text{Cov}(\Phi^+, \Phi^-) < 0\), the antithetic estimator has
**strictly lower variance** than using two independent paths, yet requires only
one set of random numbers.

For call payoffs under log-normal-like dynamics, the covariance is strongly
negative (a higher terminal price on the normal path typically means a lower one
on the antithetic), so the variance reduction is substantial in practice.

### Implementation

Both paths are computed in a **single loop pass**, so there is negligible extra
compute:

```rust
// Normal path uses (z1, z2), antithetic uses (-z1, -z2)
let dw1      =  z1 * sqrt_dt;
let dw1_anti = -z1 * sqrt_dt;

let dw2      =  (rho * z1  + rho_perp * z2)  * sqrt_dt;
let dw2_anti = -(rho * z1  + rho_perp * z2)  * sqrt_dt;  // same as negating z1,z2
```

---

## 9. Performance Optimisations

### Parallelism with Rayon

Each Monte Carlo path is **independent** — a textbook embarrassingly parallel
problem. `rayon` distributes iterations across all CPU cores with zero data
sharing:

```rust
(0..half_paths)
    .into_par_iter()
    .map(|i| { /* one antithetic pair */ })
    .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
```

Each thread holds its own RNG instance, seeded deterministically from the master
seed. Results are therefore **reproducible** regardless of thread count.

### Faster RNG — SmallRng (Xoshiro256++)

`rand::rngs::StdRng` uses ChaCha12, a cryptographically secure PRNG. For Monte
Carlo simulation we need only statistical quality, not cryptographic security.
`SmallRng` (backed by Xoshiro256++) is **2–3× faster** while passing all
standard statistical tests (BigCrush, PractRand).

Seeding per path using a linear congruential hash of the path index ensures no
two threads share state:

```rust
SmallRng::seed_from_u64(master_seed ^ path_index.wrapping_mul(LCG_CONST).wrapping_add(1))
```

### Release Profile + Native CPU

```toml
[profile.release]
opt-level = 3
lto = "thin"
```

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

`target-cpu=native` enables AVX/FMA instructions, letting the compiler
auto-vectorise the inner simulation loop (multiple floating-point ops per clock
cycle).

### Combined Speedup (100k paths × 252 steps)

| Configuration | Wall time |
|---|---|
| Debug build, sequential | ~400 ms (estimated) |
| Debug build, parallel + SmallRng + antithetic | ~52 ms |
| Release build, parallel + SmallRng + antithetic | **~19 ms** |

---

## 10. Put-Call Parity Sanity Check

For European options under any model (no arbitrage is sufficient, no model
assumptions needed), the following identity holds exactly:

$$C - P = S_0 - K e^{-rT}$$

Rearranging to solve for the put price:

$$P = C - S_0 + K e^{-rT}$$

The engine computes the put price both via this identity and via direct Monte
Carlo simulation. The difference between the two results is a **model-free
correctness check** — it should converge to zero as \(N \to \infty\) and is
typically a few cents for 100k paths.

---

## 11. Project Structure

```
heston-mc/
├── Cargo.toml
└── src/
    ├── main.rs          # entry point — wires modules together, prints results
    ├── models.rs        # HestonParams struct
    ├── option.rs        # OptionParams struct
    ├── simulation.rs    # McConfig, simulate_path, simulate_path_antithetic
    └── pricing.rs       # price_european (parallel MC engine)
```

---

## 12. Running

```bash
# Debug build (slower, overflow checks enabled)
cargo run

# Optimised release build
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

### Default Parameters

| Parameter | Value | Interpretation |
|---|---|---|
| \(\kappa\) | 2.0 | Variance mean-reverts twice per year |
| \(\theta\) | 0.04 | Long-run vol = 20% |
| \(\sigma\) | 0.3 | Vol of vol = 30% |
| \(\rho\) | −0.7 | Strong leverage effect |
| \(v_0\) | 0.04 | Starts at long-run level |
| \(S_0\) | 100 | At-the-money |
| \(K\) | 100 | Strike |
| \(r\) | 5% | Risk-free rate |
| \(T\) | 1 year | One-year expiry |
| Paths | 100,000 | Monte Carlo paths |
| Steps | 252 | One per trading day |

### References

- Heston, S.L. (1993). *A Closed-Form Solution for Options with Stochastic
  Volatility.* Review of Financial Studies, 6(2), 327–343.
- Cox, J.C., Ingersoll, J.E., Ross, S.A. (1985). *A Theory of the Term
  Structure of Interest Rates.* Econometrica, 53(2), 385–408.
- Lord, R., Koekkoek, R., van Dijk, D. (2010). *A Comparison of Biased
  Simulation Schemes for Stochastic Volatility Models.* Quantitative Finance,
  10(2), 177–194.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering.*
  Springer.
