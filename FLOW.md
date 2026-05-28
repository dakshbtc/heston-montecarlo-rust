## 1. The Heston Model

Proposed by Steven Heston (1993), the model specifies two coupled stochastic
differential equations (SDEs) under the risk-neutral measure.

### Asset Price SDE

$$dS_t = r \, S_t \, dt + \sqrt{v_t} \, S_t \, dW_t^1$$

The asset grows at the risk-free rate $r$ and has **instantaneous volatility
$\sqrt{v_t}$** that changes over time.

### Variance SDE (Cox-Ingersoll-Ross process)

$$dv_t = \kappa(\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^2$$

This is the **CIR process** (Cox, Ingersoll, Ross 1985). Key properties:

| Symbol | Name | Meaning |
|--------|------|---------|
| $\kappa$ | Mean reversion speed | How quickly $v$ snaps back to $\theta$ |
| $\theta$ | Long-run variance | The equilibrium level variance is pulled toward |
| $\sigma$ | Vol of vol | How noisy the variance process is |
| $\rho$ | Correlation | $dW^1 \cdot dW^2 = \rho \, dt$ |
| $v_0$ | Initial variance | Starting value of the variance process |

### The Correlation

$$dW_t^1 \, dW_t^2 = \rho \, dt$$

A **negative $\rho$** (typically $-0.5$ to $-0.8$ for equities) captures
the **leverage effect**: when the stock price falls (negative $dW^1$), variance
tends to spike (positive $dW^2$).

---

## 2. Constructing Correlated Brownian Increments (Cholesky)

**Code lines:**
```rust
let dw1 = z1 * sqrt_dt;
let dw2 = (heston.rho * z1 + rho_perp * z2) * sqrt_dt;
```

### The Problem

At each time step we need two Brownian increments $\Delta W^1$ and $\Delta W^2$
that satisfy $\mathbb{E}[\Delta W^1 \Delta W^2] = \rho \, \Delta t$.
A computer can only generate **independent** standard normals. We need to mix them
to produce the required correlation.

### Step 1 — What we have

Draw two independent standard normals:

$$Z_1, Z_2 \sim \mathcal{N}(0,1), \quad Z_1 \perp Z_2$$

These satisfy $\mathbb{E}[Z_1 Z_2] = 0$ — they are completely uncorrelated.

### Step 2 — The correlation matrix

The joint distribution of $(\Delta W^1, \Delta W^2)$ over one step $\Delta t$
must follow a bivariate normal with covariance matrix:

$$\Sigma = \Delta t \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}$$

The diagonal entries are the variances $\mathbb{E}[(\Delta W^i)^2] = \Delta t$,
and the off-diagonal entry is the required covariance $\rho \, \Delta t$.

### Step 3 — Cholesky decomposition of the correlation matrix

We want to factorise the correlation matrix as $\Sigma_0 = L L^\top$ where $L$
is lower-triangular. Assume:

$$L = \begin{pmatrix} a & 0 \\ b & c \end{pmatrix}$$

Compute $LL^\top$:

$$LL^\top = \begin{pmatrix} a & 0 \\ b & c \end{pmatrix} \begin{pmatrix} a & b \\ 0 & c \end{pmatrix} = \begin{pmatrix} a^2 & ab \\ ab & b^2 + c^2 \end{pmatrix}$$

Match entry-by-entry against the target $\begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}$:

| Entry | Equation | Solution |
|---|---|---|
| Top-left | $a^2 = 1$ | $a = 1$ |
| Off-diagonal | $ab = \rho$ | $b = \rho$ (since $a=1$) |
| Bottom-right | $b^2 + c^2 = 1$ | $c = \sqrt{1 - \rho^2}$ |

So the Cholesky factor is:

$$L = \begin{pmatrix} 1 & 0 \\ \rho & \sqrt{1-\rho^2} \end{pmatrix}$$

### Step 4 — Transform independent normals into correlated increments

The Cholesky factor $L$ gives us correlation structure, but $Z_1, Z_2 \sim \mathcal{N}(0,1)$
are unit normals with variance 1. A Brownian increment over a time step $\Delta t$
must have variance $\Delta t$ — this comes directly from the definition of Brownian motion:

$$\mathbb{E}[(\Delta W)^2] = \Delta t$$

So we scale by $\sqrt{\Delta t}$ to go from unit variance to the correct variance:

$$\mathbb{E}\!\left[(Z_i \sqrt{\Delta t})^2\right] = \Delta t \cdot \underbrace{\mathbb{E}[Z_i^2]}_{=1} = \Delta t \checkmark$$

Multiply the Cholesky factor by the vector of independent normals and scale by $\sqrt{\Delta t}$:

$$\begin{pmatrix} \Delta W^1 \\ \Delta W^2 \end{pmatrix} = L \begin{pmatrix} Z_1 \\ Z_2 \end{pmatrix} \sqrt{\Delta t} = \begin{pmatrix} 1 & 0 \\ \rho & \sqrt{1-\rho^2} \end{pmatrix} \begin{pmatrix} Z_1 \\ Z_2 \end{pmatrix} \sqrt{\Delta t}$$

Expanding the matrix multiplication:

$$\boxed{\Delta W^1 = Z_1 \sqrt{\Delta t}}$$

$$\boxed{\Delta W^2 = \left(\rho \, Z_1 + \sqrt{1-\rho^2} \, Z_2\right) \sqrt{\Delta t}}$$

### Step 5 — Map to code

In the code, `rho_perp` $= \sqrt{1-\rho^2}$ is precomputed once before the loop:

```rust
let rho_perp = (1.0 - heston.rho * heston.rho).sqrt();   // √(1 − ρ²)
// ...
let dw1 = z1 * sqrt_dt;                                   // Z₁ · √Δt
let dw2 = (heston.rho * z1 + rho_perp * z2) * sqrt_dt;   // (ρZ₁ + √(1−ρ²)Z₂) · √Δt
```

---

## 3. Log-Euler Discretisation of the Asset Price SDE

**Code line:**
```rust
s *= ((option.r - 0.5 * v_pos) * dt + sqrt_v * dw1).exp();
```

### The Problem with Naïve Euler on S

A direct Euler step on the asset price SDE gives:

$$S_{t+\Delta t} \approx S_t + r S_t \Delta t + \sqrt{v_t} S_t \Delta W^1$$

This is dangerous — the Gaussian term $\sqrt{v_t} S_t \Delta W^1$ can be
arbitrarily negative, making $S_{t+\Delta t} < 0$, which is economically
meaningless. We fix this by working in **log-space**.

### Step 1 — Define the log-price

Let $X_t = \ln S_t$. We want the SDE that $X_t$ follows.

### Step 2 — State Itô's Lemma

For a twice-differentiable function $f(Y_t)$ where $dY_t = \mu_t \, dt + \sigma_t \, dW_t$:

$$df(Y_t) = f'(Y_t) \, dY_t + \frac{1}{2} f''(Y_t) \, (dY_t)^2$$

The extra $\frac{1}{2} f''$ term is the **Itô correction** — it arises because
Brownian motion has non-zero quadratic variation: $(dW_t)^2 = dt$. This has no
analogue in classical calculus.

### Step 3 — Apply Itô's Lemma to $f(S) = \ln S$

Compute the derivatives:

$$f'(S) = \frac{1}{S}, \qquad f''(S) = -\frac{1}{S^2}$$

Substituting into Itô's Lemma:

$$d(\ln S_t) = \frac{1}{S_t} \, dS_t - \frac{1}{2 S_t^2} \, (dS_t)^2$$

### Step 4 — Compute $(dS_t)^2$

Substitute $dS_t = r S_t \, dt + \sqrt{v_t} S_t \, dW_t^1$ and expand,
keeping only terms of order $dt$ using the Itô multiplication table
$(dt)^2 = 0$, $dt \cdot dW_t = 0$, $(dW_t)^2 = dt$:

$$\begin{aligned}
(dS_t)^2 &= \bigl(r S_t \, dt + \sqrt{v_t} S_t \, dW_t^1\bigr)^2 \\
          &= r^2 S_t^2 \underbrace{(dt)^2}_{0}
           + 2r S_t^2 \sqrt{v_t} \underbrace{dt \cdot dW_t^1}_{0}
           + v_t S_t^2 \underbrace{(dW_t^1)^2}_{dt} \\
          &= v_t S_t^2 \, dt
\end{aligned}$$

### Step 5 — Substitute back into Itô's Lemma

$$d(\ln S_t) = \frac{1}{S_t}\bigl(r S_t \, dt + \sqrt{v_t} S_t \, dW_t^1\bigr) - \frac{1}{2 S_t^2} \cdot v_t S_t^2 \, dt$$

$$= r \, dt + \sqrt{v_t} \, dW_t^1 - \frac{v_t}{2} \, dt$$

$$\boxed{d(\ln S_t) = \left(r - \frac{1}{2} v_t\right) dt + \sqrt{v_t} \, dW_t^1}$$

This is now a **linear SDE** in $\ln S_t$ — the coefficients no longer
depend on $S_t$ itself.

### Step 6 — Integrate over one time step $[t,\, t + \Delta t]$

Freeze $v_t$ at its current value (this is the Euler approximation on $v$):

$$\ln S_{t+\Delta t} - \ln S_t = \left(r - \frac{1}{2} v_t\right) \Delta t + \sqrt{v_t} \, \Delta W^1$$

where $\Delta W^1 = Z_1 \sqrt{\Delta t}$.

### Step 7 — Exponentiate to recover $S$

$$\boxed{S_{t+\Delta t} = S_t \cdot \exp\!\left[\left(r - \frac{1}{2} v_t\right) \Delta t + \sqrt{v_t} \, \Delta W^1\right]}$$

Since $\exp(\cdot) > 0$ always, **$S$ is guaranteed strictly positive**
on every path, no matter how large the noise. The $-\frac{1}{2}v$ term is the Itô
correction — if you omit it you introduce a systematic upward bias.

### Step 8 — Map to code

```rust
s *= ((option.r - 0.5 * v_pos) * dt + sqrt_v * dw1).exp();
//     └──── r − ½v ────┘  └─Δt─┘   └─ √v · ΔW¹ ─┘
//     ↑ drift correction             ↑ diffusion term
```

`s *=` is the multiplicative update — equivalent to $S_\text{new} = S_\text{old} \times \exp(\ldots)$.

---

## 4. Euler-Maruyama Discretisation of the Variance SDE

**Code line:**
```rust
v += heston.kappa * (heston.theta - v_pos) * dt + heston.sigma * sqrt_v * dw2;
```

### Step 1 — Recall the variance SDE

$$dv_t = \underbrace{\kappa(\theta - v_t)}_{\text{drift}} \, dt + \underbrace{\sigma \sqrt{v_t}}_{\text{diffusion}} \, dW_t^2$$

### Step 2 — General Euler-Maruyama formula

For any Itô SDE $dX_t = \mu(X_t) \, dt + \sigma(X_t) \, dW_t$, the
Euler-Maruyama scheme is the direct forward-Euler analogue:

$$X_{t+\Delta t} = X_t + \mu(X_t) \, \Delta t + \sigma(X_t) \, \Delta W_t$$

It achieves **strong convergence order 0.5** and **weak convergence order 1.0**.

### Step 3 — Identify drift and diffusion for the variance SDE

$$\mu(v_t) = \kappa(\theta - v_t), \qquad \sigma(v_t) = \sigma \sqrt{v_t}$$

### Step 4 — Write the discretised step

Substituting into the Euler-Maruyama formula:

$$\boxed{v_{t+\Delta t} = v_t + \kappa(\theta - v_t) \, \Delta t + \sigma \sqrt{v_t} \, \Delta W^2}$$

where $\Delta W^2 = \left(\rho Z_1 + \sqrt{1-\rho^2} \, Z_2\right)\sqrt{\Delta t}$
is the correlated Brownian increment from Section 2.

### Step 5 — The Full-Truncation Fix

The Euler step can push $v$ negative (especially when the Feller condition is
near-violated). A negative variance makes $\sqrt{v_t}$ imaginary, which breaks
the simulation.

The **full-truncation scheme** (Lord et al. 2010) clamps $v$ to zero before
every use on the right-hand side, while letting the raw value of $v$ evolve
freely (it may temporarily go negative but will be pulled back by mean reversion):

$$v_t^{+} = \max(v_t,\; 0)$$

Then use $v_t^{+}$ in both the drift and diffusion:

$$v_{t+\Delta t} = v_t + \kappa(\theta - v_t^{+}) \, \Delta t + \sigma \sqrt{v_t^{+}} \, \Delta W^2$$

Note the **update accumulates into the raw $v$** (left-hand side), not $v^+$.
This is intentional — if the raw $v$ is negative, the drift term
$\kappa(\theta - 0) = \kappa\theta > 0$ always pushes it back up toward $\theta$.

### Step 6 — Map to code

```rust
let v_pos  = v.max(0.0);        // v⁺ = max(v, 0)  — full truncation
let sqrt_v = v_pos.sqrt();      // √v⁺

v += heston.kappa * (heston.theta - v_pos) * dt   // κ(θ − v⁺)·Δt  ← drift
   + heston.sigma * sqrt_v * dw2;                 // σ·√v⁺·ΔW²     ← diffusion
// Note: update goes into raw v, not v_pos
```

`v +=` is additive — the new variance equals the old plus the drift step plus
the diffusion shock. Unlike the price SDE we do **not** exponentiate here —
variance can in principle be zero (it just can't be imaginary), so an additive
step is appropriate.
