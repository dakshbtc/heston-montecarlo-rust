use rand_distr::{Distribution, StandardNormal};

use crate::models::HestonParams;
use crate::option::OptionParams;

/// Monte Carlo simulation settings.
pub struct McConfig {
    /// Number of independent price paths.
    pub n_paths: usize,
    /// Number of time steps per path.
    pub n_steps: usize,
}

/// Simulate one Heston path and return the terminal stock price.
///
/// Uses the log-Euler scheme for `S` (prevents negative prices) and the
/// full-truncation scheme for `v` (clamps variance to zero before each step).
///
/// Correlated Brownian increments are constructed via Cholesky decomposition:
///   dW₁ = z₁·√dt
///   dW₂ = (ρ·z₁ + √(1−ρ²)·z₂)·√dt,  z₁,z₂ ~ N(0,1) i.i.d.
pub fn simulate_path(
    heston: &HestonParams,
    option: &OptionParams,
    config: &McConfig,
    rng: &mut impl rand::Rng,
) -> f64 {
    let dt = option.t / config.n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let rho_perp = (1.0 - heston.rho * heston.rho).sqrt();

    let mut s = option.s0;
    let mut v = heston.v0;

    for _ in 0..config.n_steps {
        let z1: f64 = StandardNormal.sample(rng);
        let z2: f64 = StandardNormal.sample(rng);

        let dw1 = z1 * sqrt_dt;
        let dw2 = (heston.rho * z1 + rho_perp * z2) * sqrt_dt;

        let v_pos = v.max(0.0);
        let sqrt_v = v_pos.sqrt();

        s *= ((option.r - 0.5 * v_pos) * dt + sqrt_v * dw1).exp();
        v += heston.kappa * (heston.theta - v_pos) * dt + heston.sigma * sqrt_v * dw2;
    }

    s
}

/// Simulate one path and record `S` at every time step.
///
/// Returns a `Vec<f64>` of length `n_steps + 1`, starting with `S₀`.
/// Used by the visualisation layer to draw sample paths.
pub fn simulate_path_record(
    heston: &HestonParams,
    option: &OptionParams,
    config: &McConfig,
    rng: &mut impl rand::Rng,
) -> Vec<f64> {
    let dt = option.t / config.n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let rho_perp = (1.0 - heston.rho * heston.rho).sqrt();

    let mut s = option.s0;
    let mut v = heston.v0;
    let mut path = Vec::with_capacity(config.n_steps + 1);
    path.push(s);

    for _ in 0..config.n_steps {
        let z1: f64 = StandardNormal.sample(rng);
        let z2: f64 = StandardNormal.sample(rng);
        let dw1 = z1 * sqrt_dt;
        let dw2 = (heston.rho * z1 + rho_perp * z2) * sqrt_dt;
        let v_pos = v.max(0.0);
        let sqrt_v = v_pos.sqrt();
        s *= ((option.r - 0.5 * v_pos) * dt + sqrt_v * dw1).exp();
        v += heston.kappa * (heston.theta - v_pos) * dt + heston.sigma * sqrt_v * dw2;
        path.push(s);
    }

    path
}

/// Simulate a path and its antithetic counterpart in one pass.
///
/// The antithetic path reuses the same normals negated (`-z₁`, `-z₂`).
/// Averaging the two payoffs gives a lower-variance estimator, equivalent
/// to running twice as many paths for roughly the same compute cost.
///
/// Returns `(s_normal, s_antithetic)`.
pub fn simulate_path_antithetic(
    heston: &HestonParams,
    option: &OptionParams,
    config: &McConfig,
    rng: &mut impl rand::Rng,
) -> (f64, f64) {
    let dt = option.t / config.n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let rho_perp = (1.0 - heston.rho * heston.rho).sqrt();

    let mut s = option.s0;
    let mut v = heston.v0;
    let mut s_anti = option.s0;
    let mut v_anti = heston.v0;

    for _ in 0..config.n_steps {
        let z1: f64 = StandardNormal.sample(rng);
        let z2: f64 = StandardNormal.sample(rng);

        // Normal path
        {
            let dw1 = z1 * sqrt_dt;
            let dw2: f64 = (heston.rho * z1 + rho_perp * z2) * sqrt_dt;
            let v_pos = v.max(0.0);
            let sqrt_v = v_pos.sqrt();
            s *= ((option.r - 0.5 * v_pos) * dt + sqrt_v * dw1).exp();
            v += heston.kappa * (heston.theta - v_pos) * dt + heston.sigma * sqrt_v * dw2;
        }

        // Antithetic path: negate both normals
        {
            let dw1 = -z1 * sqrt_dt;
            let dw2 = (heston.rho * (-z1) + rho_perp * (-z2)) * sqrt_dt;
            let v_pos = v_anti.max(0.0);
            let sqrt_v = v_pos.sqrt();
            s_anti *= ((option.r - 0.5 * v_pos) * dt + sqrt_v * dw1).exp();
            v_anti += heston.kappa * (heston.theta - v_pos) * dt + heston.sigma * sqrt_v * dw2;
        }
    }

    (s, s_anti)
}
