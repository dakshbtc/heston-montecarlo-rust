/// Heston stochastic volatility model parameters.
///
/// Describes the joint dynamics of asset price and variance:
///   dS =  r·S·dt  +  √v·S·dW₁
///   dv = κ·(θ − v)·dt  +  σ·√v·dW₂
///   dW₁·dW₂ = ρ·dt
#[derive(Clone, Copy)]
pub struct HestonParams {
    /// Mean reversion speed — how fast variance pulls back to `theta`.
    pub kappa: f64,
    /// Long-run variance — the level `v` reverts toward.
    pub theta: f64,
    /// Vol of vol — noise amplitude of the variance process.
    pub sigma: f64,
    /// Correlation between price and variance shocks (leverage effect).
    pub rho: f64,
    /// Initial variance (σ² at t = 0).
    pub v0: f64,
}
