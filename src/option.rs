/// European option contract specification.
#[derive(Clone, Copy)]
pub struct OptionParams {
    /// Initial stock price.
    pub s0: f64,
    /// Strike price.
    pub strike: f64,
    /// Continuously compounded risk-free rate.
    pub r: f64,
    /// Time to expiry in years.
    pub t: f64,
    /// `true` for a call, `false` for a put.
    pub is_call: bool,
}
