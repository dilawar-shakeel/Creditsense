"""
Statistical helpers for the CreditSense test suite.

These compute the EXACT theoretical mean of a distribution after either an ordinary
clip or a proper truncation, via numerical integration — independent of the generator's
own implementation, so comparing an empirical mean against these values is a genuine
correctness check, not a circular one.
"""

from scipy import integrate


def theoretical_mean_clipped(dist, lo: float, hi: float, **params) -> float:
    """E[clip(X, lo, hi)] = lo*P(X<lo) + hi*P(X>hi) + integral_{lo}^{hi} x f(x) dx."""
    integral, _ = integrate.quad(lambda x: x * dist.pdf(x, **params), lo, hi)
    mass_below = dist.cdf(lo, **params)
    mass_above = 1.0 - dist.cdf(hi, **params)
    return lo * mass_below + hi * mass_above + integral


def theoretical_mean_truncated(dist, lo: float, hi: float, **params) -> float:
    """E[X | lo <= X <= hi] = integral_{lo}^{hi} x f(x) dx / P(lo <= X <= hi)."""
    integral, _ = integrate.quad(lambda x: x * dist.pdf(x, **params), lo, hi)
    mass_inside = dist.cdf(hi, **params) - dist.cdf(lo, **params)
    return integral / mass_inside


def theoretical_median_truncated(dist, lo: float, hi: float, **params) -> float:
    """Median of X | lo <= X <= hi, found by inverting the truncated CDF at u=0.5."""
    lo_cdf = dist.cdf(lo, **params)
    hi_cdf = dist.cdf(hi, **params)
    return dist.ppf(lo_cdf + 0.5 * (hi_cdf - lo_cdf), **params)
