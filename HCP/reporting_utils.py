"""Result-neutral numerical helpers for reports; never fit or select models."""
import math


def safe_gain_ratio(rgic_gain, comparator_gain):
    """Return NA for an undefined or nonpositive comparator improvement."""
    try:
        numerator, denominator = float(rgic_gain), float(comparator_gain)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return None
    ratio = numerator / denominator
    return ratio if math.isfinite(ratio) else None


def full_axis_limits(values, padding=.08, include_zero=True):
    """Cover every finite supplied point/error endpoint, including zero."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        raise ValueError('No finite values are available for this plot.')
    if include_zero:
        finite.append(0.)
    low, high = min(finite), max(finite)
    span = high - low
    if span == 0:
        span = max(1., abs(low))
    return low - padding * span, high + padding * span
