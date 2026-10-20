"""
threshold_functions.py

Participation threshold functions τ(e). A bidder of type (e, v) participates
iff v >= τ(e). The threshold is common to all bidders.

A threshold function is any vectorized callable mapping an array of
externalities to an array of thresholds. The value +inf excludes the
externality (no valuation participates), matching the threshold v̄⁺ of the
paper; -inf admits every valuation.

The helpers below build threshold functions from the sources used in the code
base: polynomial coefficients (the genetic algorithm output) and tabulated
values on a grid of externalities (any other source).
"""

import numpy as np


class _Threshold:
    """Base of the threshold functions below: plain classes (not closures), so that
    they can be sent to worker processes."""

    description = "custom threshold function"


class _Polynomial(_Threshold):
    def __init__(self, coefficients):
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.description = "polynomial " + str(list(np.round(self.coefficients, 6)))

    def __call__(self, externalities):
        externalities = np.asarray(externalities, dtype=float)
        # Horner's rule, highest degree first.
        values = np.zeros_like(externalities)
        for coefficient in self.coefficients[::-1]:
            values = values * externalities + coefficient
        return values


class _Tabulated(_Threshold):
    def __init__(self, externality_knots, threshold_values):
        knots = np.asarray(externality_knots, dtype=float)
        values = np.asarray(threshold_values, dtype=float)
        order = np.argsort(knots)
        self.knots, self.values = knots[order], values[order]
        self.finite = np.isfinite(self.values)
        self.description = f"tabulated on {len(knots)} knots"

    def __call__(self, externalities):
        externalities = np.asarray(externalities, dtype=float)
        knots, values, finite = self.knots, self.values, self.finite
        if finite.all():
            return np.interp(externalities, knots, values)
        # Interpolate finite values; infinite knots govern their nearest region.
        nearest = np.abs(externalities[..., None] - knots).argmin(axis=-1)
        interpolated = np.interp(externalities, knots[finite], values[finite])
        return np.where(finite[nearest], interpolated, values[nearest])


class _Constant(_Threshold):
    def __init__(self, value):
        self.value = float(value)
        self.description = f"constant {value}"

    def __call__(self, externalities):
        return np.full(np.shape(externalities), self.value)


class Rescaled(_Threshold):
    """τ(e) = base(e · ratio)."""

    def __init__(self, base, ratio, start_name=None):
        self.base, self.ratio = base, ratio
        if start_name:
            self.start_name = start_name

    def __call__(self, externalities):
        return self.base(np.asarray(externalities, dtype=float) * self.ratio)


def polynomial_threshold(coefficients):
    """τ(e) = Σ_m coefficients[m] e^m (coefficients in increasing degree)."""
    return _Polynomial(coefficients)


def tabulated_threshold(externality_knots, threshold_values):
    """Piecewise-linear interpolation of thresholds given at externality knots.

    Externalities outside the knots take the nearest end value. Infinite
    threshold values are allowed at knots; between an infinite and a finite
    knot the threshold is infinite on the half interval nearer the infinite knot.
    """
    return _Tabulated(externality_knots, threshold_values)


def constant_threshold(value):
    """τ(e) = value for every externality (value = -inf admits everybody)."""
    return _Constant(value)


def as_threshold_function(threshold_source):
    """Convert a callable, a coefficient list, or a number into a threshold function."""
    if callable(threshold_source):
        return threshold_source
    if np.isscalar(threshold_source):
        return constant_threshold(threshold_source)
    return polynomial_threshold(threshold_source)


def describe_threshold(threshold_function) -> str:
    return getattr(threshold_function, "description", "custom threshold function")
