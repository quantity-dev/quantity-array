"""
    pint_array
    ~~~~~~~~~~

    Pint interoperability with array API standard arrays.
"""

from __future__ import annotations

from .quantity import ArrayUnitQuantity

__all__ = ["pint_namespace", "ArrayUnitQuantity"]

import types


def pint_namespace(xp):

    mod = types.ModuleType(f'pint({xp.__name__})')

    def asarray(obj, /, *, units=None, dtype=None, device=None, copy=None):
        if device is not None:
            raise NotImplementedError("`device` argument is not implemented")

        magnitude = getattr(obj, 'magnitude', obj)
        magnitude = xp.asarray(magnitude, dtype=dtype, device=device, copy=copy)

        units = getattr(obj, 'units', None) if units is None else units

        return ArrayUnitQuantity(magnitude, units)
    mod.asarray = asarray

    def sum(x, /, *, axis=None, dtype=None, keepdims=False):
        x = asarray(x)
        magnitude = xp.asarray(x.magnitude, copy=True)
        units = x.units
        magnitude = xp.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
        units = (1 * units + 1 * units).units
        return ArrayUnitQuantity(magnitude, units)
    mod.sum = sum

    def var(x, /, *, axis=None, correction=0.0, keepdims=False):
        x = asarray(x)
        magnitude = xp.asarray(x.magnitude, copy=True)
        units = x.units
        magnitude = xp.var(x, axis=axis, correction=correction, keepdims=keepdims)
        units = ((1 * units + 1 * units) ** 2).units
        return ArrayUnitQuantity(magnitude, units)
    mod.var = var

    return mod
