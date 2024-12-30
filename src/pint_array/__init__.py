"""
    pint_array
    ~~~~~~~~~~

    Pint interoperability with array API standard arrays.
"""

from __future__ import annotations

from typing import Generic
import types
import textwrap

from pint.facets.plain import MagnitudeT, PlainQuantity
from pint import Quantity

__version__ = "0.0.1.dev0"
__all__ = ["pint_namespace", "__version__"]


def pint_namespace(xp):

    mod = types.ModuleType(f'pint({xp.__name__})')

    class ArrayQuantity(Generic[MagnitudeT], PlainQuantity[MagnitudeT]):
        def __init__(self, *args, **kwargs):
            super().__init__()
            magnitude = xp.asarray(self._magnitude)
            self._magnitude = magnitude
            self._dtype = magnitude.dtype
            self._device = magnitude.device
            self._ndim = magnitude.ndim
            self._shape = magnitude.shape
            self._size = magnitude.size

        __array_priority__ = 1  # make reflected operators work with NumPy

        @property
        def dtype(self):
            return self._dtype

        @property
        def device(self):
            return self._device

        @property
        def ndim(self):
            return self._ndim

        @property
        def shape(self):
            return self._shape

        @property
        def size(self):
            return self._size

        def __array_namespace__(self, api_version=None):
            if api_version is None or api_version == '2023.12':
                return mod
            else:
                raise NotImplementedError()
            
        def _call_super_method(self, method_name, *args, **kwargs):
            method = getattr(self.magnitude, method_name)
            args = [getattr(arg, 'magnitude', arg) for arg in args]
            return method(*args, **kwargs)

        ## Indexing ##
        # def __getitem__(self, key):
        #     if hasattr(key, 'mask') and xp.any(key.mask):
        #         message = ("Correct behavior for indexing with a masked array is "
        #                    "ambiguous, and no convention is supported at this time.")
        #         raise NotImplementedError(message)
        #     elif hasattr(key, 'mask'):
        #         key = key.data
        #     return MArray(self.data[key], self.mask[key])

        # def __setitem__(self, key, other):
        #     if hasattr(key, 'mask') and xp.any(key.mask):
        #         message = ("Correct behavior for indexing with a masked array is "
        #                    "ambiguous, and no convention is supported at this time.")
        #         raise NotImplementedError(message)
        #     elif hasattr(key, 'mask'):
        #         key = key.data
        #     self.mask[key] = getattr(other, 'mask', False)
        #     return self.data.__setitem__(key, getattr(other, 'data', other))


        ## Visualization ##
        def __repr__(self):
            return (
                f"<Quantity(\n"
                f"{textwrap.indent(repr(self._magnitude), '  ')},\n"
                f"  '{self.units}'\n)>"
            )

        # ## Linear Algebra Methods ##
        # def __matmul__(self, other):
        #     return mod.matmul(self, other)

        # def __imatmul__(self, other):
        #     res = mod.matmul(self, other)
        #     self.data[...] = res.data[...]
        #     self.mask[...] = res.mask[...]
        #     return

        # def __rmatmul__(self, other):
        #     other = MArray(other)
        #     return mod.matmul(other, self)
            
        ## Attributes ##

        @property
        def T(self):
            return ArrayUnitQuantity(self.magnitude.T, self.units)

        @property
        def mT(self):
            return ArrayUnitQuantity(self.magnitude.mT, self.units)

        # dlpack
        def __dlpack_device__(self):
            return self.magnitude.__dlpack_device__()

        def __dlpack__(self):
            # really not sure how to define this
            return self.magnitude.__dlpack__()

        def to_device(self, device, /, *, stream=None):
            _magnitude = self._magnitude.to_device(device, stream=stream)
            return ArrayUnitQuantity(_magnitude, self.units)

    class ArrayUnitQuantity(ArrayQuantity, Quantity):
        pass


    ## Methods ##

    # Methods that return the result of a unary operation as an array
    unary_names = (
        ['__abs__', '__floordiv__', '__invert__', '__neg__', '__pos__', '__ceil__']
    )
    for name in unary_names:
        def fun(self, name=name):
            return ArrayUnitQuantity(self._call_super_method(name), self.units)
        setattr(ArrayQuantity, name, fun)

    # Methods that return the result of a unary operation as a Python scalar
    unary_names_py = ['__bool__', '__complex__', '__float__', '__index__', '__int__']
    for name in unary_names_py:
        def fun(self, name=name):
            return self._call_super_method(name)
        setattr(ArrayQuantity, name, fun)

    # # Methods that return the result of an elementwise binary operation
    # binary_names = ['__add__', '__sub__', '__and__', '__eq__', '__ge__', '__gt__',
    #                 '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__',
    #                 '__or__', '__pow__', '__rshift__', '__sub__', '__truediv__',
    #                 '__xor__'] + ['__divmod__', '__floordiv__']
    # # Methods that return the result of an elementwise binary operation (reflected)
    # rbinary_names = ['__radd__', '__rand__', '__rdivmod__', '__rfloordiv__',
    #                 '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__rpow__',
    #                 '__rrshift__', '__rsub__', '__rtruediv__', '__rxor__']
    # for name in binary_names + rbinary_names:
    #     def fun(self, other, name=name):
    #         mask = (self.mask | other.mask) if hasattr(other, 'mask') else self.mask
    #         data = self._call_super_method(name, other)
    #         return ArrayUnitQuantity(data, mask)
    #     setattr(ArrayQuantity, name, fun)

    # # In-place methods
    # desired_names = ['__iadd__', '__iand__', '__ifloordiv__', '__ilshift__',
    #                 '__imod__', '__imul__', '__ior__', '__ipow__', '__irshift__',
    #                 '__isub__', '__itruediv__', '__ixor__']
    # for name in desired_names:
    #     def fun(self, other, name=name, **kwargs):
    #         if hasattr(other, 'mask'):
    #             # self.mask |= other.mask doesn't work because mask has no setter
    #             self.mask.__ior__(other.mask)
    #         self._call_super_method(name, other)
    #         return self
    #     setattr(ArrayQuantity, name, fun)

    def asarray(obj, /, *, units=None, dtype=None, device=None, copy=None):
        if device is not None:
            raise NotImplementedError("`device` argument is not implemented")

        magnitude = getattr(obj, 'magnitude', obj)
        magnitude = xp.asarray(magnitude, dtype=dtype, device=device, copy=copy)

        units = getattr(obj, 'units', None) if units is None else units

        return ArrayUnitQuantity(magnitude, units)
    mod.asarray = asarray

    ## Data Type Functions and Data Types ##
    dtype_fun_names = ['can_cast', 'finfo', 'iinfo', 'isdtype']
    dtype_names = ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                   'uint32', 'uint64', 'float32', 'float64', 'complex64', 'complex128']
    inspection_fun_names = ['__array_namespace_info__']
    version_attribute_names = ['__array_api_version__']
    for name in (
        dtype_fun_names + dtype_names + inspection_fun_names + version_attribute_names
    ):
        setattr(mod, name, getattr(xp, name))

    def astype(x, dtype, /, *, copy=True, device=None):
        if device is None and not copy and dtype == x.dtype:
            return x
        x = asarray(x)
        magnitude = xp.astype(x.magnitude, dtype, copy=copy, device=device)
        return ArrayUnitQuantity(magnitude, x.units)
    mod.astype = astype

    # Handle functions that ignore units on input and output
    for func_str in (
        "ones_like",
        "zeros_like",
        "empty_like",
        "argsort",
        "argmin",
        "argmax",
        "nonzero",
    ):
        def func(x, /, *args, func_str=func_str, **kwargs):
            x = asarray(x)
            magnitude = xp.asarray(x.magnitude, copy=True)
            xp_func = getattr(xp, func_str)
            magnitude = xp_func(x, *args, **kwargs)
            return ArrayUnitQuantity(magnitude, None)
        setattr(mod, func_str, func)

    # Handle functions with output unit defined by operation

    # output_unit="sum":
    # `x.units`, unless non-multiplicative, which raises `OffsetUnitCalculusError`
    for func_str in (
        "std",
        "cumulative_sum",
        "sum",
    ):
        def func(x, /, *args, func_str=func_str, **kwargs):
            x = asarray(x)
            magnitude = xp.asarray(x.magnitude, copy=True)
            units = x.units
            xp_func = getattr(xp, func_str)
            magnitude = xp_func(x, *args, **kwargs)
            units = (1 * units + 1 * units).units
            return ArrayUnitQuantity(magnitude, units)
        setattr(mod, func_str, func)

    # output_unit="variance":
    # square of `x.units`,
    # unless non-multiplicative, which raises `OffsetUnitCalculusError`
    def var(x, /, *, axis=None, correction=0.0, keepdims=False):
        x = asarray(x)
        magnitude = xp.asarray(x.magnitude, copy=True)
        units = x.units
        magnitude = xp.var(x, axis=axis, correction=correction, keepdims=keepdims)
        units = ((1 * units + 1 * units) ** 2).units
        return ArrayUnitQuantity(magnitude, units)
    mod.var = var

    #  "mul": product of all units in `all_args`
    # - "delta": `first_input_units`, unless non-multiplicative, which uses delta version
    # - "delta,div": like "delta", but divided by all units in `all_args` except the first
    # - "div": unit of first argument in `all_args` (or dimensionless if not a Quantity) divided
    #   by all following units
    # - "square": square of `first_input_units`
    # - "sqrt": square root of `first_input_units`
    # - "reciprocal": reciprocal of `first_input_units`
    # - "size": `first_input_units` raised to the power of `size`
    # - "invdiv": inverse of `div`, product of all following units divided by first argument unit

    return mod
