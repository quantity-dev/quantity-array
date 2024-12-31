"""
pint_array
~~~~~~~~~~

Pint interoperability with array API standard arrays.
"""

import importlib
import sys
import textwrap
import types
from typing import Generic

from pint import Quantity
from pint.facets.plain import MagnitudeT, PlainQuantity

__version__ = "0.0.1.dev0"
__all__ = ["__version__", "pint_namespace"]


def __getattr__(name):
    try:
        xp = importlib.import_module(name)
        mod = pint_namespace(xp)
        sys.modules[f"marray.{name}"] = mod
        return mod
    except ModuleNotFoundError as e:
        raise AttributeError(str(e)) from None


def pint_namespace(xp):
    mod = types.ModuleType(f"pint({xp.__name__})")

    class ArrayQuantity(Generic[MagnitudeT], PlainQuantity[MagnitudeT]):
        def __init__(self, *args, **kwargs):  # noqa: ARG002
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
            if api_version is None or api_version == "2023.12":
                return mod
            raise NotImplementedError()

        def _call_super_method(self, method_name, *args, **kwargs):
            method = getattr(self.magnitude, method_name)
            args = [getattr(arg, "magnitude", arg) for arg in args]
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
    unary_names = [
        "__abs__",
        "__floordiv__",
        "__invert__",
        "__neg__",
        "__pos__",
        "__ceil__",
    ]
    for name in unary_names:

        def fun(self, name=name):
            return ArrayUnitQuantity(self._call_super_method(name), self.units)

        setattr(ArrayQuantity, name, fun)

    # Methods that return the result of a unary operation as a Python scalar
    unary_names_py = ["__bool__", "__complex__", "__float__", "__index__", "__int__"]
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
            msg = "`device` argument is not implemented"
            raise NotImplementedError(msg)

        magnitude = getattr(obj, "magnitude", obj)
        magnitude = xp.asarray(magnitude, dtype=dtype, device=device, copy=copy)

        units = getattr(obj, "units", None) if units is None else units

        return ArrayUnitQuantity(magnitude, units)

    mod.asarray = asarray

    creation_functions = [
        "arange",
        "empty",
        "eye",
        "from_dlpack",
        "full",
        "linspace",
        "ones",
        "zeros",
    ]
    for func_str in creation_functions:

        def fun(*args, func_str=func_str, units=None, **kwargs):
            magnitude = getattr(xp, func_str)(*args, **kwargs)
            return ArrayUnitQuantity(magnitude, units)

        setattr(mod, func_str, fun)

    ## Data Type Functions and Data Types ##
    dtype_fun_names = ["can_cast", "finfo", "iinfo", "isdtype"]
    dtype_names = [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ]
    inspection_fun_names = ["__array_namespace_info__"]
    version_attribute_names = ["__array_api_version__"]
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

    # Functions with output units equal to input units
    for func_str in (
        "max",
        "min",
        "mean",
    ):

        def func(x, /, *args, func_str=func_str, **kwargs):
            x = asarray(x)
            magnitude = xp.asarray(x.magnitude, copy=True)
            xp_func = getattr(xp, func_str)
            magnitude = xp_func(magnitude, *args, **kwargs)
            return ArrayUnitQuantity(magnitude, x.units)

        setattr(mod, func_str, func)

    # Functions which ignore units on input and output
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
            magnitude = xp_func(magnitude, *args, **kwargs)
            return ArrayUnitQuantity(magnitude, None)

        setattr(mod, func_str, func)

    # strip_unit_input_output_ufuncs = ["isnan", "isinf", "isfinite", "signbit", "sign"]
    # matching_input_bare_output_ufuncs = [
    #     "equal",
    #     "greater",
    #     "greater_equal",
    #     "less",
    #     "less_equal",
    #     "not_equal",
    # ]
    # matching_input_set_units_output_ufuncs = {"arctan2": "radian"}
    # set_units_ufuncs = {
    #     "cumprod": ("", ""),
    #     "arccos": ("", "radian"),
    #     "arcsin": ("", "radian"),
    #     "arctan": ("", "radian"),
    #     "arccosh": ("", "radian"),
    #     "arcsinh": ("", "radian"),
    #     "arctanh": ("", "radian"),
    #     "exp": ("", ""),
    #     "expm1": ("", ""),
    #     "exp2": ("", ""),
    #     "log": ("", ""),
    #     "log10": ("", ""),
    #     "log1p": ("", ""),
    #     "log2": ("", ""),
    #     "sin": ("radian", ""),
    #     "cos": ("radian", ""),
    #     "tan": ("radian", ""),
    #     "sinh": ("radian", ""),
    #     "cosh": ("radian", ""),
    #     "tanh": ("radian", ""),
    #     "radians": ("degree", "radian"),
    #     "degrees": ("radian", "degree"),
    #     "deg2rad": ("degree", "radian"),
    #     "rad2deg": ("radian", "degree"),
    #     "logaddexp": ("", ""),
    #     "logaddexp2": ("", ""),
    # }
    # # TODO (#905 follow-up):
    # #   while this matches previous behavior, some of these have optional arguments
    #     that
    # #   should not be Quantities. This should be fixed, and tests using these optional
    # #   arguments should be added.
    # matching_input_copy_units_output_ufuncs = [
    #     "compress",
    #     "conj",
    #     "conjugate",
    #     "copy",
    #     "diagonal",
    #     "max",
    #     "mean",
    #     "min",
    #     "ptp",
    #     "ravel",
    #     "repeat",
    #     "reshape",
    #     "round",
    #     "squeeze",
    #     "swapaxes",
    #     "take",
    #     "trace",
    #     "transpose",
    #     "roll",
    #     "ceil",
    #     "floor",
    #     "hypot",
    #     "rint",
    #     "copysign",
    #     "nextafter",
    #     "trunc",
    #     "absolute",
    #     "positive",
    #     "negative",
    #     "maximum",
    #     "minimum",
    #     "fabs",
    # ]
    # copy_units_output_ufuncs = ["ldexp", "fmod", "mod", "remainder"]
    # op_units_output_ufuncs = {
    #     "var": "square",
    #     "multiply": "mul",
    #     "true_divide": "div",
    #     "divide": "div",
    #     "floor_divide": "div",
    #     "sqrt": "sqrt",
    #     "cbrt": "cbrt",
    #     "square": "square",
    #     "reciprocal": "reciprocal",
    #     "std": "sum",
    #     "sum": "sum",
    #     "cumsum": "sum",
    #     "matmul": "mul",
    # }

    elementwise_one_array = [
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "bitwise_invert",
        "ceil",
        "conj",
        "cos",
        "cosh",
        "exp",
        "expm1",
        "floor",
        "imag",
        "isfinite",
        "isinf",
        "isnan",
        "log",
        "log1p",
        "log2",
        "log10",
        "logical_not",
        "negative",
        "positive",
        "real",
        "round",
        "sign",
        "signbit",
        "sin",
        "sinh",
        "square",
        "sqrt",
        "tan",
        "tanh",
        "trunc",
    ]
    for func_str in elementwise_one_array:

        def fun(x, /, *args, func_str=func_str, **kwargs):
            x = asarray(x)
            magnitude = xp.asarray(x.magnitude, copy=True)
            magnitude = getattr(xp, func_str)(x, *args, **kwargs)
            return ArrayUnitQuantity(magnitude, x.units)

        setattr(mod, func_str, fun)

    elementwise_two_arrays = [
        "add",
        "atan2",
        "bitwise_and",
        "bitwise_left_shift",
        "bitwise_or",
        "bitwise_right_shift",
        "bitwise_xor",
        "copysign",
        "divide",
        "equal",
        "floor_divide",
        "greater",
        "greater_equal",
        "hypot",
        "less",
        "less_equal",
        "logaddexp",
        "logical_and",
        "logical_or",
        "logical_xor",
        "maximum",
        "minimum",
        "multiply",
        "not_equal",
        "pow",
        "remainder",
        "subtract",
    ]
    for func_str in elementwise_two_arrays:

        def fun(x1, x2, /, *args, func_str=func_str, **kwargs):
            x1 = asarray(x1)
            x2 = asarray(x2)

            units = x1.units

            x1_magnitude = xp.asarray(x1.magnitude, copy=True)
            x2_magnitude = x2.m_as(units)

            xp_func = getattr(xp, func_str)
            magnitude = xp_func(x1_magnitude, x2_magnitude, *args, **kwargs)
            return ArrayUnitQuantity(magnitude, units)

        setattr(mod, func_str, fun)

    def get_linalg_fun(func_str):
        def linalg_fun(x1, x2, /, **kwargs):
            x1 = asarray(x1)
            x2 = asarray(x2)
            magnitude1 = xp.asarray(x1.magnitude, copy=True)
            magnitude2 = xp.asarray(x2.magnitude, copy=True)

            xp_func = getattr(xp, func_str)
            magnitude = xp_func(magnitude1, magnitude2, **kwargs)
            return ArrayUnitQuantity(magnitude, x1.units * x2.units)

        return linalg_fun

    linalg_names = ["matmul", "tensordot", "vecdot"]
    for name in linalg_names:
        setattr(mod, name, get_linalg_fun(name))

    def matrix_transpose(x):
        return x.mT

    mod.matrix_transpose = matrix_transpose

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
            magnitude = xp_func(magnitude, *args, **kwargs)
            units = (1 * units + 1 * units).units
            return ArrayUnitQuantity(magnitude, units)

        setattr(mod, func_str, func)

    for func_str in (
        "any",
        "all",
    ):

        def func(x, /, *args, func_str=func_str, **kwargs):
            x = asarray(x)
            magnitude = xp.asarray(x.magnitude, copy=True)
            if x._is_multiplicative:
                xp_func = getattr(xp, func_str)
                magnitude = xp_func(magnitude, *args, **kwargs)
                return ArrayUnitQuantity(magnitude, None)

            msg = "Boolean value of Quantity with offset unit is ambiguous."
            raise ValueError(msg)

        setattr(mod, func_str, func)

    # output_unit="variance":
    # square of `x.units`,
    # unless non-multiplicative, which raises `OffsetUnitCalculusError`
    def var(x, /, *args, **kwargs):
        x = asarray(x)
        magnitude = xp.asarray(x.magnitude, copy=True)
        units = x.units
        magnitude = xp.var(magnitude, *args, **kwargs)
        units = ((1 * units + 1 * units) ** 2).units
        return ArrayUnitQuantity(magnitude, units)

    mod.var = var

    # Output unit is the product of the input unit with itself along axis,
    # or the input unit to the power of the size of the array for axis=None
    def prod(x, /, *args, axis=None, **kwargs):
        x = asarray(x)
        magnitude = xp.asarray(x.magnitude, copy=True)
        exponent = magnitude.shape[axis] if axis is not None else magnitude.size
        units = x.units**exponent
        magnitude = xp.prod(magnitude, *args, axis=axis, **kwargs)
        return ArrayUnitQuantity(magnitude, units)

    mod.prod = prod

    return mod
