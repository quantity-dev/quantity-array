"""
pint_array
~~~~~~~~~~

Pint interoperability with array API standard arrays.
"""

import collections
import contextlib
import importlib
import inspect
import sys
import textwrap
import types
from typing import Generic

from array_api_compat import size
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

        def _validate_key(self, key):
            if isinstance(key, tuple):
                return tuple(self._validate_key(key_i) for key_i in key)
            if hasattr(key, "units"):
                key = key.magnitude
            return key

        ## Indexing ##
        def __getitem__(self, key):
            key = self._validate_key(key)
            return ArrayUnitQuantity(self.magnitude[key], self.units)

        def __setitem__(self, key, other):
            key = self._validate_key(key)
            magnitude_other = (
                other.m_as(self.units) if hasattr(other, "units") else other
            )
            return self.magnitude.__setitem__(key, magnitude_other)

        def __iter__(self):
            return iter(self.magnitude)

        ## Visualization ##
        def __repr__(self):
            return (
                f"<Quantity(\n"
                f"{textwrap.indent(repr(self._magnitude), '  ')},\n"
                f"  '{self.units}'\n)>"
            )

        def __mul__(self, other):
            if hasattr(other, "units"):
                magnitude = self._call_super_method("__mul__", other.magnitude)
                units = self.units * other.units
            else:
                magnitude = self._call_super_method("__mul__", other)
                units = self.units
            return ArrayUnitQuantity(magnitude, units)

        def __gt__(self, other):
            if hasattr(other, "units"):
                magnitude = self._call_super_method("__gt__", other.magnitude)
            else:
                magnitude = self._call_super_method("__gt__", other)
            return ArrayUnitQuantity(magnitude, None)

        ## Linear Algebra Methods ##
        def __matmul__(self, other):
            return mod.matmul(self, other)

        def __imatmul__(self, other):
            res = mod.matmul(self, other)
            self.magnitude[...] = res.magnitude[...]
            self.units = res.units

        def __rmatmul__(self, other):
            return mod.matmul(other, self)

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

        def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):
            # really not sure how to define this
            return self.magnitude.__dlpack__(
                stream=stream, max_version=max_version, dl_device=dl_device, copy=copy
            )

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

    # Methods that return the result of an elementwise binary operation
    binary_names = [
        "__add__",
        "__sub__",
        "__and__",
        "__eq__",
        "__ge__",
        # "__gt__",
        "__le__",
        "__lshift__",
        "__lt__",
        "__mod__",
        # "__mul__",
        "__ne__",
        "__or__",
        "__pow__",
        "__rshift__",
        "__sub__",
        "__truediv__",
        "__xor__",
        "__divmod__",
        "__floordiv__",
    ]
    # Methods that return the result of an elementwise binary operation (reflected)
    rbinary_names = [
        "__radd__",
        "__rand__",
        "__rdivmod__",
        "__rfloordiv__",
        "__rlshift__",
        "__rmod__",
        "__rmul__",
        "__ror__",
        "__rpow__",
        "__rrshift__",
        "__rsub__",
        "__rtruediv__",
        "__rxor__",
    ]
    for name in binary_names + rbinary_names:

        def method(self, other, name=name):
            units = self.units
            magnitude_other = other.m_as(units) if hasattr(other, "units") else other
            magnitude = self._call_super_method(name, magnitude_other)
            # FIXME: correct units for op
            return ArrayUnitQuantity(magnitude, units)

        setattr(ArrayQuantity, name, method)

    # In-place methods
    desired_names = [
        "__iadd__",
        "__iand__",
        "__ifloordiv__",
        "__ilshift__",
        "__imod__",
        "__imul__",
        "__ior__",
        "__ipow__",
        "__irshift__",
        "__isub__",
        "__itruediv__",
        "__ixor__",
    ]
    for name in desired_names:

        def method(self, other, name=name):
            units = self.units
            magnitude_other = other.m_as(units) if hasattr(other, "units") else other
            magnitude = self._call_super_method(name, magnitude_other)
            # FIXME: correct units for op
            return ArrayUnitQuantity(magnitude, units)

        setattr(ArrayQuantity, name, method)

    ## Constants ##
    constant_names = ["e", "inf", "nan", "newaxis", "pi"]
    for name in constant_names:
        setattr(mod, name, getattr(xp, name))

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

    ## Manipulation Functions ##
    first_arg_arrays = {"broadcast_arrays", "concat", "stack", "meshgrid"}
    output_arrays = {"broadcast_arrays", "unstack", "meshgrid"}
    arbitrary_num_arrays = {"broadcast_arrays", "meshgrid"}

    def get_manip_fun(func_str):
        def manip_fun(x, *args, **kwargs):
            xp_func = getattr(xp, func_str)

            one_array = False
            if func_str not in first_arg_arrays:
                x = asarray(x)
                magnitude = xp.asarray(x.magnitude, copy=True)
                units = x.units
            elif hasattr(x, "__array_namespace__"):
                x = asarray(x)
                magnitude = xp.asarray(x.magnitude, copy=True)
                units = None
                one_array = True
            else:
                x = [asarray(x_i) for x_i in x]
                if len(x) == 0:
                    magnitude = xp.empty(0)
                    units = None
                else:
                    units = x[0].units
                    magnitude = [xp.asarray(x[0].magnitude, copy=True)]
                    for x_i in x[1:]:
                        magnitude.append(x_i.m_as(units))

            if func_str == "repeat" and hasattr(
                repeats := (args := list(args))[0], "units"
            ):
                args[0] = repeats.magnitude

            if func_str in arbitrary_num_arrays and not one_array:
                magnitude = xp_func(*magnitude, *args, **kwargs)
            else:
                magnitude = xp_func(magnitude, *args, **kwargs)

            if func_str in output_arrays:
                return tuple(
                    ArrayUnitQuantity(magnitude_i, units) for magnitude_i in magnitude
                )
            return ArrayUnitQuantity(magnitude, units)

        return manip_fun

    creation_manip_functions = ["tril", "triu", "meshgrid"]
    manip_names = [
        "broadcast_arrays",
        "broadcast_to",
        "concat",
        "expand_dims",
        "flip",
        "moveaxis",
        "permute_dims",
        "repeat",
        "reshape",
        "roll",
        "squeeze",
        "stack",
        "tile",
        "unstack",
    ]
    for name in manip_names + creation_manip_functions:
        setattr(mod, name, get_manip_fun(name))

    ## Data Type Functions and Data Types ##
    dtype_fun_names = ["can_cast", "finfo", "iinfo", "result_type"]
    for func_str in dtype_fun_names:

        def fun(*args, func_str=func_str, **kwargs):
            args = [(arg.magnitude if hasattr(arg, "units") else arg) for arg in args]
            return getattr(xp, func_str)(*args, **kwargs)

        setattr(mod, func_str, fun)

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
        "isdtype",  # function, but can treat as dtype here
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
        # https://github.com/data-apis/array-api-compat/issues/226
        # magnitude = xp.astype(x.magnitude, dtype, copy=copy, device=device)
        magnitude = xp.astype(x.magnitude, dtype, copy=copy)
        return ArrayUnitQuantity(magnitude, x.units)

    mod.astype = astype

    # Functions with output units equal to input units
    for func_str in (
        "max",
        "min",
        "mean",
        "ones_like",
        "zeros_like",
        "empty_like",
        "full_like",
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
        "argsort",
        "argmin",
        "argmax",
    ):

        def func(x, /, *args, func_str=func_str, **kwargs):
            x = asarray(x)
            magnitude = xp.asarray(x.magnitude, copy=True)
            xp_func = getattr(xp, func_str)
            magnitude = xp_func(magnitude, *args, **kwargs)
            return ArrayUnitQuantity(magnitude, None)

        setattr(mod, func_str, func)

    def nonzero(x, /):
        x = asarray(x)
        magnitude = xp.asarray(x.magnitude, copy=True)
        res = xp.nonzero(magnitude)
        return tuple(ArrayUnitQuantity(magnitude_i, None) for magnitude_i in res)

    mod.nonzero = nonzero

    def searchsorted(x1, x2, /, *, side="left", sorter=None):
        if sorter is not None:
            x1 = take(x1, sorter)

        magnitude_x1 = xp.asarray(x1.magnitude, copy=True)
        magnitude_x2 = x2.m_as(x1.units)

        magnitude = xp.searchsorted(magnitude_x1, magnitude_x2, side=side)
        return ArrayUnitQuantity(magnitude, None)

    mod.searchsorted = searchsorted

    # ignore units of condition, convert x2 to units of x1
    def where(condition, x1, x2, /):
        condition = asarray(condition)
        x1 = asarray(x1)
        x2 = asarray(x2)
        units = x1.units
        magnitude = xp.where(condition.magnitude, x1.magnitude, x2.m_as(units))
        return ArrayUnitQuantity(magnitude, units)

    mod.where = where

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
        "tan",
        "tanh",
        "trunc",
    ]
    for func_str in elementwise_one_array:

        def fun(x, /, *args, func_str=func_str, **kwargs):
            x = asarray(x)
            magnitude = xp.asarray(x.magnitude, copy=True)
            magnitude = getattr(xp, func_str)(magnitude, *args, **kwargs)
            return ArrayUnitQuantity(magnitude, x.units)

        setattr(mod, func_str, fun)
    
    for func_str in ["sqrt"]:

        def fun(x, /, *args, func_str=func_str, **kwargs):
            x = asarray(x)
            magnitude = xp.asarray(x.magnitude, copy=True)
            magnitude = getattr(xp, func_str)(magnitude, *args, **kwargs)
            return ArrayUnitQuantity(magnitude, x.units ** 0.5)

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

    def multiply(x1, x2, /, *args, **kwargs):
        x1 = asarray(x1)
        x2 = asarray(x2)

        units = x1.units * x2.units

        x1_magnitude = xp.asarray(x1.magnitude, copy=True)
        x2_magnitude = x2.m_as(x1.units)

        magnitude = xp.multiply(x1_magnitude, x2_magnitude, *args, **kwargs)
        return ArrayUnitQuantity(magnitude, units)

    mod.multiply = multiply

    ## Indexing Functions
    def take(x, indices, /, **kwargs):
        magnitude = xp.take(x.magnitude, indices.magnitude, **kwargs)
        return ArrayUnitQuantity(magnitude, x.units)

    mod.take = take

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

    ## Sorting Functions ##
    def get_sort_fun(func_str):
        def sort_fun(x, /, **kwargs):
            x = asarray(x)
            magnitude = xp.asarray(x.magnitude, copy=True)
            xp_func = getattr(xp, func_str)
            magnitude = xp_func(magnitude, **kwargs)
            units = x.units if func_str == "sort" else None
            return ArrayUnitQuantity(magnitude, units)

        return sort_fun

    sort_names = ["sort", "argsort"]
    for name in sort_names:
        setattr(mod, name, get_sort_fun(name))

    ## Set Functions ##
    def get_set_fun(func_str):
        def set_fun(x, /):
            x = asarray(x)
            units = x.units
            magnitude = xp.asarray(x.magnitude, copy=True)

            xp_func = getattr(xp, func_str)
            res = xp_func(magnitude)
            if func_str == "unique_values":
                return ArrayUnitQuantity(res, units)

            fields = res._fields
            name_tuple = res.__class__.__name__
            result_class = collections.namedtuple(name_tuple, fields)

            result_list = []
            for res_i, field_i in zip(res, fields, strict=False):
                units_i = units if field_i == "values" else None
                result_list.append(ArrayUnitQuantity(res_i, units_i))
            return result_class(*result_list)

        return set_fun

    unique_names = ["unique_values", "unique_counts", "unique_inverse", "unique_all"]
    for name in unique_names:
        setattr(mod, name, get_set_fun(name))

    # Handle functions with output unit defined by operation

    # output_unit="sum":
    # `x.units`, unless non-multiplicative, which raises `OffsetUnitCalculusError`
    for func_str in (
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

    # output_unit="delta"
    def std(x, /, **kwargs):
        x = asarray(x)
        magnitude = xp.asarray(x.magnitude, copy=True)
        units = x.units
        magnitude = xp.std(magnitude, **kwargs)
        units = (1 * units - 1 * units).units
        return ArrayUnitQuantity(magnitude, units)

    mod.std = std

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
    # square of delta `x.units`,
    # unless non-multiplicative, which raises `OffsetUnitCalculusError`
    def var(x, /, *args, **kwargs):
        x = asarray(x)
        magnitude = xp.asarray(x.magnitude, copy=True)
        units = x.units
        magnitude = xp.var(magnitude, *args, **kwargs)
        units = ((1 * units - 1 * units) ** 2).units
        return ArrayUnitQuantity(magnitude, units)

    mod.var = var

    # Output unit is the product of the input unit with itself along axis,
    # or the input unit to the power of the size of the array for axis=None
    def prod(x, /, *args, axis=None, **kwargs):
        x = asarray(x)
        magnitude = xp.asarray(x.magnitude, copy=True)
        exponent = magnitude.shape[axis] if axis is not None else size(magnitude)
        units = x.units**exponent
        magnitude = xp.prod(magnitude, *args, axis=axis, **kwargs)
        return ArrayUnitQuantity(magnitude, units)

    mod.prod = prod

    def clip(x, /, min=None, max=None):
        x = asarray(x)
        units = x.units
        magnitude_x = xp.asarray(x.magnitude, copy=True)
        magnitude_min = min.m_as(units) if hasattr(min, "units") else min
        magnitude_max = max.m_as(units) if hasattr(max, "units") else max
        magnitude = xp.clip(magnitude_x, magnitude_min, magnitude_max)
        return ArrayUnitQuantity(magnitude, units)

    mod.clip = clip

    preface = [
        "The following is the documentation for the corresponding "
        f"attribute of `{xp.__name__}`.",
        "The behavior on pint-wrapped arrays is the same for dimensionless "
        "quantities, and may differ for quantities with units.\n\n",
    ]
    preface = "\n".join(preface)
    for attribute in mod.__dict__:
        # Add documentation if it is not already present
        if getattr(mod, attribute).__doc__:
            continue

        xp_attr = getattr(xp, attribute, None)
        mod_attr = getattr(mod, attribute, None)
        if xp_attr is not None and mod_attr is not None:
            if hasattr(xp_attr, "__doc__"):
                with contextlib.suppress(AttributeError, TypeError):
                    xp_doc = xp_attr.__doc__
                    getattr(mod, attribute).__doc__ = preface + xp_doc

            with contextlib.suppress(ValueError, TypeError):
                mod_attr.__signature__ = inspect.signature(xp_attr)

            with contextlib.suppress(AttributeError, TypeError):
                mod_attr.__name__ = xp_attr.__name__

    return mod
