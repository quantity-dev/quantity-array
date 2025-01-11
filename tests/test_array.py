from __future__ import annotations

import copy
import operator as op

import array_api_strict as xp
import numpy as np
import pytest
from pint import DimensionalityError, OffsetUnitCalculusError
from pint.testsuite import helpers

import pint_array

pxp = pint_array.pint_namespace(xp)

class TestNumPyMethods:
    @classmethod
    def setup_class(cls):
        from pint import _DEFAULT_REGISTRY

        cls.ureg = _DEFAULT_REGISTRY
        cls.Q_ = pxp.ArrayUnitQuantity

    @classmethod
    def teardown_class(cls):
        cls.ureg = None
        cls.Q_ = None

    @property
    def q(self):
        return pxp.asarray([[1, 2], [3, 4]], units=self.ureg.m)

    @property
    def q_scalar(self):
        return pxp.asarray(5, units=self.ureg.m)

    @property
    def q_nan(self):
        return pxp.asarray([[1, 2], [3, pxp.nan]], units=self.ureg.m)

    @property
    def q_zero_or_nan(self):
        return pxp.asarray([[0, 0], [0, pxp.nan]], units=self.ureg.m)

    @property
    def q_temperature(self):
        return pxp.asarray([[1, 2], [3, 4]], units = self.ureg.degC)

    def assertNDArrayEqual(self, actual, desired):
        # Assert that the given arrays are equal, and are not Quantities
        np.testing.assert_array_equal(actual, desired)
        assert not isinstance(actual, self.Q_)
        assert not isinstance(desired, self.Q_)


class TestNumPyArrayCreation(TestNumPyMethods):
    # https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html

    @pytest.mark.xfail(reason="Scalar argument issue ")
    def test_ones_like(self):
        self.assertNDArrayEqual(pxp.ones_like(self.q), pxp.asarray([[1, 1], [1, 1]]))

    @pytest.mark.xfail(reason="should this be using NDArrayEqual?")
    def test_zeros_like(self):
        self.assertNDArrayEqual(pxp.zeros_like(self.q), pxp.asarray([[0, 0], [0, 0]]))

    def test_empty_like(self):
        ret = pxp.empty_like(self.q)
        assert ret.shape == (2, 2)
        assert ret.magnitude.__array_namespace__() is xp

    @pytest.mark.xfail(reason="Scalar argument issue ")
    def test_full_like(self):
        helpers.assert_quantity_equal(
            pxp.full_like(self.q, self.Q_(0, self.ureg.degC)),
            self.Q_([[0, 0], [0, 0]], self.ureg.degC),
        )
        self.assertNDArrayEqual(pxp.full_like(self.q, 2), pxp.asarray([[2, 2], [2, 2]]))


class TestNumPyArrayManipulation(TestNumPyMethods):
    # Changing array shape

    def test_reshape(self):
        helpers.assert_quantity_equal(
            pxp.reshape(self.q, [1, 4]), pxp.asarray([[1, 2, 3, 4]] ) * self.ureg.m
        )

    # Transpose-like operations

    def test_moveaxis(self):
        helpers.assert_quantity_equal(
            pxp.moveaxis(self.q, 1, 0), pxp.asarray([[1, 2], [3, 4]]).T * self.ureg.m
        )

    def test_transpose(self):
        helpers.assert_quantity_equal(
            pxp.matrix_transpose(self.q), pxp.asarray([[1, 3], [2, 4]]) * self.ureg.m
        )

    def test_flip_numpy_func(self):
        helpers.assert_quantity_equal(
            pxp.flip(self.q, axis=0), pxp.asarray([[3, 4], [1, 2]]) * self.ureg.m
        )

    # Changing number of dimensions

    def test_broadcast_to(self):
        helpers.assert_quantity_equal(
            pxp.broadcast_to(self.q[:, 1], (2, 2)),
            pxp.asarray([[2, 4], [2, 4]]) * self.ureg.m,
        )

    def test_expand_dims(self):
        helpers.assert_quantity_equal(
            pxp.expand_dims(self.q, axis=0), pxp.asarray([[[1, 2], [3, 4]]]) * self.ureg.m
        )

    def test_squeeze(self):
        helpers.assert_quantity_equal(
            pxp.squeeze(pxp.asarray([[[0], [1], [2]]]) *self.ureg.m, axis=0),
            pxp.asarray([0,1,2]) * self.ureg.m
        )

    # Changing number of dimensions
    # Joining arrays

    def test_concat_stack(self, subtests):
        for func in (pxp.concat, pxp.stack):
            with subtests.test(func=func):
                helpers.assert_quantity_equal(
                    func([self.q] * 2), pxp.asarray(func([self.q.m] * 2), units = "m")
                )
                # One or more of the args is a bare array full of zeros or NaNs
                # helpers.assert_quantity_equal(
                #     func([self.q_zero_or_nan.m, self.q]),
                #     self.Q_(func([self.q_zero_or_nan.m, self.q.m]), self.ureg.m),
                # )
                # One or more of the args is a bare array with at least one non-zero,
                # non-NaN element
                nz = self.q_zero_or_nan
                nz.m[0, 0] = 1
                with pytest.raises(DimensionalityError):
                    func([nz.m, self.q])

    def test_astype(self):
        dtype=pxp.float32
        actual = pxp.astype(self.q, dtype)
        expected = pxp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, units = "m")
        helpers.assert_quantity_equal(actual, expected)
        assert actual.m.dtype == expected.m.dtype

    def test_item(self):
        helpers.assert_quantity_equal(self.Q_([[0]], "m").item(), 0 * self.ureg.m)

    def test_broadcast_arrays(self):
        x = pxp.asarray([[1, 2, 3]], units= "m")
        y = pxp.asarray([[4], [5]], units= "nm")
        result = pxp.broadcast_arrays(x, y)
        expected = (
            pxp.asarray([[1, 2, 3], [1, 2, 3]], units= "m"),
            pxp.asarray([[4, 4, 4], [5, 5, 5]], units= "nm")
        )
        helpers.assert_quantity_equal(result, expected)

    def test_roll(self):
        helpers.assert_quantity_equal(
            pxp.roll(self.q, 1), [[4, 1], [2, 3]] * self.ureg.m
        )


class TestNumPyMathematicalFunctions(TestNumPyMethods):
    # https://www.numpy.org/devdocs/reference/routines.math.html

    def test_prod_numpy_func(self):
        axis = 0

        helpers.assert_quantity_equal(pxp.prod(self.q), 24 * self.ureg.m**4)
        helpers.assert_quantity_equal(
            pxp.prod(self.q, axis=axis), [3, 8] * self.ureg.m**2
        )

    def test_sum_numpy_func(self):
        helpers.assert_quantity_equal(pxp.sum(self.q, axis=0), [4, 6] * self.ureg.m)
        with pytest.raises(OffsetUnitCalculusError):
            pxp.sum(self.q_temperature)

    # Arithmetic operations
    def test_addition_with_scalar(self):
        a = pxp.asarray([0, 1, 2], dtype = pxp.float32)
        b = 10.0 * self.ureg("gram/kilogram")
        helpers.assert_quantity_almost_equal(
            a + b, pxp.asarray([0.01, 1.01, 2.01], units = "")
        )
        helpers.assert_quantity_almost_equal(
            b + a, pxp.asarray([0.01, 1.01, 2.01], units = "")
        )

    def test_addition_with_incompatible_scalar(self):
        a = pxp.asarray([0, 1, 2])
        b = 1.0 * self.ureg.m
        with pytest.raises(DimensionalityError):
            op.add(a, b)
        with pytest.raises(DimensionalityError):
            op.add(b, a)

    def test_power(self):
        arr = xp.asarray(range(3), dtype=pxp.float32)
        q  = pxp.asarray(range(3), dtype=pxp.float32, units="meter")

        for op_ in [pxp.pow]:
            with pytest.raises(DimensionalityError):
                op_(2.0, q)
            with pytest.raises(DimensionalityError):
                op_(q, arr)
            with pytest.raises(DimensionalityError):
                op_(q, q)

        q = pxp.asarray(self.q, dtype=pxp.float32)
        helpers.assert_quantity_equal(
            pxp.pow(q, self.Q_(2.)), pxp.asarray([[1, 4], [9, 16]], dtype=pxp.float32, units= "m**2")
        )
        helpers.assert_quantity_equal(
            q ** self.Q_(2.), self.Q_([[1, 4], [9, 16]], "m**2")
        )
        self.assertNDArrayEqual(arr ** self.Q_(2.), xp.asarray([0, 1, 4]))

    def test_sqrt(self):
        q = self.Q_(100.0, "m**2")
        helpers.assert_quantity_equal(pxp.sqrt(q), self.Q_(10.0, "m"))

    @pytest.mark.xfail
    def test_exponentiation_array_exp_2(self):
        arr = pxp.asarray(range(3), dtype=float)
        # q = self.Q_(copy.copy(arr), None)
        q = self.Q_(copy.copy(arr), "meter")
        arr_cp = copy.copy(arr)
        q_cp = copy.copy(q)
        # this fails as expected since numpy 1.8.0 but...
        with pytest.raises(DimensionalityError):
            op.pow(arr_cp, q_cp)
        # ..not for op.ipow !
        # q_cp is treated as if it is an array. The units are ignored.
        # Quantity.__ipow__ is never called
        arr_cp = copy.copy(arr)
        q_cp = copy.copy(q)
        with pytest.raises(DimensionalityError):
            op.ipow(arr_cp, q_cp)


class TestNumPyUnclassified(TestNumPyMethods):
    def test_repeat(self):
        helpers.assert_quantity_equal(
            pxp.repeat(self.q, 2), [1, 1, 2, 2, 3, 3, 4, 4] * self.ureg.m
        )

    def test_sort_numpy_func(self):
        q = [4, 5, 2, 3, 1, 6] * self.ureg.m
        helpers.assert_quantity_equal(pxp.sort(q), [1, 2, 3, 4, 5, 6] * self.ureg.m)

    def test_argsort_numpy_func(self):
        self.assertNDArrayEqual(
            pxp.argsort(pxp.asarray(self.q), axis=0), xp.asarray([[0, 0], [1, 1]])
        )

    def test_searchsorted_numpy_func(self):
        """Test searchsorted as numpy function."""
        q = self.q.flatten()
        self.assertNDArrayEqual(pxp.searchsorted(q, pxp.asarray([1.5, 2.5], units="m")), xp.asarray([1, 2])
        )

    def test_nonzero_numpy_func(self):
        q = [1, 0, 5, 6, 0, 9] * self.ureg.m
        self.assertNDArrayEqual(pxp.nonzero(q)[0], [0, 2, 3, 5])

    def test_any_numpy_func(self):
        q = [0, 1] * self.ureg.m
        assert pxp.any(q)
        with pytest.raises(ValueError, match="offset unit is ambiguous"):
            pxp.any(self.q_temperature)

    def test_all_numpy_func(self):
        q = [0, 1] * self.ureg.m
        assert not pxp.all(q)
        with pytest.raises(ValueError, match="offset unit is ambiguous"):
            pxp.all(self.q_temperature)

    def test_max_numpy_func(self):
        assert pxp.max(self.q) == 4 * self.ureg.m

    def test_max_with_axis_arg(self):
        helpers.assert_quantity_equal(pxp.max(self.q, axis=1), [2, 4] * self.ureg.m)

    def test_argmax_numpy_func(self):
        self.assertNDArrayEqual(pxp.argmax(self.q, axis=0), xp.asarray([1, 1]))

    def test_maximum(self):
        helpers.assert_quantity_equal(
            pxp.maximum(self.q, self.Q_([0, 5], "m")), self.Q_([[1, 5], [3, 5]], "m")
        )

    def test_min_numpy_func(self):
        assert pxp.min(self.q) == 1 * self.ureg.m

    def test_min_with_axis_arg(self):
        helpers.assert_quantity_equal(pxp.min(self.q, axis=1), [1, 3] * self.ureg.m)

    def test_argmin_numpy_func(self):
        self.assertNDArrayEqual(pxp.argmin(self.q, axis=0), xp.asarray([0, 0]))

    def test_minimum(self):
        helpers.assert_quantity_equal(
            pxp.minimum(self.q, self.Q_([0, 5], "m")), self.Q_([[0, 2], [0, 4]], "m")
        )

    def test_clip_numpy_func(self):
        helpers.assert_quantity_equal(
            pxp.clip(pxp.asarray(self.q, dtype=pxp.float32), 150 * self.ureg.cm, None), [[1.5, 2], [3, 4]] * self.ureg.m
        )

    def test_round_numpy_func(self):
        helpers.assert_quantity_equal(
            pxp.round(102.75 * self.ureg.m, ), 103 * self.ureg.m
        )

    def test_cumulative_sum(self):
        helpers.assert_quantity_equal(
            pxp.cumulative_sum(self.q, axis=0), [[1, 2], [4, 6]] * self.ureg.m
        )

    def test_mean_numpy_func(self):
        assert pxp.mean(pxp.asarray(self.q, dtype=pxp.float32)) == 2.5 * self.ureg.m
        assert pxp.mean(pxp.asarray(self.q_temperature, dtype=pxp.float32)) == self.Q_(2.5, self.ureg.degC)

    def test_var_numpy_func(self):
        dtype=pxp.float32
        assert pxp.var(pxp.asarray(self.q, dtype=dtype)) == 1.25 * self.ureg.m**2
        assert pxp.var(pxp.asarray(self.q_temperature, dtype=dtype)) == 1.25 * self.ureg.delta_degC**2

    def test_std_numpy_func(self):
        dtype=pxp.float32
        helpers.assert_quantity_almost_equal(
            pxp.std(pxp.asarray(self.q, dtype=dtype)), 1.11803 * self.ureg.m, rtol=1e-5
        )
        helpers.assert_quantity_almost_equal(
            pxp.std(pxp.asarray(self.q_temperature, dtype=dtype)), 1.11803 * self.ureg.delta_degC, rtol=1e-5
        )

    def test_conj(self):
        arr = pxp.asarray(self.q, dtype = pxp.complex64)  * (1 + 1j)
        helpers.assert_quantity_equal(pxp.conj(arr), arr * (1 - 1j))
        # helpers.assert_quantity_equal(
        #     (self.q * (1 + 1j)).conjugate(), self.q * (1 - 1j)
        # )

    def test_getitem(self):
        with pytest.raises(IndexError):
            self.q.__getitem__((0, 10))
        helpers.assert_quantity_equal(self.q[0], [1, 2] * self.ureg.m)
        assert self.q[1, 1] == 4 * self.ureg.m

    def test_setitem(self):
        with pytest.raises(TypeError):
            self.q[0, 0] = 1
        with pytest.raises(DimensionalityError):
            self.q[0, 0] = 1 * self.ureg.J
        with pytest.raises(DimensionalityError):
            self.q[0] = 1
        with pytest.raises(DimensionalityError):
            self.q[0] = pxp.asarray([1, 2])
        with pytest.raises(DimensionalityError):
            self.q[0] = 1 * self.ureg.J

        q = self.q.copy()
        q[0] = 1 * self.ureg.m
        helpers.assert_quantity_equal(q, [[1, 1], [3, 4]] * self.ureg.m)

        q = self.q.copy()
        q[...] = 1 * self.ureg.m
        helpers.assert_quantity_equal(q, [[1, 1], [1, 1]] * self.ureg.m)

        q = self.q.copy()
        q[:] = 1 * self.ureg.m
        helpers.assert_quantity_equal(q, [[1, 1], [1, 1]] * self.ureg.m)

        # check and see that dimensionless numbers work correctly
        q = [0, 1, 2, 3] * self.ureg.dimensionless
        q[0] = 1
        helpers.assert_quantity_equal(q, pxp.asarray([1, 1, 2, 3]))
        q[0] = self.ureg.m / self.ureg.mm
        helpers.assert_quantity_equal(q, pxp.asarray([1000, 1, 2, 3]))

        q = [0.0, 1.0, 2.0, 3.0] * self.ureg.m / self.ureg.mm
        q[0] = 1.0
        helpers.assert_quantity_equal(q, [0.001, 1, 2, 3] * self.ureg.m / self.ureg.mm)

    def test_reversible_op(self):
        """ """
        q=pxp.asarray(self.q, dtype=pxp.float64)
        x = xp.asarray(self.q.magnitude, dtype=xp.float64)
        u = pxp.asarray(pxp.ones(x.shape), dtype=pxp.float64)
        helpers.assert_quantity_equal(x / q, u * x / q)
        helpers.assert_quantity_equal(x * q, u * x * q)
        helpers.assert_quantity_equal(x + u, u + x)
        helpers.assert_quantity_equal(x - u, -(u - x))

    def test_equal(self):
        x = self.q.magnitude
        u = pxp.ones(x.shape)
        false = xp.zeros_like(x, dtype=xp.bool)

        helpers.assert_quantity_equal(u, u)
        helpers.assert_quantity_equal(u, u.magnitude)
        helpers.assert_quantity_equal(u == 1, u.magnitude == 1)

        v = pxp.asarray((pxp.zeros(x.shape)), units = "m")
        w = pxp.asarray((pxp.ones(x.shape)), units = "m")
        self.assertNDArrayEqual(v == 1, false)
        self.assertNDArrayEqual(
            pxp.asarray(pxp.zeros_like(x), units="m") == pxp.asarray(pxp.zeros_like(x), units="s") ,
            false,
        )
        self.assertNDArrayEqual(v == w, false)
        self.assertNDArrayEqual(v == w.to("mm"), false)
        self.assertNDArrayEqual(u == v, false)

    def test_dtype(self):
        dtype=pxp.uint32
        u = pxp.asarray([1, 2, 3], dtype=dtype)  * self.ureg.m

        assert u.dtype == dtype

    def test_shape_numpy_func(self):
        assert pxp.asarray(self.q).shape == (2, 2)

    def test_ndim_numpy_func(self):
        assert pxp.asarray(self.q).ndim == 2

    def test_meshgrid_numpy_func(self):
        x = pxp.asarray([1, 2]) * self.ureg.m
        y = pxp.asarray([0, 50, 100]) * self.ureg.mm
        xx, yy = pxp.meshgrid(x, y)
        helpers.assert_quantity_equal(xx, [[1, 2], [1, 2], [1, 2]] * self.ureg.m)
        helpers.assert_quantity_equal(yy, [[0, 0], [50, 50], [100, 100]] * self.ureg.mm)

    def test_comparisons(self):
        # self.assertNDArrayEqual(
        #     pxp.asarray(self.q) > 2 * self.ureg.m, xp.asarray([[False, False], [True, True]])
        # )
        self.assertNDArrayEqual(
            pxp.asarray(self.q) < 2 * self.ureg.m, xp.asarray([[True, False], [False, False]])
        )

    def test_where(self):
        helpers.assert_quantity_equal(
            pxp.where(self.q >= 2 * self.ureg.m, self.q, 20 * self.ureg.m),
            [[20, 2], [3, 4]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            pxp.where(self.q >= 2 * self.ureg.m, self.q, 0),
            [[0, 2], [3, 4]] * self.ureg.m,
        )
        q_float = self.q.astype(float)
        helpers.assert_quantity_equal(
            pxp.where(q_float >= 2 * self.ureg.m, q_float, pxp.nan),
            [[pxp.nan, 2], [3, 4]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            pxp.where(q_float >= 3 * self.ureg.m, 0., q_float),
            [[1, 2], [0, 0]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            pxp.where(q_float >= 3 * self.ureg.m, pxp.nan, q_float),
            [[1, 2], [pxp.nan, pxp.nan]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            pxp.where(q_float >= 2 * self.ureg.m, q_float, pxp.asarray(pxp.nan)* self.ureg.m),
            [[pxp.nan, 2], [3, 4]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            pxp.where(q_float >= 3 * self.ureg.m, pxp.asarray(pxp.nan)* self.ureg.m, q_float),
            [[1, 2], [pxp.nan, pxp.nan]] * self.ureg.m,
        )
        with pytest.raises(DimensionalityError):
            pxp.where(
                q_float < 2 * self.ureg.m,
                q_float,
                0 * self.ureg.J,
            )

        helpers.assert_quantity_equal(
            pxp.where(pxp.asarray([-1., 0., 1.]) * self.ureg.m, pxp.asarray([1., 2., 1.]) * self.ureg.s, pxp.nan),
            pxp.asarray([1., pxp.nan, 1.]) * self.ureg.s,
        )
        with pytest.raises(
            ValueError,
            match=".*Boolean value of Quantity with offset unit is ambiguous",
        ):
            pxp.where(
                self.ureg.Quantity([-1, 0, 1], "degC"), [1, 2, 1] * self.ureg.s, pxp.nan
            )

    def test_tile(self):
        helpers.assert_quantity_equal(
            pxp.tile(pxp.asarray([1,2,3,4]) *self.ureg.m, (4,1)),
            pxp.asarray([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]]) * self.ureg.m
                        )
        helpers.assert_quantity_equal(
            pxp.tile(pxp.asarray([[1, 2], [3, 4]]) *self.ureg.m, (2,1)),
            pxp.asarray([[1, 2],
                        [3, 4],
                        [1, 2],
                        [3, 4]]) * self.ureg.m
                        )   