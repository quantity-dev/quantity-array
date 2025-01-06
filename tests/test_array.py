from __future__ import annotations

import copy
import operator as op
import pickle

import numpy as np
import pytest
from pint import DimensionalityError, OffsetUnitCalculusError, UnitRegistry
from pint.testsuite import helpers

import pint_array

pnp = pint_array.pint_namespace(np)
ureg = UnitRegistry()

<<<<<<< HEAD
<<<<<<< HEAD
=======

<<<<<<< HEAD
# @helpers.requires_numpy
>>>>>>> 1ad9ca9 (style: pre-commit fixes)
=======

>>>>>>> 8498256 (style: pre-commit fixes)
class TestNumpyMethods:
=======
class TestNumPyMethods:
>>>>>>> 573558d (fix xp-tests)
    @classmethod
    def setup_class(cls):
        from pint import _DEFAULT_REGISTRY

        cls.ureg = _DEFAULT_REGISTRY
        cls.Q_ = cls.ureg.Quantity

    @classmethod
    def teardown_class(cls):
        cls.ureg = None
        cls.Q_ = None

    @property
    def q(self):
        return [[1, 2], [3, 4]] * self.ureg.m

    @property
    def q_scalar(self):
        return np.array(5) * self.ureg.m

    @property
    def q_nan(self):
        return [[1, 2], [3, pnp.nan]] * self.ureg.m

    @property
    def q_zero_or_nan(self):
        return [[0, 0], [0, pnp.nan]] * self.ureg.m

    @property
    def q_temperature(self):
        return self.Q_([[1, 2], [3, 4]], self.ureg.degC)

    def assertNDArrayEqual(self, actual, desired):
        # Assert that the given arrays are equal, and are not Quantities
        np.testing.assert_array_equal(actual, desired)
        assert not isinstance(actual, self.Q_)
        assert not isinstance(desired, self.Q_)


class TestNumPyArrayCreation(TestNumPyMethods):
    # https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html

    @pytest.mark.xfail(reason="Scalar argument issue ")
    def test_ones_like(self):
        self.assertNDArrayEqual(pnp.ones_like(self.q), np.array([[1, 1], [1, 1]]))

    def test_zeros_like(self):
        self.assertNDArrayEqual(pnp.zeros_like(self.q), np.array([[0, 0], [0, 0]]))

    def test_empty_like(self):
        ret = pnp.empty_like(self.q)
        assert ret.shape == (2, 2)
        assert isinstance(ret.magnitude, np.ndarray)

    @pytest.mark.xfail(reason="Scalar argument issue ")
    def test_full_like(self):
        helpers.assert_quantity_equal(
            pnp.full_like(self.q, self.Q_(0, self.ureg.degC)),
            self.Q_([[0, 0], [0, 0]], self.ureg.degC),
        )
        self.assertNDArrayEqual(pnp.full_like(self.q, 2), np.array([[2, 2], [2, 2]]))


class TestNumPyArrayManipulation(TestNumPyMethods):
    # Changing array shape

    def test_flatten(self):
        helpers.assert_quantity_equal(self.q.flatten(), [1, 2, 3, 4] * self.ureg.m)

    def test_flat(self):
        for q, v in zip(self.q.flat, [1, 2, 3, 4], strict=False):
            assert q == v * self.ureg.m

    def test_reshape(self):
        helpers.assert_quantity_equal(
            self.q.reshape([1, 4]), [[1, 2, 3, 4]] * self.ureg.m
        )

    # Transpose-like operations

    def test_moveaxis(self):
        helpers.assert_quantity_equal(
            pnp.moveaxis(self.q, 1, 0), np.array([[1, 2], [3, 4]]).T * self.ureg.m
        )

    def test_transpose(self):
        helpers.assert_quantity_equal(
            pnp.matrix_transpose(self.q), [[1, 3], [2, 4]] * self.ureg.m
        )

    def test_flip_numpy_func(self):
        helpers.assert_quantity_equal(
            pnp.flip(self.q, axis=0), [[3, 4], [1, 2]] * self.ureg.m
        )

    # Changing number of dimensions

    def test_broadcast_to(self):
        helpers.assert_quantity_equal(
            pnp.broadcast_to(self.q[:, 1], (2, 2)),
            np.array([[2, 4], [2, 4]]) * self.ureg.m,
        )

    def test_expand_dims(self):
        helpers.assert_quantity_equal(
            pnp.expand_dims(self.q, 0), np.array([[[1, 2], [3, 4]]]) * self.ureg.m
        )

    def test_squeeze(self):
        helpers.assert_quantity_equal(pnp.squeeze(self.q), self.q)
        helpers.assert_quantity_equal(
            pnp.squeeze(self.q.reshape([1, 4])), [1, 2, 3, 4] * self.ureg.m
        )

    # Changing number of dimensions
    # Joining arrays

    def test_concat_stack(self, subtests):
        for func in (pnp.concat, pnp.stack):
            with subtests.test(func=func):
                helpers.assert_quantity_equal(
                    func([self.q] * 2), self.Q_(func([self.q.m] * 2), self.ureg.m)
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
        actual = self.q.astype(pnp.float32)
        expected = self.Q_(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=pnp.float32), "m")
        helpers.assert_quantity_equal(actual, expected)
        assert actual.m.dtype == expected.m.dtype

    def test_item(self):
        helpers.assert_quantity_equal(self.Q_([[0]], "m").item(), 0 * self.ureg.m)

    def test_broadcast_arrays(self):
        x = self.Q_(np.array([[1, 2, 3]]), "m")
        y = self.Q_(np.array([[4], [5]]), "nm")
        result = pnp.broadcast_arrays(x, y)
        expected = (
            self.Q_(np.array([[1, 2, 3], [1, 2, 3]]), "m"),
            self.Q_(np.array([[4, 4, 4], [5, 5, 5]]), "nm"),
        )
        helpers.assert_quantity_equal(result, expected)

    def test_roll(self):
        helpers.assert_quantity_equal(
            pnp.roll(self.q, 1), [[4, 1], [2, 3]] * self.ureg.m
        )


class TestNumPyMathematicalFunctions(TestNumPyMethods):
    # https://www.numpy.org/devdocs/reference/routines.math.html

    def test_prod_numpy_func(self):
        axis = 0

        helpers.assert_quantity_equal(pnp.prod(self.q), 24 * self.ureg.m**4)
        helpers.assert_quantity_equal(
            pnp.prod(self.q, axis=axis), [3, 8] * self.ureg.m**2
        )

    def test_sum_numpy_func(self):
        helpers.assert_quantity_equal(pnp.sum(self.q, axis=0), [4, 6] * self.ureg.m)
        with pytest.raises(OffsetUnitCalculusError):
            pnp.sum(self.q_temperature)

    # Arithmetic operations
    def test_addition_with_scalar(self):
        a = np.array([0, 1, 2])
        b = 10.0 * self.ureg("gram/kilogram")
        helpers.assert_quantity_almost_equal(
            a + b, self.Q_([0.01, 1.01, 2.01], self.ureg.dimensionless)
        )
        helpers.assert_quantity_almost_equal(
            b + a, self.Q_([0.01, 1.01, 2.01], self.ureg.dimensionless)
        )

    def test_addition_with_incompatible_scalar(self):
        a = np.array([0, 1, 2])
        b = 1.0 * self.ureg.m
        with pytest.raises(DimensionalityError):
            op.add(a, b)
        with pytest.raises(DimensionalityError):
            op.add(b, a)

    def test_power(self):
        arr = np.array(range(3), dtype=float)
        q = self.Q_(arr, "meter")

        for op_ in [pnp.pow]:
            q_cp = copy.copy(q)
            with pytest.raises(DimensionalityError):
                op_(2.0, q_cp)
            arr_cp = copy.copy(arr)
            q_cp = copy.copy(q)
            with pytest.raises(DimensionalityError):
                op_(q_cp, arr_cp)
            q_cp = copy.copy(q)
            q2_cp = copy.copy(q)
            with pytest.raises(DimensionalityError):
                op_(q_cp, q2_cp)

        helpers.assert_quantity_equal(
            pnp.pow(self.q, self.Q_(2)), self.Q_([[1, 4], [9, 16]], "m**2")
        )
        helpers.assert_quantity_equal(
            self.q ** self.Q_(2), self.Q_([[1, 4], [9, 16]], "m**2")
        )
        self.assertNDArrayEqual(arr ** self.Q_(2), np.array([0, 1, 4]))

    def test_sqrt(self):
        q = self.Q_(100, "m**2")
        helpers.assert_quantity_equal(pnp.sqrt(q), self.Q_(10, "m"))

    @pytest.mark.xfail
    def test_exponentiation_array_exp_2(self):
        arr = np.array(range(3), dtype=float)
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
            pnp.repeat(self.q, 2), [1, 1, 2, 2, 3, 3, 4, 4] * self.ureg.m
        )

    def test_sort_numpy_func(self):
        q = [4, 5, 2, 3, 1, 6] * self.ureg.m
        helpers.assert_quantity_equal(pnp.sort(q), [1, 2, 3, 4, 5, 6] * self.ureg.m)

    def test_argsort_numpy_func(self):
        self.assertNDArrayEqual(pnp.argsort(self.q, axis=0), np.array([[0, 0], [1, 1]]))

    def test_searchsorted_numpy_func(self):
        """Test searchsorted as numpy function."""
        q = self.q.flatten()
        self.assertNDArrayEqual(pnp.searchsorted(q, [1.5, 2.5] * self.ureg.m), [1, 2])

    def test_nonzero_numpy_func(self):
        q = [1, 0, 5, 6, 0, 9] * self.ureg.m
        self.assertNDArrayEqual(pnp.nonzero(q)[0], [0, 2, 3, 5])

    def test_any_numpy_func(self):
        q = [0, 1] * self.ureg.m
        assert pnp.any(q)
        with pytest.raises(ValueError, match="offset unit is ambiguous"):
            pnp.any(self.q_temperature)

    def test_all_numpy_func(self):
        q = [0, 1] * self.ureg.m
        assert not pnp.all(q)
        with pytest.raises(ValueError, match="offset unit is ambiguous"):
            pnp.all(self.q_temperature)

    def test_max_numpy_func(self):
        assert pnp.max(self.q) == 4 * self.ureg.m

    def test_max_with_axis_arg(self):
        helpers.assert_quantity_equal(pnp.max(self.q, axis=1), [2, 4] * self.ureg.m)

    def test_argmax_numpy_func(self):
        self.assertNDArrayEqual(pnp.argmax(self.q, axis=0), np.array([1, 1]))

    def test_maximum(self):
        helpers.assert_quantity_equal(
            pnp.maximum(self.q, self.Q_([0, 5], "m")), self.Q_([[1, 5], [3, 5]], "m")
        )

    def test_min_numpy_func(self):
        assert pnp.min(self.q) == 1 * self.ureg.m

    def test_min_with_axis_arg(self):
        helpers.assert_quantity_equal(pnp.min(self.q, axis=1), [1, 3] * self.ureg.m)

    def test_argmin_numpy_func(self):
        self.assertNDArrayEqual(pnp.argmin(self.q, axis=0), np.array([0, 0]))

    def test_minimum(self):
        helpers.assert_quantity_equal(
            pnp.minimum(self.q, self.Q_([0, 5], "m")), self.Q_([[0, 2], [0, 4]], "m")
        )

    def test_clip_numpy_func(self):
        helpers.assert_quantity_equal(
            pnp.clip(self.q, 150 * self.ureg.cm, None), [[1.5, 2], [3, 4]] * self.ureg.m
        )

    def test_round_numpy_func(self):
        helpers.assert_quantity_equal(
            pnp.round(1.0275 * self.ureg.m, decimals=2), 1.03 * self.ureg.m
        )

    def test_cumulative_sum(self):
        helpers.assert_quantity_equal(
            pnp.cumulative_sum(self.q, axis=0), [[1, 2], [4, 6]] * self.ureg.m
        )

    def test_mean_numpy_func(self):
        assert pnp.mean(self.q) == 2.5 * self.ureg.m
        assert pnp.mean(self.q_temperature) == self.Q_(2.5, self.ureg.degC)

    def test_var_numpy_func(self):
        assert pnp.var(self.q) == 1.25 * self.ureg.m**2
        assert pnp.var(self.q_temperature) == 1.25 * self.ureg.delta_degC**2

    def test_std_numpy_func(self):
        helpers.assert_quantity_almost_equal(
            pnp.std(self.q), 1.11803 * self.ureg.m, rtol=1e-5
        )
        helpers.assert_quantity_almost_equal(
            pnp.std(self.q_temperature), 1.11803 * self.ureg.delta_degC, rtol=1e-5
        )

    def test_conj(self):
        helpers.assert_quantity_equal((self.q * (1 + 1j)).conj(), self.q * (1 - 1j))
        helpers.assert_quantity_equal(
            (self.q * (1 + 1j)).conjugate(), self.q * (1 - 1j)
        )

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
            self.q[0] = np.ndarray([1, 2])
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
        helpers.assert_quantity_equal(q, pnp.asarray([1, 1, 2, 3]))
        q[0] = self.ureg.m / self.ureg.mm
        helpers.assert_quantity_equal(q, pnp.asarray([1000, 1, 2, 3]))

        q = [0.0, 1.0, 2.0, 3.0] * self.ureg.m / self.ureg.mm
        q[0] = 1.0
        helpers.assert_quantity_equal(q, [0.001, 1, 2, 3] * self.ureg.m / self.ureg.mm)

    def test_reversible_op(self):
        """ """
        x = self.q.magnitude
        u = self.Q_(pnp.ones(x.shape))
        helpers.assert_quantity_equal(x / self.q, u * x / self.q)
        helpers.assert_quantity_equal(x * self.q, u * x * self.q)
        helpers.assert_quantity_equal(x + u, u + x)
        helpers.assert_quantity_equal(x - u, -(u - x))

    def test_pickle(self, subtests):
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with subtests.test(protocol):
                q1 = [10, 20] * self.ureg.m
                q2 = pickle.loads(pickle.dumps(q1, protocol))
                self.assertNDArrayEqual(q1.magnitude, q2.magnitude)
                assert q1.units == q2.units

    def test_equal(self):
        x = self.q.magnitude
        u = self.Q_(pnp.ones(x.shape))
        false = pnp.zeros_like(x, dtype=np.bool_)

        helpers.assert_quantity_equal(u, u)
        helpers.assert_quantity_equal(u, u.magnitude)
        helpers.assert_quantity_equal(u == 1, u.magnitude == 1)

        v = self.Q_(pnp.zeros(x.shape), "m")
        w = self.Q_(pnp.ones(x.shape), "m")
        self.assertNDArrayEqual(v == 1, false)
        self.assertNDArrayEqual(
            self.Q_(pnp.zeros_like(x), "m") == self.Q_(pnp.zeros_like(x), "s"),
            false,
        )
        self.assertNDArrayEqual(v == w, false)
        self.assertNDArrayEqual(v == w.to("mm"), false)
        self.assertNDArrayEqual(u == v, false)

    def test_dtype(self):
        u = self.Q_(pnp.arange(12, dtype="uint32"))

        assert u.dtype == "uint32"

    def test_shape_numpy_func(self):
        assert pnp.asarray(self.q).shape == (2, 2)

    def test_ndim_numpy_func(self):
        assert pnp.asarray(self.q).ndim == 2

    def test_meshgrid_numpy_func(self):
        x = [1, 2] * self.ureg.m
        y = [0, 50, 100] * self.ureg.mm
        xx, yy = pnp.meshgrid(x, y)
        helpers.assert_quantity_equal(xx, [[1, 2], [1, 2], [1, 2]] * self.ureg.m)
        helpers.assert_quantity_equal(yy, [[0, 0], [50, 50], [100, 100]] * self.ureg.mm)

    def test_comparisons(self):
        self.assertNDArrayEqual(
            self.q > 2 * self.ureg.m, np.array([[False, False], [True, True]])
        )
        self.assertNDArrayEqual(
            self.q < 2 * self.ureg.m, np.array([[True, False], [False, False]])
        )

    def test_where(self):
        helpers.assert_quantity_equal(
            pnp.where(self.q >= 2 * self.ureg.m, self.q, 20 * self.ureg.m),
            [[20, 2], [3, 4]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            pnp.where(self.q >= 2 * self.ureg.m, self.q, 0),
            [[0, 2], [3, 4]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            pnp.where(self.q >= 2 * self.ureg.m, self.q, pnp.nan),
            [[pnp.nan, 2], [3, 4]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            pnp.where(self.q >= 3 * self.ureg.m, 0, self.q),
            [[1, 2], [0, 0]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            pnp.where(self.q >= 3 * self.ureg.m, pnp.nan, self.q),
            [[1, 2], [pnp.nan, pnp.nan]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            pnp.where(self.q >= 2 * self.ureg.m, self.q, np.array(pnp.nan)),
            [[pnp.nan, 2], [3, 4]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            pnp.where(self.q >= 3 * self.ureg.m, np.array(pnp.nan), self.q),
            [[1, 2], [pnp.nan, pnp.nan]] * self.ureg.m,
        )
        with pytest.raises(DimensionalityError):
            pnp.where(
                self.q < 2 * self.ureg.m,
                self.q,
                0 * self.ureg.J,
            )

        helpers.assert_quantity_equal(
            pnp.where([-1, 0, 1] * self.ureg.m, [1, 2, 1] * self.ureg.s, pnp.nan),
            [1, pnp.nan, 1] * self.ureg.s,
        )
        with pytest.raises(
            ValueError,
            match=".*Boolean value of Quantity with offset unit is ambiguous",
        ):
            pnp.where(
                self.ureg.Quantity([-1, 0, 1], "degC"), [1, 2, 1] * self.ureg.s, pnp.nan
            )

    def test_tile(self):
        helpers.assert_quantity_equal(
            pnp.tile(self.q, 2), np.array([[1, 2, 1, 2], [3, 4, 3, 4]]) * self.ureg.m
        )
