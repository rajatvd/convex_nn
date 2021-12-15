"""
Tests for root-finding with Newton's method.
"""

import unittest

import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab

from cvx_nn.utils.root_finding.newton import newton, TOL


@parameterized_class(lab.TEST_GRID)
class TestNewtonsMethod(unittest.TestCase):
    """Test parallel root-finding with Newton's method."""

    d: int = 2
    n: int = 4
    rng: np.random.Generator = np.random.default_rng(778)

    n_parallel: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

    def test_linear_newton(self):
        """Test root-finding on simple linear functions."""

        # simplest possible problem: 1-d linear function
        a: float = lab.tensor(2.0)
        b: float = lab.tensor(-1.0)
        root: float = lab.tensor(0.5)

        def simple_obj(x):
            return a * x + b

        def simple_grad(x):
            return a

        w_newton, exit_status = newton(simple_obj, simple_grad, lab.tensor(0.0))

        self.assertTrue(exit_status["success"], "Newton root-finder reported failure!")

        self.assertTrue(
            lab.abs(simple_obj(w_newton)) <= TOL,
            "Newton method failed to find root within given tolerance.",
        )
        self.assertTrue(
            lab.allclose(w_newton, root),
            "Newton method approximation is not close enough to real root.",
        )

        a = lab.tensor(self.rng.standard_normal(self.n_parallel, dtype=self.dtype))
        b = lab.tensor(self.rng.standard_normal(self.n_parallel, dtype=self.dtype))

        root = lab.divide(-b, a)

        def linear_obj(x: lab.Tensor) -> lab.Tensor:
            return lab.multiply(a, x) + b

        def linear_grad(x: lab.Tensor) -> lab.Tensor:
            return a

        w_newton, exit_status = newton(
            linear_obj, linear_grad, lab.zeros(self.n_parallel)
        )

        self.assertTrue(exit_status["success"], "Newton root-finder reported failure!")

        self.assertTrue(
            lab.max(lab.abs(linear_obj(w_newton))) <= TOL,
            "Newton method failed to find at least on of the roots within given tolerance.",
        )

        self.assertTrue(
            lab.allclose(w_newton, root),
            "Newton method approximation is not close enough to real roots.",
        )

    def test_quadratic_newton(self):
        """Test root-finding on simple quadratic functions."""

        # more complex problem: 1d quadratic.
        a = lab.abs(
            lab.tensor(self.rng.standard_normal(self.n_parallel, dtype=self.dtype))
        )
        b = lab.tensor(self.rng.standard_normal(self.n_parallel, dtype=self.dtype))
        c = lab.abs(
            lab.tensor(self.rng.standard_normal(self.n_parallel, dtype=self.dtype))
        )

        def quadratic_obj(x: lab.Tensor) -> lab.Tensor:
            return lab.multiply(a, x ** 2) + lab.multiply(b, x) - c

        def quadratic_grad(x: lab.Tensor) -> lab.Tensor:
            return 2 * lab.multiply(a, x) + b

        w_newton, exit_status = newton(
            quadratic_obj, quadratic_grad, lab.zeros(self.n_parallel)
        )

        self.assertTrue(exit_status["success"], "Newton root-finder reported failure!")

        self.assertTrue(
            lab.max(lab.abs(quadratic_obj(w_newton))) <= TOL,
            "Newton method failed to find at least one of the roots within given tolerance.",
        )

    def test_guarded_quadratic_newton(self):
        """Test root-finding on simple quadratic functions."""

        # simplest possible problem: 1-d linear function
        a = lab.abs(
            lab.tensor(self.rng.standard_normal(self.n_parallel, dtype=self.dtype))
        )
        b = lab.tensor(self.rng.standard_normal(self.n_parallel, dtype=self.dtype))
        c = -lab.abs(
            lab.tensor(self.rng.standard_normal(self.n_parallel, dtype=self.dtype))
        )

        def quadratic_obj(x: lab.Tensor) -> lab.Tensor:
            return lab.multiply(a, x ** 2) + lab.multiply(b, x) + c

        def quadratic_grad(x: lab.Tensor) -> lab.Tensor:
            return 2 * lab.multiply(a, x) + b

        # manually compute roots using the quadratic equation
        plus_roots = (-b + lab.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        minus_roots = (-b - lab.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        midpoint = -b / (2 * a)

        w_newton, exit_status = newton(
            quadratic_obj, quadratic_grad, midpoint + 1, lower_b=midpoint
        )
        self.assertTrue(exit_status["success"], "Newton root-finder reported failure!")

        self.assertTrue(
            lab.allclose(plus_roots, w_newton),
            "Newton method should find only the 'plus' roots when properly guarded.",
        )

        w_newton, exit_status = newton(
            quadratic_obj, quadratic_grad, midpoint - 1, upper_b=midpoint
        )
        self.assertTrue(exit_status["success"], "Newton root-finder reported failure!")

        self.assertTrue(
            lab.allclose(minus_roots, w_newton),
            "Newton method should find only the 'minus' roots when properly guarded.",
        )


if __name__ == "__main__":
    unittest.main()
