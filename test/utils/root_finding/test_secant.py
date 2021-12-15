"""
Tests for root-finding with the secant method.
"""

import unittest

import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab

from cvx_nn.utils.root_finding.secant import secant, TOL


@parameterized_class(lab.TEST_GRID)
class TestSecantMethod(unittest.TestCase):
    """Test parallel root-finding with the secant method."""

    d: int = 2
    n: int = 4
    rng: np.random.Generator = np.random.default_rng(778)

    n_parallel: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

    def test_linear_secant(self):
        """Test root-finding on simple linear functions."""

        # simplest possible problem: 1-d linear function
        a: float = 2.0
        b: float = -1.0
        root: float = 0.5

        def simple_obj(x):
            return a * x + b

        w_secant, exit_status = secant(simple_obj, lab.tensor(0.0), lab.tensor(1.0))

        self.assertTrue(exit_status["success"], "Secant root-finder reported failure!")

        self.assertTrue(
            lab.abs(simple_obj(w_secant)) <= TOL,
            "Secant method failed to find root within given tolerance.",
        )
        self.assertTrue(
            lab.allclose(w_secant, lab.tensor(root)),
            "Secant method approximation is not close enough to real root.",
        )

        a = lab.tensor(self.rng.standard_normal(self.n_parallel, dtype=self.dtype))
        b = lab.tensor(self.rng.standard_normal(self.n_parallel, dtype=self.dtype))

        root = lab.divide(-b, a)

        def linear_obj(x: lab.Tensor) -> lab.Tensor:
            return lab.multiply(x, a) + b

        w_secant, exit_status = secant(
            linear_obj, lab.zeros(self.n_parallel), lab.ones(self.n_parallel)
        )

        self.assertTrue(exit_status["success"], "Secant root-finder reported failure!")

        self.assertTrue(
            lab.max(lab.abs(linear_obj(w_secant))) <= TOL,
            "Secant method failed to find at least on of the roots within given tolerance.",
        )

        self.assertTrue(
            lab.allclose(w_secant, root),
            "Secant method approximation is not close enough to real roots.",
        )

    def test_quadratic_secant(self):
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

        w_secant, exit_status = secant(
            quadratic_obj, lab.zeros(self.n_parallel), lab.ones(self.n_parallel)
        )

        self.assertTrue(exit_status["success"], "Secant root-finder reported failure!")

        self.assertTrue(
            lab.max(lab.abs(quadratic_obj(w_secant))) <= TOL,
            "Secant method failed to find at least one of the roots within given tolerance.",
        )

    def test_guarded_quadratic_secant(self):
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

        # manually compute roots using the quadratic equation
        plus_roots = (-b + lab.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        minus_roots = (-b - lab.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        midpoint = -b / (2 * a)

        w_secant, exit_status = secant(
            quadratic_obj,
            midpoint,
            midpoint + lab.ones_like(midpoint),
            lower_b=midpoint,
        )
        self.assertTrue(exit_status["success"], "Secant root-finder reported failure!")

        self.assertTrue(
            lab.allclose(plus_roots, w_secant),
            "Secant method should find only the 'plus' roots when properly guarded.",
        )

        w_secant, exit_status = secant(
            quadratic_obj,
            midpoint,
            midpoint - lab.ones_like(midpoint),
            upper_b=midpoint,
        )
        self.assertTrue(exit_status["success"], "Secant root-finder reported failure!")

        self.assertTrue(
            lab.allclose(minus_roots, w_secant),
            "Secant method should find only the 'minus' roots when properly guarded.",
        )


if __name__ == "__main__":
    unittest.main()
