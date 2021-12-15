"""
Test implementation of conjugate-gradient for block-diagonal matrices.
"""

import unittest
import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab
from cvx_nn.utils.linear import preconditioners, block_cg
from cvx_nn.utils import BlockDiagonalMatrix


@parameterized_class(lab.TEST_GRID)
class TestBlockCG(unittest.TestCase):
    """Test conjugate-gradient method for solving block-diagonal linear systems."""

    d: int = 50
    n: int = 200

    num_blocks: int = 5
    block_d: int = 10
    block_n: int = 50

    rng: np.random.Generator = np.random.default_rng(778)

    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        # generate PD systems to solve.

        X = lab.tensor(self.rng.standard_normal((self.n, self.d), dtype=self.dtype))
        self.w_opt = lab.tensor(self.rng.standard_normal((1, self.d), dtype=self.dtype))

        self.matrix = lab.matmul(X.T, X)
        self.linear_op = BlockDiagonalMatrix(
            forward=lambda v, z: lab.einsum("ij, kj->ki", self.matrix, v)
        )
        self.targets = self.linear_op.matvec(self.w_opt)

        self.forward = preconditioners.column_norm(self.matrix)

    def test_linear_solver(self):
        """Test solving a simple linear system."""
        w_cg, exit_status = block_cg.block_cg_solve(
            self.linear_op, self.targets, 1, self.d, flatten=True
        )

        self.assertTrue(exit_status["success"], "The cg solver reported failure!")

        # solve using numpy.solve
        w_np = lab.solve(self.matrix, self.targets.squeeze())

        self.assertTrue(
            lab.allclose(w_cg, self.w_opt, atol=1e-5, rtol=1e-5),
            "Iterative solution failed to match ground truth.",
        )
        self.assertTrue(
            lab.allclose(w_cg, w_np, atol=1e-5, rtol=1e-5),
            "Iterative solution failed to match direct solver.",
        )

    def test_solving_block_systems(self):
        """Test solving block-diagonal linear systems."""

        matrix_blocks = []
        w_opt = []
        targets = []

        for i in range(self.num_blocks):
            block_X = lab.tensor(
                self.rng.standard_normal((self.block_n, self.block_d), dtype=self.dtype)
            )
            block_XX = lab.matmul(block_X.T, block_X)
            block_opt = lab.tensor(
                self.rng.standard_normal(self.block_d, dtype=self.dtype)
            )
            targets.append(block_XX @ block_opt)

            matrix_blocks.append(block_XX)
            w_opt.append(block_opt)

        w_opt = lab.stack(w_opt)
        targets = lab.stack(targets)

        def block_matvec(v, indices):
            results = []
            w = v.reshape(len(indices), self.block_d)
            for i, block_index in enumerate(indices):
                results.append(matrix_blocks[block_index] @ w[i])

            return lab.stack(results)

        linear_op = BlockDiagonalMatrix(forward=block_matvec)

        w_iter, exit_status = block_cg.block_cg_solve(
            linear_op, targets, self.num_blocks, self.block_d
        )

        self.assertTrue(exit_status["success"], "The linear solver reported failure!")

        # compare against ground truth.

        self.assertTrue(
            lab.allclose(w_iter, w_opt, atol=1e-5, rtol=1e-5),
            "Iterative solution failed to match ground truth.",
        )

        # test solving only a subset of the systems.
        starting_blocks = lab.tensor([0, 2, 4], dtype=int)
        w_iter, exit_status = block_cg.block_cg_solve(
            linear_op,
            targets,
            self.num_blocks,
            self.block_d,
            starting_blocks=starting_blocks,
        )

        self.assertTrue(exit_status["success"], "The linear solver reported failure!")

        # compare against ground truth.

        self.assertTrue(
            lab.allclose(
                w_iter[starting_blocks],
                w_opt[starting_blocks],
                atol=1e-5,
                rtol=1e-5,
            ),
            "Iterative solution failed to match ground truth when solving only some of the blocks.",
        )


if __name__ == "__main__":
    unittest.main()
