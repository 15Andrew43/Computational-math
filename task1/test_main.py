import unittest
import numpy as np
from main import solve_eq, norm


eps = 0.001

class LinearSystem:
    def __init__(self, matrix, b, x):
        self.matrix = matrix
        self.b = b
        self.x = x


class Test(unittest.TestCase):
    tests_NN = [
        LinearSystem(
            np.array([
                [1, -1],
                [3, 2]
            ]),
            np.array([
                [7],
                [16]
            ]),
            np.array([6, -1])
        ),
        LinearSystem(
            np.array([
                [1 / 17, 1 / 51, 1 / 51],
                [1 / 17, -5 / 51, 10 / 51],
                [-4 / 17, 1 / 51, -20 / 51]
            ]),
            np.array([
                [20],
                [100],
                [1000]
            ]),
            np.array([6300, -11000, -6880])
        )
    ]
    tests_MN = [
        LinearSystem(
            np.array([
                [1, 0.5],
                [0, 0.5],
                [0, 0]
            ]),
            np.array([
                [3],
                [4],
                [1000]
            ]),
            np.array([])
        )
    ]
    def test_solve_eq_NN(self):
        for system in Test.tests_NN:
            A = system.matrix
            b = system.b
            x = system.x
            self.assertTrue(norm(solve_eq(A, b)[-1] - x) < eps, f"\nwrong solution with matrix {A.shape[0]}x{A.shape[1]}\n")

    def test_solve_eq_MN(self):
        for system in Test.tests_MN:
            A = system.matrix
            b = system.b
            res = A @ solve_eq(A, b)[2]
            print(res)
            self.assertTrue(norm(res[:-1] - b[:-1, 0]) < eps, f"\nwrong solution with matrix {A.shape[0]}x{A.shape[1]}\n")

if __name__ == "__main__":
  unittest.main()