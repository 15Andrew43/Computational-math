import numpy as np


def scalar_product(vector1, vector2):
    return sum(vector1 * vector2)


def norm(vector):
    return np.sqrt(scalar_product(vector, vector))


def qr_decomposition(matrix):
    m, n = matrix.shape
    P = Q = np.zeros(shape=(m, n))
    R = np.zeros(shape=(n, n))

    for j in range(n):
        P[:, j] = matrix[:, j]
        for i in range(0, j):
            sc = scalar_product(P[:, j], Q[:, i])
            P[:, j] = P[:, j] - Q[:, i] * sc
            R[i, j] = sc
        CurNorm = norm(P[:, j])
        R[j, j] = CurNorm
        Q[:, j] = P[:, j] / CurNorm

    return Q, R


def reverse_substitution_method(R, vector):
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = vector[i, 0]
        for j in range(n-1, i, -1):
            x[i] -= x[j] * R[i, j]
        x[i] /= R[i, i]

    return x

def leasts_quares_method(Q, R, vector):
    Rx = Q.T @ vector
    x = reverse_substitution_method(R, Rx)
    return x


def solve_eq(matrix, vector):
    Q, R = qr_decomposition(matrix)
    ApproximateSolution = leasts_quares_method(Q, R, vector)
    return Q, R, ApproximateSolution


def Print(Q, R, S, real_solution):
    print(f"Q = \n{Q},\n R = \n{R},\n solution = {S},\nreal solution = \n{real_solution}\n\n\n")

if __name__ == '__main__':

    A = np.array([
            [1/17, 1/51, 1/51],
            [1/17, -5/51, 10/51],
            [-4/17, 1/51, -20/51]
    ])
    b = np.array([
        [20],
        [100],
        [1000]
    ])
    real_solution = np.linalg.solve(A, b)
    Q, R, S = solve_eq(A, b)
    Print(Q, R, S, real_solution)


    A = np.array([
        [1, 0.5],
        [0, 0.5],
        [0, 0]
    ])
    b = np.array([
        [3],
        [4],
        [1000]
    ])
    real_solution = np.linalg.lstsq(A, b, rcond=None)
    Q, R, S = solve_eq(A, b)
    Print(Q, R, S, real_solution)

