import sympy as sp


def quantum_fourier_matrix(n: int) -> sp.Matrix:
  N = 2**n
  omega = sp.exp(sp.I * 2 * sp.pi / N)

  return (1 / sp.sqrt(N)) * sp.Matrix(N, N, lambda i, j: omega ** (i * j))
