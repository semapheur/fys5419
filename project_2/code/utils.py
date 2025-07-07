import sympy as sp


def quantum_fourier_matrix(n: int) -> sp.Matrix:
  """
  Generate the quantum Fourier transform matrix for n qubits using Sympy.

  Args:
    n (int): The number of qubits, determining the size of the matrix as 2^n x 2^n.

  Returns:
    sp.Matrix: The quantum Fourier transform matrix of size 2^n x 2^n.
  """

  N = 2**n
  omega = sp.exp(sp.I * 2 * sp.pi / N)

  return (1 / sp.sqrt(N)) * sp.Matrix(N, N, lambda i, j: omega ** (i * j))
