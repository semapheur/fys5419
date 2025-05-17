import numpy as np
from numpy.typing import NDArray

QFT2 = 0.5 * np.array(
  [
    [1, 1, 1, 1],
    [1, 1j, -1, -1j],
    [1, -1, 1, -1j],
    [1, -1j, -1, 1j],
  ],
  dtype=np.complex128,
)

QFT3 = np.array(
  [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [
      1,
      (1 + 1j) / np.sqrt(2),
      1j,
      (-1 + 1j) / np.sqrt(2),
      -1,
      (-1 - 1j) / np.sqrt(2),
      -1j,
      (1 - 1j) / np.sqrt(2),
    ],
    [1, 1j, -1, -1j, 1, 1j, -1, -1j],
    [
      1,
      (-1 + 1j) / np.sqrt(2),
      -1j,
      (1 + 1j) / np.sqrt(2),
      -1,
      (1 - 1j) / np.sqrt(2),
      1j,
      (-1 - 1j) / np.sqrt(2),
    ],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [
      1,
      (-1 - 1j) / np.sqrt(2),
      1j,
      (1 - 1j) / np.sqrt(2),
      -1,
      (1 + 1j) / np.sqrt(2),
      -1j,
      (-1 + 1j) / np.sqrt(2),
    ],
    [1, -1j, -1, 1j, 1, -1j, -1, 1j],
    [
      1,
      (1 - 1j) / np.sqrt(2),
      -1j,
      (-1 - 1j) / np.sqrt(2),
      -1,
      (-1 + 1j) / np.sqrt(2),
      1j,
      (1 + 1j) / np.sqrt(2),
    ],
  ],
  dtype=np.complex128,
) / np.sqrt(8)


def qft_matrix(qubits: int) -> NDArray[np.complex128]:
  """
  Generate the Quantum Fourier Transform (QFT) matrix for a given number of qubits.

  Args:
    qubits (int): Number of qubits.

  Returns:
    NDArray[np.complex128]: The QFT matrix of size 2^qubits x 2^qubits.
  """
  N = 1 << qubits  # 2**qubits
  omega = np.exp(2j * np.pi / N)

  # Use broadcasting to efficiently compute the QFT matrix
  indices = np.arange(N)
  matrix = omega ** np.outer(indices, indices)

  return matrix / np.sqrt(N)


def iqft_matrix(qubits: int) -> NDArray[np.complex128]:
  """
  Generate the Inverse Quantum Fourier Transform (IQFT) matrix for a given number of qubits.

  Args:
    qubits (int): Number of qubits.

  Returns:
    NDArray[np.complex128]: The IQFT matrix of size 2^qubits x 2^qubits.
  """
  N = 1 << qubits  # 2**qubit
  omega = np.exp(-2j * np.pi / N)

  # Use broadcasting to efficiently compute the IQFT matrix
  indices = np.arange(N)
  matrix = omega ** np.outer(indices, indices)

  return matrix / np.sqrt(N)
