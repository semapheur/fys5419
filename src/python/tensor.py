from __future__ import annotations
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

COMPLEX_DTYPE = np.complex128


class Tensor(np.ndarray):
  """A class for representing tensors as numpy arrays.

  Based on Hundt (2022) Quantum Computing for Programmers
  (https://doi.org/10.1017/9781009099974.004)
  https://github.com/qcc4cp/qcc/

  Attributes:
    num_qubits (int): The number of qubits in the tensor.
  """

  def __new__(cls, array: ArrayLike) -> Tensor:
    return np.asarray(array, dtype=COMPLEX_DTYPE, copy=True).view(cls)

  def __array_finalize__(self, obj: np.ndarray | None):
    if obj is None:
      return

  @property
  def num_qubits(self) -> int:
    dim = cast(int, self.shape[0])

    if (dim & (dim - 1)) != 0:
      raise ValueError(
        f"Dimension {dim} is not a power of 2; cannot determine number of qubits"
      )

    return dim.bit_length() - 1

  def kron(self, other: Tensor) -> Tensor:
    """
    Compute the Kronecker product of this tensor with another tensor.

    Args:
      other (Tensor): The tensor to compute the Kronecker product with.

    Returns:
      Tensor: The result of the Kronecker product as a new Tensor instance.
    """

    return self.__class__(np.kron(self, other))

  def __mul__(self, other) -> Tensor:
    # Overload the * operator to compute the Kronecker product

    if not isinstance(other, Tensor):
      raise TypeError("other must be an instance of Tensor")
    return self.kron(other)

  def kpow(self, power: int) -> Tensor:
    """Calculate the Kronecker power of a gate.

    Args:
      power (int): The number of times to apply the Kronecker product with itself.

    Returns:
      Gate: A new gate representing the Kronecker product of this gate with itself `power` times.
    """

    if power < 0:
      raise ValueError(f"Power must be non-negative. Got {power}")

    if power == 0:
      return self.__class__([1.0])

    if power == 1:
      return self.copy()

    # Apply binary exponentiation with O(log n) complexity
    result = np.array([1.0])
    base = self.copy()
    while power > 0:
      if power & 1 == 1:
        result = np.kron(result, base)

      base = base * base
      power >>= 1

    return self.__class__(result)

  def is_close(self, other: Tensor, atol=1e-6) -> bool:
    """Check if two tensors are approximately equal within a tolerance.

    Args:
      other (Tensor): The tensor to compare with.
      atol (float, optional): Absolute tolerance for the comparison. Defaults to 1e-6.

    Returns:
      bool: True if the tensors are approximately equal, False otherwise.
    """

    return np.allclose(self, other, atol=atol)
