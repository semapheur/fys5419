import numpy as np


def tensor_type(width: int = 64):
  if width == 64:
    return np.complex64
  return np.complex128


class Tensor(np.ndarray):
  """Tensor representation using numpy.

  Based on Hundt (2022) Quantum Computing for Programmers
  (https://doi.org/10.1017/9781009099974.004)"""

  @property
  def num_bits(self) -> int:
    return int(np.log2(self.shape[0]))

  def __new__(cls, input_array: np.ndarray, width=64) -> "Tensor":
    return np.asarray(input_array, dtype=tensor_type(width)).view(cls)

  def __array_finalize__(self, obj):
    if obj is None:
      return

  def kron(self, arg: "Tensor") -> "Tensor":
    return self.__class__(np.kron(self, arg))

  def __mul__(self, arg) -> "Tensor":
    if not isinstance(arg, Tensor):
      raise TypeError("arg must be an instance of Tensor")
    return self.kron(arg)

  def kpow(self, power: int) -> "Tensor":
    if power == 0:
      return self.__class__(np.array([1.0], dtype=tensor_type(64)))

    tensor = self
    for _ in range(1, power):
      tensor = self.__class__(np.kron(tensor, self))

    return self.__class__(tensor)

  def is_close(self, arg: "Tensor") -> bool:
    return np.allclose(self, arg, atol=1e-6)

  def is_hermitian(self) -> bool:
    if len(self.shape) != 2:
      return False

    if self.shape[0] != self.shape[1]:
      return False

    return self.is_close(np.conj(np.transpose(self)))

  def is_unitary(self) -> bool:
    return Tensor(np.conj(self.T) @ self).is_close(Tensor(np.eye(self.shape[0])))

  def is_permutation(self) -> bool:
    x = self

    return (
      x.ndim == 2
      and x.shape[0] == x.shape[1]
      and (x.sum(axis=0) == 1).all()
      and (x.sum(axis=1) == 1).all()
      and ((x == 1) or (x == 0)).all()
    )
