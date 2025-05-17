from __future__ import annotations
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from qubit import NQubitState
from tensor import Tensor


class Gate(Tensor):
  """
  A class for representing quantum gates as numpy arrays.

  Parameters:
    matrix (np.ndarray): The input array representing the quantum gate.
  """

  def __new__(cls, matrix: ArrayLike) -> Gate:
    obj = np.asarray(matrix, dtype=np.complex128, copy=True)

    if obj.ndim != 2:
      raise ValueError(f"Gate must be a 2D matrix. Got {obj.shape}")

    if obj.shape[0] != obj.shape[1]:
      raise ValueError(f"Gate must be a square matrix. Got {obj.shape}")

    dim = obj.shape[0]
    if (dim & (dim - 1)) != 0:
      raise ValueError(f"Gate must be a power of 2. Got {dim}")

    return obj.view(cls)

  def __matmul__(self, other) -> Gate | NQubitState:
    # Overload the @ operator

    if isinstance(other, Gate):
      return self.__class__(np.matmul(self, other))

    if not isinstance(other, NQubitState):
      raise TypeError("other must be an instance of Gate or NQubitState")

    return NQubitState(np.matmul(self, other))

  def __call__(self, other: Gate | NQubitState, idx: int = 0) -> Gate | NQubitState:
    # Enable calling a Gate instance as a function
    return self.apply(other, idx)

  def apply(self, other: Gate | NQubitState, idx: int) -> Gate | NQubitState:
    """
    Apply this gate to a qubit state or compose it with another gate, starting at the specified qubit index.

    Args:
      other (Gate | NQubitState): The target to which this gate is applied. Can be either:
        - a `Gate`, in which case the result is a composition (operator product),
        - an `NQubitState`, in which case the gate evolves the state.
      idx (int): Index of the first qubit this gate acts on within `other`

    Returns:
      Gate | NQubitState:
        - A new `Gate` instance representing the composed operator if `other` is a `Gate`.
        - A new `NQubitState` instance representing the evolved state if `other` is an `NQubitState`.
    """

    if not hasattr(other, "num_qubits"):
      raise TypeError("other must be an instance of Gate or NQubitState")

    self_qubits = self.num_qubits
    other_qubits = other.num_qubits

    if idx < 0:
      raise ValueError("Index must be non-negative")

    if isinstance(other, Gate):
      if other_qubits > self_qubits:
        raise ValueError(
          f"Other gate qubits ({other_qubits}) cannot exceed self qubits ({self_qubits})"
        )

      if idx > self_qubits - other_qubits:
        raise ValueError(
          f"Index must be less than {self_qubits - other_qubits}. Got {idx}"
        )

      if idx > 0:  # left-pad other with identity
        other = cast(Gate, identity_gate(idx) * other)

      right_pad = self_qubits - other_qubits - idx
      if right_pad > 0:  # right-pad other with identity
        other = cast(Gate, other * identity_gate(right_pad))

      assert self_qubits == other.num_qubits, (
        f"Dimension mismatch: self ({self_qubits}) != other ({other.num_qubits})"
      )

      return other @ self

    if not isinstance(other, NQubitState):
      raise TypeError("other must be an instance of Gate or NQubitState")

    if self_qubits > other_qubits:
      raise ValueError(
        f"Gate qubits ({self_qubits}) cannot exceed state qubits ({other_qubits})"
      )

    if idx > other_qubits - self_qubits:
      raise ValueError(
        f"Index must be less than {other_qubits - self_qubits}. Got {idx}"
      )

    gate = self
    if idx > 0:  # left-pad gate with identity
      gate = cast(Gate, identity_gate(idx) * gate)

    right_pad = other_qubits - self_qubits - idx
    if right_pad > 0:  # right-pad gate with identity
      gate = cast(Gate, gate * identity_gate(right_pad))

    assert gate.num_qubits == other_qubits, (
      f"Dimension mismatch: gate ({gate.num_qubits}) != state ({other_qubits})"
    )

    return gate @ other

  @property
  def dim(self) -> int:
    return self.shape[0]

  def dagger(self):
    """Return the Hermitian adjoint (complex transpose) of the gate."""
    return Gate(np.conj(np.transpose(self)))

  def is_unitary(self, atol=1e-6) -> bool:
    """Check if a gate is unitary.

    Args:
      atol (float, optional): Absolute tolerance for the comparison. Defaults to 1e-6.

    Returns:
      bool: True if the gate is unitary, False otherwise.
    """

    return Gate(np.conj(self.T) @ self).is_close(Gate(np.eye(self.shape[0])), atol=atol)


def identity_gate(num_qubits: int = 1) -> Gate:
  return Gate(np.eye(1 << num_qubits, dtype=np.complex128))  # 2**num_qubits


def pauli_x_gate(num_qubits: int = 1) -> Gate:
  return cast(Gate, Gate([[0.0, 1.0], [1.0, 0.0]]).kpow(num_qubits))


def hadamard_gate(num_qubits: int = 1) -> Gate:
  h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
  return cast(Gate, Gate(h).kpow(num_qubits))


def discrete_phase_gate(k: int, num_bits: int = 1) -> Gate:
  p_k = np.array([[1.0, 0.0], [0.0, np.exp(2.0j * np.pi / 2**k)]])
  return cast(Gate, Gate(p_k).kpow(num_bits))


def controlled_gate(control: int, target: int, gate: Gate) -> Gate:
  """
  Construct a controlled gate from a given gate and control and target qubits.

  Args:
    control (int): Index of control qubit
    target (int): Index of target qubit
    gate (Gate): Gate to control

  Returns:
    Gate: The controlled gate
  """

  if control == target:
    raise ValueError("Control and target qubits must be different")

  p0 = Gate(np.array([[1.0, 0.0], [0.0, 0.0]]))  # |0><0|
  p1 = Gate(np.array([[0.0, 0.0], [0.0, 1.0]]))  # |1><1|

  # space between qubits
  i_fill = identity_gate(abs(target - control) - 1)

  # width of operator in terms of identity gates
  gate_fill = identity_gate(gate.num_qubits)

  if target > control:
    c_gate = p0 * i_fill * gate_fill + p1 * i_fill * gate
  else:
    c_gate = gate_fill * i_fill * p0 + gate * i_fill * p1

  return cast(Gate, c_gate)


def cnot_gate(control: int = 0, target: int = 1) -> Gate:
  return controlled_gate(control, target, pauli_x_gate())


def swap_gate(qubit1: int, qubit2: int) -> Gate:
  return Gate(
    cnot_gate(qubit2, qubit1) @ cnot_gate(qubit1, qubit2) @ cnot_gate(qubit2, qubit1)
  )


def fourier_transform(num_qubits: int) -> Gate:
  qft = identity_gate(num_qubits)
  h = hadamard_gate()

  for j in range(num_qubits):
    qft = cast(Gate, qft(h, j))

    for k in range(2, num_qubits - j + 1):
      control = j + k - 1
      qft = cast(Gate, qft(controlled_gate(control, j, discrete_phase_gate(k)), j))

  for j in range(num_qubits // 2):
    qft = cast(Gate, qft(swap_gate(j, num_qubits - j - 1), j))

  if not qft.is_unitary():
    raise ValueError("QFT gate is not unitary")

  return qft


def inverse_fourier_transform(num_qubits: int) -> Gate:
  return fourier_transform(num_qubits).dagger()


def inverse_fourier_transform_(num_qubits: int) -> Gate:
  iqft = identity_gate(num_qubits)
  h = hadamard_gate()

  for j in range(num_qubits // 2):
    iqft = cast(Gate, iqft(swap_gate(j, num_qubits - j - 1), j))

  for j in reversed(range(num_qubits)):
    for k in reversed(range(2, num_qubits - j + 1)):
      control = j + k - 1
      iqft = cast(Gate, iqft(controlled_gate(control, j, discrete_phase_gate(k)), j))

    iqft = cast(Gate, iqft(h, j))

  if not iqft.is_unitary():
    raise ValueError("IQFT gate is not unitary")

  return iqft
