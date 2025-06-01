from __future__ import annotations
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from qubit import NQubitState
from tensor import Tensor, COMPLEX_DTYPE


class Gate(Tensor):
  """
  A class for representing quantum gates as numpy arrays.

  Based on Hundt (2022) Quantum Computing for Programmers
  (https://doi.org/10.1017/9781009099974.004)
  https://github.com/qcc4cp/qcc/

  Parameters:
    matrix (np.ndarray): The input array representing the quantum gate.
  """

  name: str

  def __new__(cls, matrix: ArrayLike, name: str | None = None) -> Gate:
    obj = np.asarray(matrix, dtype=COMPLEX_DTYPE, copy=True)

    if obj.ndim != 2:
      raise ValueError(f"Gate must be a 2D matrix. Got {obj.shape}")

    if obj.shape[0] != obj.shape[1]:
      raise ValueError(f"Gate must be a square matrix. Got {obj.shape}")

    dim = obj.shape[0]
    if (dim & (dim - 1)) != 0:
      raise ValueError(f"Gate must be a power of 2. Got {dim}")

    gate = obj.view(cls)
    gate.name = name or "U"
    return gate

  def __array_finalize__(self, obj: np.ndarray | None):
    if obj is None:
      return

    self.name = getattr(obj, "name", "U")

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

  def adjoint(self):
    """Return the Hermitian adjoint (complex transpose) of the gate."""
    return Gate(np.conj(np.transpose(self)))

  def power(self, n: int) -> Gate:
    """Return the n-th power of the gate."""
    return Gate(np.linalg.matrix_power(self, n))

  def is_unitary(self, atol=1e-6) -> bool:
    """Check if a gate is unitary.

    Args:
      atol (float, optional): Absolute tolerance for the comparison. Defaults to 1e-6.

    Returns:
      bool: True if the gate is unitary, False otherwise.
    """

    return Gate(np.conj(self.T) @ self).is_close(Gate(np.eye(self.shape[0])), atol=atol)


def identity_gate(num_qubits: int = 1) -> Gate:
  return Gate(np.eye(1 << num_qubits, dtype=COMPLEX_DTYPE))  # 2**num_qubits


def pauli_x_gate(num_qubits: int = 1) -> Gate:
  return cast(Gate, Gate([[0.0, 1.0], [1.0, 0.0]]).kpow(num_qubits))


def hadamard_gate(num_qubits: int = 1) -> Gate:
  h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=COMPLEX_DTYPE) / np.sqrt(2.0)
  return cast(Gate, Gate(h).kpow(num_qubits))


def phase_gate(phase: float, num_qubits: int = 1) -> Gate:
  p = np.array([[1.0, 0.0], [0.0, np.exp(1.0j * phase)]])
  return cast(Gate, Gate(p).kpow(num_qubits))


def discrete_phase_gate(k: int, num_qubits: int = 1) -> Gate:
  return phase_gate(2.0 * np.pi / 2**k, num_qubits)


def inverse_discrete_phase_gate(k: int, num_qubits: int = 1) -> Gate:
  return phase_gate(-2.0 * np.pi / 2**k, num_qubits)


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


def flip_gate(gate: Gate) -> Gate:
  """
  Reverse the order of qubits in the input gate.

  Args:
    gate (Gate): The quantum gate to be flipped.

  Returns:
    Gate: A new gate with the qubits in reversed order.
  """

  qubits = gate.num_qubits
  flipped_gate = gate.copy()
  for j in range(qubits // 2):
    flipped_gate = cast(Gate, flipped_gate(swap_gate(j, qubits - j - 1), j))

  return flipped_gate


def fourier_transform(num_qubits: int, inverse: bool = False) -> Gate:
  """
  Construct the quantum Fourier transform (QFT) gate for a given number of qubits.

  Args:
    num_qubits (int): The number of qubits the QFT gate acts on.
    inverse (bool): If True, returns the inverse QFT gate.

  Returns:
    Gate: The quantum Fourier transform gate.
  """
  qft = identity_gate(num_qubits)
  h = hadamard_gate()

  for j in range(num_qubits):
    # Apply Hadamard gate
    qft = cast(Gate, qft(h, j))

    # Apply controlled discrete phase gates
    for k in range(2, num_qubits - j + 1):
      control = j + k - 1
      r_k = discrete_phase_gate(k) if not inverse else inverse_discrete_phase_gate(k)

      qft = cast(Gate, qft(controlled_gate(control, j, r_k), j))

  # Reverse qubit order
  qft = flip_gate(qft)

  if not qft.is_unitary():
    raise ValueError("QFT gate is not unitary")

  return qft


def qft_dagger(num_qubits: int) -> Gate:
  """
  Construct the inverse quantum Fourier transform (QFT) gate for a given number of qubits
  by taking the conjugate transpose of the QFT gate.

  Args:
    num_qubits (int): The number of qubits the IQFT gate acts on.

  Returns:
    Gate: The conjugate transpose of the QFT gate
  """

  return fourier_transform(num_qubits).adjoint()


def inverse_fourier_transform(num_qubits: int) -> Gate:
  """
  Construct the inverse quantum Fourier transform (QFT) gate for a given number of qubits.

  Args:
    num_qubits (int): The number of qubits the QFT gate acts on.
    inverse (bool): If True, returns the inverse QFT gate.

  Returns:
    Gate: The inverse quantum Fourier transform gate.
  """
  qft = identity_gate(num_qubits)
  h = hadamard_gate()

  # Reverse qubit order
  qft = flip_gate(qft)

  for j in reversed(range(num_qubits)):
    # Apply controlled discrete phase gates
    for k in reversed(range(2, num_qubits - j + 1)):
      control = j + k - 1
      r_k = inverse_discrete_phase_gate(k)

      qft = cast(Gate, qft(controlled_gate(control, j, r_k), j))

    # Apply Hadamard gate
    qft = cast(Gate, qft(h, j))

  if not qft.is_unitary():
    raise ValueError("IQFT gate is not unitary")

  return qft
