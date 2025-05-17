from __future__ import annotations
from typing import cast

import numpy as np

from . import tensor
from . import state


class Operator(tensor.Tensor):
  """Opearator representation using numpy.

  Based on Hundt (2022) Quantum Computing for Programmers
  (https://doi.org/10.1017/9781009099974.004)
  https://github.com/qcc4cp/qcc/
  """

  def __call__(
    self, arg: state.State | Operator, idx: int = 0
  ) -> state.State | Operator:
    return self.apply(arg, idx)

  def apply(self, arg: state.State | Operator, idx: int) -> state.State | Operator:
    """Apply operator to a state or operator."""

    if isinstance(arg, Operator):
      self_bits = self.num_bits
      arg_bits = arg.num_bits
      if idx > 0:
        arg = Operator(identity_gate().kpow(idx) * arg)

      if self_bits > arg_bits:
        arg = Operator(arg * identity_gate().kpow(self.num_bits - idx - arg_bits))

      if self_bits != arg_bits:
        raise AssertionError(
          f"Mismatched operator dimensions: {self_bits} != {arg_bits}"
        )

      return Operator(arg @ self)

    if not isinstance(arg, state.State):
      raise AssertionError("arg must be an instance of State")

    op = self
    if idx > 0:
      op = Operator(identity_gate().kpow(idx) * op)
    if arg.num_bits - idx - self.num_bits > 0:
      op = Operator(op * identity_gate().kpow(arg.num_bits - idx - self.num_bits))

    return state.State(np.matmul(op, arg))

  def __repr__(self) -> str:
    o = "Operator("
    return f"{o}{super().__str__().replace('\n', '\n ' * len(o))})"

  def __str__(self) -> str:
    return (
      f"Operator for {self.num_bits}-qubit state space. Tensor:\n{super().__str__()}"
    )

  def adjoint(self) -> Operator:
    return Operator(np.conj(self.transpose()))

  def dump_(
    self, description: str | None = None, decimals: int = 2, zeros: bool = False
  ) -> None:
    result = ""
    if description is not None:
      result += f"{description} ({self.num_bits}-qubits operator\n"

    for row in range(self.shape[0]):
      for col in range(self.shape[1]):
        value = self[row, col]
        result += f"{value.real:+.{decimals}f}{value.imag:+.{decimals}f}j "

      result += "\n"

      if not zeros:
        result = result.replace("+0.0j", " ")
        result = result.replace("+0.0", " - ")
        result = result.replace("-0.0", " - ")
        result = result.replace("+", " ")

      print(result)


def trace_out_single(density: Operator, index: int) -> Operator:
  num_bits = int(np.log2(density.shape[0]))
  if index > num_bits:
    raise AssertionError("Invalid index")

  eye = identity_gate()
  zero = Operator(np.array((1.0, 0.0)))
  one = Operator(np.array((0.0, 1.0)))

  p0 = tensor.Tensor(np.array([1.0]))
  p1 = tensor.Tensor(np.array([1.0]))

  for idx in range(num_bits):
    if idx == index:
      p0 = tensor.Tensor(p0 * zero)
      p1 = tensor.Tensor(p1 * one)
    else:
      p0 = tensor.Tensor(p0 * eye)
      p1 = tensor.Tensor(p1 * eye)

  density_0 = p0 @ density
  density_0 = density_0 @ p0.transpose()
  density_1 = p1 @ density
  density_1 = density_1 @ p1.transpose()
  reduced_density = density_0 + density_1

  return reduced_density


def trace_out(density: Operator, indices: list[int]) -> Operator:
  for idx, val in enumerate(indices):
    num_bits = int(np.log2(density.shape[0]))
    if val > num_bits:
      raise AssertionError("Invalid index")

    density = trace_out_single(density, val)

    for i in range(idx + 1, len(indices)):
      indices[i] -= 1

  return density


def measure(
  state_: state.State, target: int, to_state: int = 0, collapse: bool = True
) -> tuple[float, state.State]:
  density = state_.density()
  op = (
    projector_gate(state.zeros(1)) if to_state == 0 else projector_gate(state.ones(1))
  )

  if target > 0:
    op = Operator(identity_gate().kpow(target) * op)
  if target < state_.num_bits - 1:
    op = Operator(op * identity_gate().kpow(state_.num_bits - target - 1))

  prob0 = np.trace(np.matmul(op, density))

  if collapse:
    mvmul = np.dot(op, state_)
    divisor = np.real(np.linalg.norm(mvmul))
    if divisor > 1e-10:
      normed = mvmul / divisor
    else:
      raise AssertionError("Measurement collapsed to 0 probability state")
    return np.real(prob0), state.State(normed)

  return np.real(prob0), state_


def identity_gate(num_bits: int = 1) -> Operator:
  return Operator(Operator(np.array([[1.0, 0.0], [0.0, 1.0]])).kpow(num_bits))


def pauli_x_gate(num_bits: int = 1) -> Operator:
  return Operator(Operator(np.array([[0.0, 1.0], [1.0, 0.0]])).kpow(num_bits))


def pauli_y_gate(num_bits: int = 1) -> Operator:
  return Operator(Operator(np.array([[0.0, -1.0j], [1.0j, 0.0]])).kpow(num_bits))


def pauli_z_gate(num_bits: int = 1) -> Operator:
  return Operator(Operator(np.array([[1.0, 0.0], [0.0, -1.0]])).kpow(num_bits))


_PAULI_X = pauli_x_gate()
_PAULI_Y = pauli_y_gate()
_PAULI_Z = pauli_z_gate()


def rotation_gate(vector: np.ndarray, theta: float) -> Operator:
  vector = np.asarray(vector)
  if (vector.shape != (3,) or not np.isclose(vector @ vector, 1.0)) or not np.all(
    np.isreal(vector)
  ):
    raise ValueError("Rotation vector must be 3D real unit vector")

  return Operator(
    np.cos(theta / 2) * identity_gate()
    - 1j
    * np.sin(theta / 2)
    * (vector[0] * _PAULI_X + vector[1] * _PAULI_Y + vector[2] * _PAULI_Z)
  )


def rotation_x_gate(theta: float) -> Operator:
  return rotation_gate(np.array([1.0, 0.0, 0.0]), theta)


def rotation_y_gate(theta: float) -> Operator:
  return rotation_gate(np.array([0.0, 1.0, 0.0]), theta)


def rotation_z_gate(theta: float) -> Operator:
  return rotation_gate(np.array([0.0, 0.0, 1.0]), theta)


def phase_gate(angle: float, num_bits: int = 1) -> Operator:
  return Operator(
    Operator(np.array([[1.0, 0.0], [0.0, np.exp(1j * angle)]])).kpow(num_bits)
  )


def s_gate(num_bits: int = 1) -> Operator:
  return Operator(Operator(np.array([[1.0, 0.0], [0.0, 1.0j]])).kpow(num_bits))


def discrete_phase_gate(k: int, num_bits: int = 1) -> Operator:
  return Operator(
    Operator(np.array([[1.0, 0.0], [0.0, np.exp(2.0 * np.pi * 1j / 2**k)]])).kpow(
      num_bits
    )
  )


def projector_gate(state: state.State) -> Operator:
  """Projector for a given state."""

  return Operator(state.density())


def v_gate(num_bits: int = 1) -> Operator:
  """V gate is sqrt(X)"""
  return Operator(
    Operator(0.5 * np.array([[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]])).kpow(
      num_bits
    )
  )


def t_gate(num_bits: int = 1) -> Operator:
  """T gate is sqrt(S)"""
  return Operator(
    Operator(np.array([[1.0, 0.0], [0.0, np.exp(np.pi * 1j / 4)]])).kpow(num_bits)
  )


def y_root_gate(num_bits: int = 1) -> Operator:
  return Operator(
    Operator(
      0.5 * np.array([[1.0 + 1.0j, -1.0 - 1.0j], [1.0 + 1.0j, 1.0 + 1.0j]])
    ).kpow(num_bits)
  )


def hadamard_gate(num_bits: int = 1) -> Operator:
  return Operator(
    Operator(1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]])).kpow(num_bits)
  )


def controlled_gate(control: int, target: int, op: Operator) -> Operator:
  """Controlled gate with specified control and target qubits."""

  if control == target:
    raise ValueError("Control and target qubits must be different")

  p0 = projector_gate(state.zeros(1))  # |0><0|
  p1 = projector_gate(state.ones(1))  # |1><1|

  # space between qubits
  i_fill = identity_gate(abs(target - control) - 1)

  # width of operator in terms of identity gates
  op_fill = identity_gate().kpow(op.num_bits)

  if target > control:
    if target - control > 1:
      c_operator = p0 * i_fill * op_fill + p1 * i_fill * op
    else:
      c_operator = p0 * op_fill + p1 * op
  else:
    if control - target > 1:
      c_operator = op_fill * i_fill * p0 + op * i_fill * p1
    else:
      c_operator = op_fill * p0 + op * p1

  return c_operator


def cnot_gate(control: int = 0, target: int = 1) -> Operator:
  return controlled_gate(control, target, _PAULI_X)


def toffoli_gate(control1: int, control2: int, target: int) -> Operator:
  cnot = cnot_gate(control2, target)
  return controlled_gate(control1, control1, cnot)


def swap_gate(qubit1: int, qubit2: int) -> Operator:
  """Apply a SWAP gate between two qubits."""

  return Operator(
    cnot_gate(qubit2, qubit1) @ cnot_gate(qubit1, qubit2) @ cnot_gate(qubit2, qubit1)
  )


def quantum_fourier_transform(num_bits: int) -> Operator:
  qft = identity_gate(num_bits)
  h = hadamard_gate()

  for j in range(num_bits):
    qft = cast(Operator, qft(h, j))

    for k in range(2, num_bits - j + 1):
      control = j + k - 1
      qft = cast(Operator, qft(controlled_gate(control, j, discrete_phase_gate(k)), j))

  for j in range(num_bits // 2):
    qft = cast(Operator, qft(swap_gate(j, num_bits - j - 1), j))

  assert qft.is_unitary()

  return qft


def inverse_quantum_fourier_transform(num_bits: int) -> Operator:
  return quantum_fourier_transform(num_bits).adjoint()
