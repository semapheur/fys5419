import random
import numpy as np

import operator
import state


class QuantumCircuit:
  """
  Based on Hundt (2022) Quantum Computing for Programmers
  (https://doi.org/10.1017/9781009099974.004)
  https://github.com/qcc4cp/qcc/
  """

  def __init__(self, name: str | None = None):
    self.name = name
    self.state = state.State(np.array([1.0]))
    self.global_reg = 0

  @property
  def num_bits(self) -> int:
    return self.state.num_bits

  def register(self, size: int, it, *, name: str | None = None):
    ret = state.Register(size, it, self.global_reg)
    self.global_reg += size
    self.state = state.State(self.state * ret.state())
    return ret

  def qubit(
    self,
    alpha: np.complexfloating | None = None,
    beta: np.complexfloating | None = None,
  ):
    self.state = state.State(self.state * state.qubit(alpha, beta))
    self.global_reg += 1

  def zeros(self, num_qubits: int):
    self.state = state.State(self.state * state.zeros(num_qubits))
    self.global_reg += num_qubits

  def ones(self, num_qubits: int):
    self.state = state.State(self.state * state.ones(num_qubits))
    self.global_reg += num_qubits

  def bit_state(self, *bits):
    self.state = state.State(self.state * state.bit_state(*bits))
    self.global_reg += len(bits)

  def arange(self, num_qubits: int):
    self.zeros(num_qubits)
    for i in range(num_qubits):
      self.state[i] = float(i)

    self.global_reg += num_qubits

  def rand(self, num_qubits: int):
    self.state = state.State(self.state * state.rand_state(num_qubits))
    self.global_reg += num_qubits

  def apply_unary_gate(
    self,
    gate: operator.Operator,
    target: int,
    name: str | None = None,
    *,
    value: float | None = None,
  ): ...

  def apply_controlled_gate(
    self,
    gate: operator.Operator,
    control: int,
    target: int,
    name: str | None = None,
    *,
    value: float | None = None,
  ): ...

  def multi_control(
    self, controls, targets, aux, gates, description: str | None = None
  ):
    """Multi-controlled gate, using aux as ancilla"""

    with self.scope(self.ir, f"multi({controls}, {targets}) # {description}"):
      if len(controls) == 0:
        self.apply_unary_gate(gates, targets, description)
        return

      if len(controls) == 1:
        self.apply_controlled_gate(gates, controls[0], targets, description)
        return

      # Compute predicate
      self.ccx(controls[0], controls[1], aux[0])
      aux_index = 0
      for i in range(2, len(controls)):
        self.ccx(controls[i], aux[aux_index], aux[aux_index + 1])
        aux_index += 1

      # Use predicate to single-control qubit at target
      self.apply_controlled_gate(gates, aux[aux_index], targets, description)

      # Uncompute predicate
      aux_index -= 1
      for i in range(len(controls) - 1, 1, -1):
        self.ccx(controls[i], aux[aux_index], aux[aux_index + 1])
        aux_index -= 1

      self.ccx(controls[0], controls[1], aux[0])

  def cv(self, control: int, target: int):
    self.apply_controlled_gate(operator.v_gate(), control, target, "cv")

  def cv_adj(self, control: int, target: int):
    self.apply_controlled_gate(operator.v_gate().adjoint(), control, target, "cv_adj")

  def cx(self, control: int, target: int):
    self.apply_controlled_gate(operator.pauli_x_gate(), control, target, "cx")

  def cy(self, control: int, target: int):
    self.apply_controlled_gate(operator.pauli_y_gate(), control, target, "cy")

  def cz(self, control: int, target: int):
    self.apply_controlled_gate(operator.pauli_z_gate(), control, target, "cx")

  def cphase(self, control: int, target: int, angle: float):
    self.apply_controlled_gate(
      operator.phase_gate(angle), control, target, "cphase", value=angle
    )

  def crk(self, k: int, control: int, target: int):
    self.apply_controlled_gate(operator.discrete_phase_gate(k), control, target, "crk")

  def ccx(self, control: int, target1: int, target2: int):
    """Sleator-Weinfurter Construction."""

    self.cv(control, target2)
    self.cx(control, target1)
    self.cv_adj(control, target2)
    self.cx(control, target1)
    self.cv(control, target2)

  def swap(self, target1: int, target2: int):
    self.cx(target2, target1)
    self.cx(target1, target2)
    self.cx(target2, target1)

  def cswap(self, control: int, target1: int, target2: int):
    self.ccx(control, target2, target1)
    self.ccx(control, target1, target2)
    self.ccx(control, target2, target1)

  def toffoli(self, control: int, target1: int, target2: int):
    self.ccx(control, target1, target2)

  def h(self, target: int):
    self.apply_unary_gate(operator.hadamard_gate(), target, "h")

  def t(self, target: int):
    self.apply_unary_gate(operator.t_gate(), target, "t")

  def phase_gate(self, target: int, angle: float):
    self.apply_unary_gate(operator.phase_gate(angle), target, "phase_gate", value=angle)

  def v(self, target: int):
    self.apply_unary_gate(operator.v_gate(), target, "v")

  def s(self, target: int):
    self.apply_unary_gate(operator.s_gate(), target, "s")

  def x(self, target: int):
    self.apply_unary_gate(operator.pauli_x_gate(), target, "x")

  def y(self, target: int):
    self.apply_unary_gate(operator.pauli_y_gate(), target, "y")

  def z(self, target: int):
    self.apply_unary_gate(operator.pauli_z_gate(), target, "z")

  def rx(self, target: int, angle: float):
    self.apply_unary_gate(operator.rotation_x_gate(angle), target, "rx", value=angle)

  def ry(self, target: int, angle: float):
    self.apply_unary_gate(operator.rotation_y_gate(angle), target, "ry", value=angle)

  def rz(self, target: int, angle: float):
    self.apply_unary_gate(operator.rotation_z_gate(angle), target, "rz", value=angle)

  def measure_qubit(
    self, target: int, to_state: int = 0, collapse: bool = True
  ) -> tuple[float, state.State]:
    return operator.measure(self.state, target, to_state, collapse)

  def pauli_expectation(self, target: int) -> float:
    p0, _ = self.measure_qubit(target, 0, collapse=False)
    return p0 - (1 - p0)

  def sample_state(self, prob_state0: float):
    if prob_state0 < random.random():
      return 1
    return 0
