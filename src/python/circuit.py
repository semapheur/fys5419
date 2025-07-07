from enum import Enum
from typing import cast, Protocol, runtime_checkable

import numpy as np
from scipy.linalg import sqrtm

import gate
from qubit import NQubitState, QubitRegister


class Operation(Enum):
  """Enum class for operation types in a quantum circuit"""

  UNKNOWN = 0
  UNARY = 1
  CONTROLLED = 2
  SECTION = 3
  END_SECTION = 4


class Node:
  """Represents a node in the quantum circuit intermediate representation"""

  def __init__(
    self,
    operation: Operation,
    name: str | None = None,
    qubit_indices: tuple[int, ...] | None = None,
    gate: gate.Gate | None = None,
    parameter: float | None = None,
  ):
    self.operation = operation
    self.name = name
    self.qubit_indices = qubit_indices or ()
    self.gate = gate
    self.parameter = parameter

    self._validate()

  def _validate(self):
    if self.operation == Operation.UNARY and len(self.qubit_indices) != 1:
      raise ValueError("Single qubit gate must have a single qubit index")

    elif self.operation == Operation.CONTROLLED and len(self.qubit_indices) != 2:
      raise ValueError("Controlled gate must have a control and a target qubit index")

    elif (
      self.operation in (Operation.UNARY, Operation.CONTROLLED) and self.gate is None
    ):
      raise ValueError("Gate operations must have an associated gate")

  def __str__(self) -> str:
    if self.is_unary():
      result = f"{self.name}({self.qubit_indices[0]})"

    elif self.is_controlled():
      result = (
        f"{self.name}(control={self.qubit_indices[0]}, target={self.qubit_indices[1]})"
      )

    elif self.is_section():
      result = f"|-- {self.name} --"

    elif self.is_end_section():
      result = ""

    else:
      return f"UNKNOWN({self.name})"

    if self.parameter is not None:
      result += f"({self.parameter})"

    return result

  def to_controlled(self, control: int) -> "Node":
    if not self.is_unary():
      raise ValueError("Only single qubit gates can be converted to controlled gates")

    return Node(
      operation=Operation.CONTROLLED,
      name=f"c{self.name}",
      qubit_indices=(control, self.qubit_indices[0]),
      gate=self.gate,
      parameter=self.parameter,
    )

  def is_unary(self) -> bool:
    return self.operation == Operation.UNARY

  def is_controlled(self) -> bool:
    return self.operation == Operation.CONTROLLED

  def is_gate(self) -> bool:
    return self.is_unary() or self.is_controlled()

  def is_section(self) -> bool:
    return self.operation == Operation.SECTION

  def is_end_section(self) -> bool:
    return self.operation == Operation.END_SECTION


class IR:
  """Intermediate representation of a quantum circuit"""

  def __init__(self) -> None:
    self.nodes: list[Node] = []
    self.registers: list[tuple[int, str | None, int]] = []
    self.registerset: list[tuple[str | None, int, QubitRegister]] = []
    self.total_qubits = 0

  def __str__(self) -> str:
    nesting = 0
    s = ""
    for node in self.nodes:
      if node.is_section():
        nesting += 1
      if node.is_end_section():
        nesting -= 1
        continue

      s = s + (" " * nesting) + str(node) + "\n"
    return s

  @property
  def num_nodes(self) -> int:
    return len(self.nodes)

  @property
  def num_registers(self) -> int:
    return len(self.registers)

  def register(self, size: int, name: str | None, register: QubitRegister):
    self.registerset.append((name, size, register))
    for i in range(size):
      self.registers.append((self.num_registers + i, name, i))

  def add_node(self, node: Node):
    self.nodes.append(node)

  def add_unary_gate(
    self, name: str | None, qubit: int, gate: gate.Gate, value: float | None = None
  ):
    self.add_node(Node(Operation.UNARY, name, (qubit,), gate, value))

  def add_controlled_gate(
    self,
    name: str | None,
    control: int,
    target: int,
    gate: gate.Gate,
    value: float | None = None,
  ):
    self.add_node(Node(Operation.CONTROLLED, name, (control, target), gate, value))

  def begin_section(self, description: str):
    self.add_node(Node(Operation.SECTION, description))

  def end_section(self):
    self.add_node(Node(Operation.END_SECTION))


@runtime_checkable
class UnaryGateCallable(Protocol):
  def __call__(self, idx: int | QubitRegister, cond: bool = True): ...


@runtime_checkable
class ControlledGateCallable(Protocol):
  def __call__(
    self, control: int | QubitRegister, target: int | QubitRegister, cond: bool = True
  ): ...


class QuantumCircuit:
  """A class for representing quantum circuits as numpy arrays.

  Based on Hundt (2022) Quantum Computing for Programmers
  (https://doi.org/10.1017/9781009099974.004)
  https://github.com/qcc4cp/qcc/
  """

  x: UnaryGateCallable
  xdag: UnaryGateCallable
  cx: ControlledGateCallable
  cxdag: ControlledGateCallable
  h: UnaryGateCallable
  hdag: UnaryGateCallable
  ch: ControlledGateCallable
  chdag: ControlledGateCallable

  def __init__(self, name: str | None = None, eager: bool = True) -> None:
    self.name = name
    self.state = NQubitState([1.0])
    self.ir = IR()
    self.eager = eager

    self._add_simple_gates()

  def _add_simple_gates(self):
    simple_gates = {
      "x": gate.pauli_x_gate(),
      "h": gate.hadamard_gate(),
    }

    for k, v in simple_gates.items():
      self.add_unary_gate(k, v)
      self.add_unary_gate(f"{self.name}dag", v.adjoint())
      self.add_controlled_gate(f"c{k}", v)
      self.add_controlled_gate(f"c{k}dag", v.adjoint())

  @property
  def num_qubits(self) -> int:
    return self.state.num_qubits

  class scope:
    def __init__(self, ir_scope: IR, desc: str):
      self.ir = ir_scope
      self.desc = desc

    def __enter__(self):
      self.ir.begin_section(self.desc)

    def __exit__(self, exc_type, exc_value, traceback):
      self.ir.end_section()

  def product_state(self, new_state: NQubitState):
    self.state = cast(NQubitState, self.state * new_state)

  def add_register(
    self, initial_state: NQubitState, name: str | None = None
  ) -> QubitRegister:
    reg = QubitRegister(self.num_qubits, initial_state)
    self.product_state(reg.state)
    self.ir.register(len(reg), name, reg)
    return reg

  def add_unary_gate(self, name: str, gate: gate.Gate):
    setattr(
      self,
      name,
      lambda idx, cond=True: self.apply_unary_gate(gate, idx, name) if cond else None,
    )

  def add_controlled_gate(self, name: str, gate: gate.Gate):
    setattr(
      self,
      name,
      lambda control, target, cond=True: self.apply_controlled_gate(
        gate, control, target, name
      )
      if cond
      else None,
    )

  def add_circuit(self, circuit: "QuantumCircuit", offset: int = 0):
    for node in circuit.ir.nodes:
      if node.is_unary():
        self.apply_unary_gate(
          cast(gate.Gate, node.gate),
          node.qubit_indices[0] + offset,
          node.name,
          value=node.parameter,
        )
      if node.is_controlled():
        self.apply_controlled_gate(
          cast(gate.Gate, node.gate),
          node.qubit_indices[0] + offset,
          node.qubit_indices[1] + offset,
          node.name,
          value=node.parameter,
        )

  def validate_qubit_indices(self, indices: tuple[int, ...] | QubitRegister):
    num_qubits = self.num_qubits
    for i in indices:
      if i < 0 or i >= num_qubits:
        raise ValueError(f"Qubit index must be between 0 and {num_qubits - 1}. Got {i}")

  def apply_unary_gate(
    self,
    gate: gate.Gate,
    idx: int | QubitRegister,
    name: str | None,
    *,
    value: float | None = None,
  ) -> None:
    idx_ = (idx,) if isinstance(idx, int) else idx

    num_qubits = self.num_qubits
    for i in idx_:
      if i > num_qubits:
        raise ValueError(
          f"Qubit index must be between 0 and {num_qubits - 1}. Got {idx}"
        )

      if not self.eager:
        self.ir.add_unary_gate(name, i, gate, value)

      else:
        self.state.apply_unary_gate(gate, i)

  def apply_controlled_gate(
    self,
    gate: gate.Gate,
    control: int | QubitRegister,
    target: int | QubitRegister,
    name: str | None,
    *,
    value: float | None = None,
    control_by_zero: bool = False,
  ) -> None:
    if isinstance(control, QubitRegister):
      assert len(control) == 1, "Controlled multi-qubit gate not supported"
      control = cast(int, control[0])

    if isinstance(target, QubitRegister):
      assert len(target) == 1, "Controlled multi-qubit gate not supported"
      target = cast(int, target[0])

    self.validate_qubit_indices((control, target))

    if control == target:
      raise ValueError("Control and target qubits must be different")

    self.x(control, control_by_zero)
    if not self.eager:
      self.ir.add_controlled_gate(name, control, target, gate, value)

    else:
      self.state.apply_controlled_gate(gate, control, target)

    self.x(control, control_by_zero)

  def run(self):
    if self.eager:
      return

    self.eager = True
    self.add_circuit(self)

  def max_probability(self, *registers: QubitRegister | range):
    qubit_indices: list[int] = []

    for reg in registers:
      qubit_indices.extend(reg)

    max_state, max_prob = self.state.get_max_probability(qubit_indices)

    return max_state, max_prob

  def cx0(self, control: int, target: int):
    x = gate.pauli_x_gate()
    self.apply_unary_gate(x, control, "x")
    self.apply_controlled_gate(x, control, target, "cx")
    self.apply_unary_gate(x, control, "x")

  def cu(self, control: int, target: int, gate: gate.Gate, desc: str | None = None):
    if gate.shape != (2, 2):
      raise ValueError(f"Gate must be a 2x2 matrix. Got {gate.shape}")

    self.apply_controlled_gate(gate, control, target, desc)

  def ccu(
    self,
    control1: int,
    control2: int,
    target: int,
    gate_: gate.Gate,
    desc: str | None = None,
  ):
    with self.scope(self.ir, f"CC{gate_.name}\\{desc}({control1},{control2},{target})"):
      self.x(control1, False)
      self.x(control2, False)

      v = gate.Gate(sqrtm(gate))
      self.cu(control1, target, v, gate_.name + "^{1/2}")
      self.cx(control1, control2)
      self.cu(control2, target, v.adjoint(), gate_.name + "^t")
      self.cx(control1, control2)
      self.cu(control2, target, v, gate_.name + "^{1/2}")

      self.x(control2, False)
      self.x(control1, False)

  def ccx(self, control1: int, control2: int, target: int):
    self.ccu(control1, control2, target, gate.pauli_x_gate(), "ccx")

  def p(self, target: int, phase: float):
    self.apply_unary_gate(gate.phase_gate(phase), target, "p", value=phase)

  def cp(self, control: int, target: int, phase: float):
    self.apply_controlled_gate(
      gate.phase_gate(phase), control, target, "cp", value=phase
    )

  def ccp(self, control1: int, control2: int, target: int, phase: float):
    self.ccu(control1, control2, target, gate.phase_gate(phase), "ccp")

  def swap(self, target1: int, target2: int):
    with self.scope(self.ir, f"swap({target1}, {target2})"):
      self.cx(target2, target1)
      self.cx(target1, target2)
      self.cx(target2, target1)

  def cswap(self, control: int, target1: int, target2: int):
    with self.scope(self.ir, f"cswap({control}, {target1}, {target2})"):
      self.cx(target2, target1)
      self.ccx(control, target1, target2)
      self.cx(target2, target1)

  def flip(self, register: QubitRegister):
    for i in range(len(register) // 2):
      self.swap(register[i], register[len(register) - 1 - i])

  def qft(self, register: QubitRegister, inverse: bool = False, flip: bool = True):
    for j in range(len(register)):
      self.h(register[j])
      for k in range(j + 1, len(register)):
        phase = np.pi / (1 << (k - j))  # np.pi / (2 ** (k - j))

        if inverse:
          phase = -phase

        self.cp(register[k], register[j], phase)

    if flip:
      self.flip(register)

  def stats(self) -> str:
    return f"""Circuit statistics
      Qubits: {self.num_qubits}
      Gates: {self.ir.num_nodes}        
    """
