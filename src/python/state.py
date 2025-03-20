import operator
from typing import Literal, cast
import numpy as np

import bitutils
from bitutils import Bits
import tensor


class State(tensor.Tensor):
  """Class representing qubit states
  Based on Hundt (2022) Quantum Computing for Programmers
  (https://doi.org/10.1017/9781009099974.004)
  """

  def __repr__(self) -> str:
    s = "State("
    return f"{s}{super().__str__().replace('\n', '\n ' * len(s))})"

  def __str__(self) -> str:
    return f"{self.num_bits}-qubit state. Tensor:\n{super().__str__()}"

  def normalize(self):
    """Renormalize the state."""

    norm_squared = np.conj(self) @ self
    if norm_squared.is_close(0.0):
      raise AssertionError("Normalizing to zero-probability state.")

    self /= np.sqrt(np.real(norm_squared))

  def amplitude(self, *bits) -> np.complexfloating:
    """Return the amplitude of the state indexed by 'bits'."""

    idx = bitutils.bits_to_decimal(*bits)
    return self[idx]

  def phase(self, *bits) -> float:
    """Compute the phase in degrees of the state indexed by 'bits'."""

    amplitude = self.amplitude(*bits)
    return np.degrees(np.angle(amplitude))

  def probability(self, *bits) -> float:
    """Return the probability of the state indexed by 'bits'."""

    amplitude = self.amplitude(*bits)
    return np.real(amplitude.conj() * amplitude)

  def max_probability(self) -> tuple[Bits, float]:
    """Return the most probable state and its probability."""

    maxbits: Bits | None = None
    maxprob = 0.0
    for bits in bitutils.bit_product(self.num_bits):
      prob = self.probability(*bits)
      if prob > maxprob:
        maxbits, maxprob = bits, prob
    return cast(Bits, maxbits), maxprob

  def density(self) -> tensor.Tensor:
    """Return the density matrix of the state."""
    return tensor.Tensor(np.outer(self, self.conj()))

  def apply_unary_gate(self, gate: operator.Operator, target: int):
    target = self.num_bits - target - 1
    pow_2_target = 1 << target
    g00 = gate[0, 0]
    g01 = gate[0, 1]
    g10 = gate[1, 0]
    g11 = gate[1, 1]

    for g in range(0, 1 << self.num_bits, 1 << (target + 1)):
      for i in range(g, g + pow_2_target):
        t1 = g00 * self[i] + g01 * self[i + pow_2_target]
        t2 = g10 * self[i] + g11 * self[i + pow_2_target]
        self[i] = t1
        self[i + pow_2_target] = t2

  def apply_controlled_gate(self, gate: operator.Operator, control: int, target: int):
    qubit = self.num_bits - target - 1
    pow_2_index = 2**qubit
    control = self.num_bits - control - 1
    g00 = gate[0, 0]
    g01 = gate[0, 1]
    g10 = gate[1, 0]
    g11 = gate[1, 1]

    for g in range(0, 1 << self.num_bits, 1 << (qubit + 1)):
      index_base = g * (1 << self.num_bits)
      for i in range(g, g + pow_2_index):
        index = index_base + i
        if index & (1 << control):
          t1 = g00 * self[i] + g01 * self[i + pow_2_index]
          t2 = g10 * self[i] + g11 * self[i + pow_2_index]
          self[i] = t1
          self[i + pow_2_index] = t2

  def dump_(
    self, description: str | None = None, probability_only: bool = True
  ) -> None:
    dump_state(self, description, probability_only)


class Register:
  def __init__(self, size: int, it=0, global_reg: int | None = None):
    self.size = size
    self.global_idx = list(range(global_reg, global_reg + size))
    self.value = [0] * size

    if it:
      if isinstance(it, int):
        it = bin(it)[2:].zfill(size)

      if isinstance(it, (str, tuple, list)):
        for idx, val in enumerate(it):
          if val == "1" or val == 1:
            self.value[idx] = 1

  @property
  def num_bits(self):
    return self.size

  def __getitem__(self, idx: int):
    return self.global_idx[idx]

  def __setitem__(self, idx: int, val: int):
    self.value[idx] = val

  def state(self) -> State:
    return bit_state(*self.value)

  def __str__(self):
    return f"|{self.value!s:0{self.size}b}⟩"


def ket_string(bits: Bits) -> str:
  """Convert a state to a ket string."""

  bit_string = "".join([str(bit) for bit in bits])
  return f"|{bit_string}⟩ ({int(bit_string, 2)})"


def dump_state(
  state: State, description: str | None = None, probability_only: bool = True
) -> None:
  """Dump printable information about a state."""

  if description is not None:
    print("|", end="")
    for i in range(state.num_bits - 1, -1, -1):
      print(i % 10, end="")
    print(f"> '{description}'")

  states: list[str] = []

  for bits in bitutils.bit_product(state.num_bits):
    if probability_only:
      continue

    states.append(
      f"{ket_string(bits)}: "
      f"ampl = {state.amplitude(*bits)} "
      f"prob: {state.probability(*bits)} "
      f"phase: {state.phase(*bits)}°"
    )

  states.sort()
  print(*states, sep="\n")


def qubit(
  alpha: np.complexfloating | None = None, beta: np.complexfloating | None = None
) -> State:
  """Create a single qubit state |ψ⟩ = α|0⟩ + β|1⟩.

  Args:
    alpha (np.complexfloating): Amplitude of |0⟩ state
    beta (np.complexfloating): Amplitude of |1⟩ state

  Returns:
    State: Single qubit quantum state

  Raises:
    ValueError: If state is not normalized or if neither amplitude is provided
  """

  if alpha is None and beta is None:
    raise ValueError("At least one of alpha or beta must be provided")

  if beta is None:
    beta = np.sqrt(1.0 - np.conj(alpha) * alpha)

  if alpha is None:
    alpha = np.sqrt(1 - np.conj(beta) * beta)

  if not np.isclose(np.conj(alpha) * alpha + np.conj(beta) * beta, 1.0):
    raise ValueError("Qubit state must be normalized")

  return State(np.array([alpha, beta], dtype=tensor.tensor_type()))


def zeros_or_ones(num_qubits: int, idx: int) -> State:
  """Create a state with all qubits set to |0⟩ or |1⟩.

  Args:
    num_qubits (int): Number of qubits in the state
    idx (int): Index of qubit to set to |1⟩

  Returns:
    State: Quantum state with all qubits set to |0⟩ or |1⟩
  """

  if num_qubits < 1:
    raise ValueError("Number of qubits must be at least 1")

  shape = 2**num_qubits
  state_tensor = np.zeros(shape, dtype=tensor.tensor_type())
  state_tensor[idx] = 1.0
  return State(state_tensor)


def zeros(num_qubits: int) -> State:
  """Create a state with all qubits set to |0⟩.

  Args:
    num_qubits (int): Number of qubits in the state

  Returns:
    State: Quantum state with all qubits set to |0⟩
  """

  return zeros_or_ones(num_qubits, 0)


def ones(num_qubits: int) -> State:
  """Create a state with all qubits set to |1⟩.

  Args:
    num_qubits (int): Number of qubits in the state

  Returns:
    State: Quantum state with all qubits set to |1⟩
  """

  return zeros_or_ones(num_qubits, 2**num_qubits - 1)


def bit_state(*bits) -> State:
  """Create a state with all qubits set to |bits⟩.

  Args:
    *bits (int): Bits to set to |1⟩

  Returns:
    State: Quantum state with all qubits set to |bits⟩
  """

  num_qubits = len(bits)
  if num_qubits == 0:
    raise ValueError("Number of qubits must be at least 1")

  state_tensor = np.zeros(1 << num_qubits, dtype=tensor.tensor_type())
  state_tensor[bitutils.bits_to_decimal(*bits)] = 1.0
  return State(state_tensor)


def rand_state(num_bits: int) -> State:
  """Create a random state.

  Args:
    num_bits (int): Number of qubits in the state

  Returns:
    State: Random quantum state
  """

  bits: Bits = [cast(Literal[0, 1], np.random.randint(0, 1)) for _ in range(num_bits)]

  return bit_state(bits)


def bell_state(a: Literal[0, 1], b: Literal[0, 1]) -> State:
  """Create a Bell state using a bit string"""

  import operator

  state = bit_state(a, b)
  state = operator.hadamard_gate()(state)
  return operator.cnot_gate()(state)


def ghz_state(num_qubits: int) -> State:
  """Create"""

  import operator

  state = zeros(num_qubits)
  state = operator.hadamard_gate()(state)
  for offset in range(num_qubits - 1):
    state = operator.cnot_gate(0, 1)(state, offset)

  return state
