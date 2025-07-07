from __future__ import annotations
from collections import Counter
from itertools import chain
from typing import Literal

from numba import jit
import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit

from bitutils import bits_to_decimal, Bits
from tensor import Tensor, COMPLEX_DTYPE
from typehints import PolarAngle, AzimuthalAngle

BELL_STATES = [
  np.array([1, 0, 0, 1]) / np.sqrt(2),  # |Φ+⟩
  np.array([0, 1, 1, 0]) / np.sqrt(2),  # |Ψ+⟩
  np.array([1, 0, 0, -1]) / np.sqrt(2),  # |Φ-⟩
  np.array([0, 1, -1, 0]) / np.sqrt(2),  # |Ψ-⟩
]


class Qubit:
  """Class representing a single qubit as a 2D complex vector:
  |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩

  Attributes:
    _state (NDArray[np.complexfloating]): The quantum state vector of the qubit
  """

  def __init__(self, theta: PolarAngle, phi: AzimuthalAngle):
    """Initialize a qubit with given angles in the Bloch sphere.

    Args:
      theta (float): Polar angle in radians [0, π]
      phi (float): Azimuthal angle in radians [0, 2π)
    """

    if theta == 0:  # Explicitly set |0> state
      state = np.array([1, 0], dtype=COMPLEX_DTYPE)
    elif theta == np.pi:  # Explicitly set |1> state
      state = np.array([0, np.exp(1j * phi)], dtype=COMPLEX_DTYPE)
    else:  # General case
      state = np.array(
        [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=COMPLEX_DTYPE
      )

    self.state = state

  @property
  def state(self) -> NDArray[np.complexfloating]:
    return self._state

  @state.setter
  def state(self, state: NDArray[np.complexfloating]):
    self._state = state

  def get_bloch_angles(self) -> tuple[PolarAngle, AzimuthalAngle]:
    """Get the Bloch sphere angles (θ, φ) representing the current state.

    Returns:
      Tuple of (theta, phi) in radians
    """
    theta = 2.0 * np.arccos(np.abs(self.state[0]))
    if np.isclose(theta, 0) or np.isclose(theta, np.pi):
      phi = 0  # Convention: φ is undefined at poles, set to 0
    else:
      phi = (np.angle(self.state[1]) - np.angle(self.state[0])) % (2.0 * np.pi)
    return theta, phi

  def get_probabilities(self) -> NDArray[np.float64]:
    """Get the probabilities of the qubit state.

    Returns:
      Array of probabilities [p_0, p_1]
    """
    return np.square(np.abs(self.state))

  def collapse(self):
    """Collapse the qubit state to a measurement outcome."""

    probabilities = self.get_probabilities()
    outcome = np.random.choice((0, 1), p=probabilities)
    state = [1.0, 0.0] if outcome == 0 else [0.0, 1.0]
    self.state = np.array(state, dtype=COMPLEX_DTYPE)

  def measure(self, shots: int) -> list[int]:
    """Simulate measurements on the qubit.

    Args:
      shots (int): Number of measurements to perform

    Returns:
      List of measured outcomes
    """

    return [int(np.random.choice(2, p=self.get_probabilities())) for _ in range(shots)]

  def x_gate(self):
    """Apply Pauli-X gate: Bit flip."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    self.state = x @ self.state

  def y_gate(self):
    """Apply Pauli-Y gate."""
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    self.state = y @ self.state

  def z_gate(self):
    """Apply Pauli-Z gate: Phase flip."""
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    self.state = z @ self.state

  def dump_state(self, basis: Literal["x", "y", "z"] = "z", decimals: int = 3) -> str:
    """Return a string representation of the current qubit in the specified basis."""
    if basis == "x":
      x = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
      state = x @ self.state
      basis_states = ("|+⟩", "|-⟩")

    elif basis == "y":
      y = np.array([[1.0, -1.0j], [1.0j, 1.0]]) / np.sqrt(2)
      state = y @ self.state
      basis_states = ("|i⟩", "|-i⟩")

    elif basis == "z":
      state = self.state
      basis_states = ("|0⟩", "|1⟩")

    return f"({state[0]:.{decimals}f}){basis_states[0]} + ({state[1]:.{decimals}f}){basis_states[1]}"

  def hadamard_gate(self):
    """Apply Hadamard gate: Creates superposition."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    self.state = hadamard @ self.state

  def phase_gate(self, alpha: float):
    """
    Apply phase gate with given angle.

    Args:
      alpha (float): Phase angle in radians
    """
    phase = np.array([[1.0, 0.0], [0.0, np.exp(1.0j * alpha)]])
    self.state = phase @ self.state

  def s_gate(self):
    """Apply the S gate."""
    s = np.array([[1.0, 0.0], [0.0, 1.0j]])
    self.state = s @ self.state

  def bloch_str(self, decimals: int = 3) -> str:
    """Return a string representation of the qubit in Bloch angles."""
    theta, phi = self.get_bloch_angles()
    rad2deg = 180 / np.pi
    return f"θ={theta * rad2deg:.{decimals}f}, φ={phi * rad2deg:.{decimals}f}"

  def __eq__(self, other: object) -> bool:
    """Check if two qubits have equal states."""
    if not isinstance(other, Qubit):
      return NotImplemented

    return np.allclose(self.state, other.state)

  def __repr__(self):
    """Return a string representation of the qubit state"""
    return f"Qubit(state={self.state})"


class NQubitState(Tensor):
  """Represents an n-qubit state vector as a numpy array.

  The computational basis states are indexed in binary order, e.g. for 3 qubits:
  |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩

  Based on Hundt (2022) Quantum Computing for Programmers
  (https://doi.org/10.1017/9781009099974.004)
  https://github.com/qcc4cp/qcc/

  Attributes:
    num_qubits (int): Number of qubits in the system
    dim (int): Dimension of the quantum state vector
    state (NDArray[np.complexfloating]): The quantum state vector of dimension 2^n
  """

  def __new__(cls, vector: ArrayLike) -> NQubitState:
    """Initialize an n-qubit system.

    Args:
      vector (NDArray[np.complexfloating]): Initial state vector of dimension 2^n

    Returns:
      NQubitState: Initialized n-qubit system
    """

    array = np.asarray(vector, dtype=COMPLEX_DTYPE, copy=True)
    array = cls.validate_state(array)

    return array.view(cls)

  def __array_finalize__(self, obj):
    if obj is None:
      return

  @property
  def state(self) -> NDArray[np.complexfloating]:
    return self.view(np.ndarray)

  @state.setter
  def state(self, new_state: NDArray[np.complexfloating]):
    new_state = np.asarray(new_state.copy(), dtype=COMPLEX_DTYPE)
    if new_state.shape != self.shape:
      raise ValueError(
        f"State vector must have shape {self.shape}. Got {new_state.shape}"
      )
    norm = np.linalg.norm(new_state)
    if not np.isclose(norm, 1.0, rtol=1e-6):
      raise ValueError(f"State vector must be normalized. Norm = {norm}")
    self[:] = new_state

  @property
  def dim(self) -> int:
    return self.shape[0]

  @staticmethod
  def validate_state(
    vector: NDArray[np.complexfloating], rtol: float = 1e-6
  ) -> NDArray[np.complexfloating]:
    if vector.ndim != 1:
      raise ValueError(f"Qubit state must be 1D vector. Got {vector.shape}")

    dim = vector.shape[0]
    if dim == 0:
      raise ValueError("State vector cannot be empty")

    if (dim & (dim - 1)) != 0:
      raise ValueError(f"State vector must be a power of 2. Got {dim}")

    norm = np.linalg.norm(vector)
    if not np.isclose(norm, 1.0, rtol=rtol):
      raise ValueError(
        f"State vector must be normalized. Got state vector with norm {norm}"
      )

    return vector

  def amplitude(self, bits: Bits) -> np.complexfloating:
    """Get the amplitude of a given bitstring."""

    return self[bits_to_decimal(bits)]

  def get_probability(self, bits: Bits) -> float:
    """Get the probability of a given bitstring."""

    amplitude = self.amplitude(bits)
    return np.real(amplitude * np.conj(amplitude))

  def get_probabilities(
    self, register: QubitRegister | list[int] | None = None
  ) -> NDArray[np.float64]:
    """Get the probabilities of the qubit state for a given register.

    Args:
      register (QubitRegister | list[int] | None): Qubits to measure. If None, returns the full probability distribution.

    Returns:
      NDArray[np.float64]: Probability distribution of the qubit state
    """

    if register is None:
      return np.real(self.conj() * self)

    num_qubits = self.num_qubits
    for q in register:
      if q < 0 or q >= num_qubits:
        raise ValueError(f"Qubit index {q} is out of range for {num_qubits} qubits")

    if isinstance(register, list):
      register = sorted(register)

    register_probs = np.zeros(2 ** len(register))

    for i in range(self.dim):
      register_state = 0

      for j, q in enumerate(register):
        bit = (i >> q) & 1
        register_state |= bit << j

      register_probs[register_state] += np.abs(self[i]) ** 2

    return register_probs

  def get_max_probability(
    self, register: QubitRegister | list[int] | None = None
  ) -> tuple[int, float]:
    """Return the index and probability of the most probable state in the given register.

    Args:
      register (QubitRegister | list[int] | None): Qubits to measure. If None, returns the full probability distribution.

    Returns:
      tuple[int, float]: Index and probability of the most probable state
    """

    probabilities = self.get_probabilities(register)
    max_idx = np.argmax(probabilities).astype(int)
    max_prob = probabilities[max_idx].astype(float)
    return max_idx, max_prob

  def get_reduced_density_matrix(
    self, target_qubits: list[int]
  ) -> NDArray[np.complexfloating]:
    """Calculate the reduced density matrix for specified qubits.

    Args:
      target_qubits (list[int]): List of qubit indices to keep (others are traced out)

    Returns:
      NDArray[np.complexfloating]: Reduced density matrix for the specified qubits
    """

    # Convert state vector to density matrix
    density_matrix = np.outer(self, np.conj(self))

    # Reshape into tensor with 2^n dimensions
    tensor_shape = [2] * (2 * self.num_qubits)  # twice number of qubits for bra and ket
    tensor = density_matrix.reshape(tensor_shape)

    # Determine which qubits to trace out
    trace_qubits = [i for i in range(self.num_qubits) if i not in target_qubits]

    # Trace out the specified qubits
    for qubit in sorted(trace_qubits, reverse=True):
      ket_index = qubit
      bra_index = qubit + self.num_qubits
      reduced_density = np.trace(tensor, axis1=ket_index, axis2=bra_index)

    return reduced_density

  def get_entanglement_entropy(self, partition: list[int]) -> float:
    """Calculate the entanglement entropy for a given partition of qubits.

    Args:
      partition (list[int]): List of qubit indices for the partition

    Returns:
      float: Linear entropy of entanglement (0 for unentangled, approaches 1 for maximally entangled)
    """
    reduced_density = self.get_reduced_density_matrix(partition)
    purity = np.real(np.trace(reduced_density @ reduced_density))
    return 1.0 - purity

  def is_entangled(self, partition: list[int] | None = None) -> bool:
    """Check if qubits are entangled across a given partition.

    Args:
      partition (list[int]|None): List of qubit indices for first subsystem (default: [0])

    Returns:
      bool: True if the partition is entangled with its complement
    """

    if partition is None:
      partition = [0]
    entropy = self.get_entanglement_entropy(partition)
    return not np.isclose(entropy, 0.0, rtol=1e-6)

  def measure(
    self, targets: list[int], shots: int = 1, as_dict: bool = True
  ) -> dict[str, int] | list[tuple[int, ...]]:
    """Perform sequential measurements on specified qubits.

    Args:
      target (list[int]): List of qubit indices to measure in sequence
      shots (int): Number of measurements to perform

    Returns:
      dict[str, int] | list[tuple[int, ...]]: Measurement outcomes
    """

    # Generate measurement probabilities for each shot
    random_outcomes = np.random.random(size=(shots, len(targets)))

    outcomes = _measure_shots(self.copy(), targets, random_outcomes, self.dim)

    result = [tuple(outcome) for outcome in outcomes]
    if not as_dict:
      return result

    outcome_strings = ["".join(map(str, outcome)) for outcome in result]
    return dict(sorted(Counter(outcome_strings).items()))

  def measure_simple(
    self, shots: int, register: QubitRegister | list[int] | None = None
  ) -> dict[str, int]:
    probs = self.get_probabilities(register)

    outcomes = np.random.choice(len(probs), size=(shots), p=probs)
    counts: dict[str, int] = {}

    for o in outcomes:
      bitstring = format(o, f"0{self.num_qubits}b")
      counts[bitstring] = counts.get(bitstring, 0) + 1

    return dict(sorted(counts.items()))

  def apply_unary_gate(self, gate: np.ndarray, target: int):
    # Reverse target index
    target = self.num_qubits - target - 1
    target_stride = 1 << target  # 2**target

    # Gate elements
    g00 = gate[0, 0]
    g01 = gate[0, 1]
    g10 = gate[1, 0]
    g11 = gate[1, 1]

    target_block = 1 << (target + 1)  # 2**(target + 1)
    for g in range(0, self.dim, target_block):
      for i in range(g, g + target_stride):
        t1 = g00 * self[i] + g01 * self[i + target_stride]
        t2 = g10 * self[i] + g11 * self[i + target_stride]
        self[i] = t1
        self[i + target_stride] = t2

  def apply_controlled_gate(
    self, gate: NDArray[np.complexfloating], control: int, target: int
  ):
    """Apply a controlled single-qubit gate.

    Args:
      gate (NDArray[np.complexfloating]): 2x2 unitary matrix representing the gate
      control (int): Index of control qubit
      target (int): Index of target qubit"""

    if gate.shape != (2, 2):
      raise ValueError("Gate must be a 2x2 matrix")

    self_qubits = self.num_qubits

    if control < 0 or control >= self_qubits:
      raise ValueError(f"Control index must be between 0 and {self_qubits - 1}")

    if target < 0 or target >= self_qubits:
      raise ValueError(f"Target index must be between 0 and {self_qubits - 1}")

    if control == target:
      raise ValueError("Control and target qubits must be different")

    # Reverse control and target indices
    target = self_qubits - target - 1
    target_stride = 1 << target  # 2**target
    control = self_qubits - control - 1

    # Gate elements
    g00 = gate[0, 0]
    g01 = gate[0, 1]
    g10 = gate[1, 0]
    g11 = gate[1, 1]

    target_block = 1 << (target + 1)  # 2**(target + 1)
    for g in range(0, self.dim, target_block):
      index_base = g * self.dim
      for i in range(g, g + target_stride):
        index = index_base + i
        if index & (1 << control):  # Apply gate if control qubit is 1
          t1 = g00 * self[i] + g01 * self[i + target_stride]
          t2 = g10 * self[i] + g11 * self[i + target_stride]
          self[i] = t1
          self[i + target_stride] = t2

  def hadamard(self, target: int):
    """Apply a Hadamard gate to the specified qubit.

    Args:
      target (int): Index of target qubit"""

    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=COMPLEX_DTYPE) / np.sqrt(2.0)
    self.apply_unary_gate(h, target)

  def pauli_x(self, target: int):
    """Apply a Pauli-X gate to the specified qubit.

    Args:
      target (int): Index of target qubit"""

    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=COMPLEX_DTYPE)
    self.apply_unary_gate(x, target)

  def pauli_y(self, target: int):
    """
    Apply a Pauli-Y gate to the specified qubit.

    Args:
      target (int): Index of target qubit"""

    y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    self.apply_unary_gate(y, target)

  def pauli_z(self, target: int):
    """Apply a Pauli-Z gate to the specified qubit(s).

    Args:
      target (int): Index of target qubit"""

    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=COMPLEX_DTYPE)
    self.apply_unary_gate(z, target)

  def rotation_x(self, theta: float, target: int):
    """Apply a rotation around the X axis.

    Args:
      theta (float): Rotation angle in radians
      target (int): Index of target qubit"""

    R_x = np.array(
      [
        [np.cos(theta / 2.0), -1.0j * np.sin(theta / 2.0)],
        [-1.0j * np.sin(theta / 2.0), np.cos(theta / 2.0)],
      ]
    )
    self.apply_unary_gate(R_x, target)

  def rotation_y(self, theta: float, target: int):
    """Apply a rotation around the Y axis.

    Args:
      theta (float): Rotation angle in radians
      target (int): Index of target qubit"""

    R_y = np.array(
      [
        [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
        [np.sin(theta / 2.0), np.cos(theta / 2.0)],
      ]
    )
    self.apply_unary_gate(R_y, target)

  def rotation_z(self, theta: float, target: int):
    """Apply a rotation around the Z axis.

    Args:
      theta (float): Rotation angle in radians
      target (int): Index of target qubit"""

    R_z = np.array(
      [[np.exp(-1.0j * theta / 2.0), 0.0], [0.0, np.exp(1.0j * theta / 2.0)]]
    )
    self.apply_unary_gate(R_z, target)

  def s_gate(self, target: int):
    """Apply the S gate to the specified qubit(s).

    Args:
      target (int): Index of target qubit"""

    s = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=COMPLEX_DTYPE)
    self.apply_unary_gate(s, target)

  def s_dagger(self, target: int):
    """Apply the S† gate to the specified qubit(s).

    Args:
      target (int): Index of target qubit"""

    s_dagger = np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=COMPLEX_DTYPE)
    self.apply_unary_gate(s_dagger, target)

  def cnot(self, control: int, target: int):
    """
    Apply the controlled-NOT (CNOT) gate with specified control qubit.

    Args:
      control (int): Index of control qubit
      target (int): Index of target qubit"""

    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=COMPLEX_DTYPE)
    self.apply_controlled_gate(x, control, target)

  def swap(self, qubit1: int, qubit2: int):
    """Apply a SWAP gate between two qubits using three CNOT gates

    Args:
      qubit1 (int): Index of first qubit
      qubit2 (int): Index of second qubit

    Raises:
      ValueError: If qubit1 and qubit2 are the same
    """

    self_qubits = self.num_qubits
    if qubit1 < 0 or qubit1 >= self_qubits:
      raise ValueError(f"Index of qubit1 must be between 0 and {self_qubits - 1}")

    if qubit2 < 0 or qubit2 >= self_qubits:
      raise ValueError(f"Index of qubit2 must be between 0 and {self_qubits - 1}")

    if qubit1 == qubit2:
      raise ValueError("Qubits must be different")

    self.cnot(qubit1, qubit2)
    self.cnot(qubit2, qubit1)
    self.cnot(qubit1, qubit2)

  def flip(self, register: QubitRegister | None = None):
    qubit_range = register or range(self.num_qubits)
    n = len(qubit_range)

    for i in range(n // 2):
      self.swap(qubit_range[i], qubit_range[n - 1 - i])

  def controlled_phase(self, control: int, target: int, phase: float):
    """Apply a controlled phase gate.

    Args:
      control (int): Index of control qubit
      target (int): Index of target qubit
      phase (float): Phase angle in radians"""

    p = np.array([[1.0, 0.0], [0.0, np.exp(1.0j * phase)]], dtype=COMPLEX_DTYPE)
    self.apply_controlled_gate(p, control, target)

  def fourier_transform(
    self,
    register: QubitRegister | None = None,
    inverse: bool = False,
  ):
    """Apply the quantum Fourier transform (QFT) to the specified qubit register.

    Args:
      register (QubitRegister | None): The qubit register to which the QFT is applied.
        If None, the QFT is applied to all qubits in the quantum circuit.
      inverse (bool): If True, the inverse QFT is applied.
    """

    qubit_range = register or range(self.num_qubits)
    n = len(qubit_range)

    for j in qubit_range:
      self.hadamard(qubit_range[j])

      for k in range(j + 1, n):
        phase = np.pi / (1 << (k - j))  # np.pi / (2 ** (k - j))

        if inverse:
          phase = -phase

        self.controlled_phase(qubit_range[k], qubit_range[j], phase)

    # Reverse qubit order
    self.flip(register)

  def dump_state(self, decimals: int = 3) -> str:
    """Print the current state in the computational basis."""
    terms = []
    for i in range(self.dim):
      if abs(self[i]) > 1e-10:  # Ignore very small amplitudes
        basis = f"|{i:0{self.num_qubits}b}⟩"
        terms.append(f"({self[i]:.{decimals}f}){basis}")

    return " + ".join(terms)

  def __repr__(self) -> str:
    return (
      f"NQubitSystem(num_qubits={self.num_qubits}, dim={self.dim}, state={self.state})"
    )


class QubitRegister:
  def __init__(self, start_idx: int, initial_state: NQubitState):
    """Initialize a QubitRegister with given start index and initial state.

    Args:
      start_idx (int): The starting index of the qubit register.
      initial_state (NQubitState): The initial state of the qubit register.
    """

    if start_idx < 0:
      raise ValueError(f"Start index must be non-negative. Got {start_idx}")

    self._start_idx = start_idx
    self._state = initial_state.copy()

  @property
  def state(self) -> NQubitState:
    return self._state

  def __getitem__(self, idx: int):
    """Get qubit index within register"""

    size = len(self)
    if idx == -1:
      return self._start_idx + size - 1

    if idx < 0 or idx >= size:
      raise IndexError(f"Index must be between 0 and {size - 1}. Got {idx}")

    return self._start_idx + idx

  def __len__(self):
    """Get number of qubits in register"""

    return self._state.num_qubits

  def __iter__(self):
    """Get iterator over qubit indices in register"""

    return iter(range(self._start_idx, self._start_idx + len(self)))

  def __str__(self) -> str:
    """Print the current state in the computational basis."""

    return self.state.dump_state()


@jit(nopython=True)
def _measure_shots(
  state: NDArray[np.complexfloating],
  targets: list[int],
  random_outcomes: NDArray[np.float64],
  dim: int,
):
  """JIT-compiled function to perform quantum measurements.

  Args:
    state (np.ndarray[np.complexfloating]): The quantum state vector
    targets (list): List of qubit indices to measure
    random_outcomes (np.ndarray): Pre-generated random numbers
    dim (int): Dimension of the state vector

  Returns:
      np.ndarray: Measurement outcomes for all shots
  """

  shots = random_outcomes.shape[0]
  num_targets = len(targets)

  outcomes = np.zeros((shots, num_targets), dtype=np.int32)

  for shot in range(shots):
    current_state = state.copy()

    for i, target in enumerate(targets):
      # Bit mask for target qubit
      mask = 1 << target

      # Calculate probability for measuring |0⟩ on target qubit
      prob_0 = 0.0
      for j in range(dim):
        if (j & mask) == 0:  # If target qubit is |0⟩
          prob_0 += np.abs(current_state[j]) ** 2

      # Measure target qubit
      outcome = int(random_outcomes[shot, i] > prob_0)
      outcomes[shot, i] = outcome

      # Collapse state
      norm_squared = 0.0
      for j in range(dim):
        bit_value = (j & mask) >> target
        if bit_value != outcome:
          current_state[j] = 0.0
        else:
          norm_squared += np.abs(current_state[j]) ** 2

      # Normalize state
      if norm_squared > 1e-10:
        norm = np.sqrt(norm_squared)
        for j in range(dim):
          current_state[j] /= norm

  return outcomes


def extract_contiguous_qubits(basis_state: int, register: QubitRegister | range) -> int:
  """Extract contiguous qubits from a basis state, given a register of qubits.

  Args:
    basis_state (int): The basis state from which to extract the qubits
    register (QubitRegister | range): The register of qubits to extract

  Returns:
    int: The extracted qubits as a single integer
  """
  mask = (1 << len(register)) - 1
  return (basis_state >> register[0]) & mask


def extract_qubits(basis_state: int, *registers: QubitRegister | range) -> int:
  """Extract qubits from a basis state, given multiple registers of qubits.

  Args:
    basis_state (int): The basis state from which to extract the qubits
    *registers (QubitRegister | range): The registers of qubits to extract

  Returns:
    int: The extracted qubits as a single integer
  """
  indices = chain.from_iterable(registers)
  result = 0
  for i, qubit_idx in enumerate(indices):
    bit = (basis_state >> qubit_idx) & 1
    result |= bit << i

  return result


def basis_state(num_qubits: int, basis_index: int) -> NQubitState:
  """Create a quantum state in the given basis state.

  Args:
    num_qubits (int): Number of qubits in the system
    basis_index (int): Index of the basis state in binary order (0 to 2^n-1)

  Returns:
    NQubitState: Quantum state in the given basis state
  """
  dim = 1 << num_qubits  # 2**num_qubits

  if basis_index < 0 or basis_index >= dim:
    raise ValueError(f"Index must be between 0 and {dim - 1}. Got {basis_index}")

  state = np.zeros(dim, dtype=COMPLEX_DTYPE)
  state[basis_index] = 1.0

  return NQubitState(state)


def bitstring_state(*bits: Literal[0, 1]) -> NQubitState:
  """Create a quantum state in the given bitstring."""
  return basis_state(len(bits), bits_to_decimal(bits))


def random_qubit_state(dim: int) -> NDArray[np.complexfloating]:
  """Create a random qubit state."""

  if not (dim > 0 and (dim & (dim - 1)) == 0):
    raise ValueError("Dimension must be a power of 2")

  state = np.random.rand(dim) + 1.0j * np.random.rand(dim)
  state /= np.linalg.norm(state)

  return state


def create_bell_state(state: Literal[0, 1, 2, 3]) -> NQubitState:
  """
  Create a qubit in a Bell state using Hadamard and CNOT gates.

  Args:
    state (int): Bell state to initialize the qubit in. Can be 0, 1, 2, or 3.
    - 0 initializes |00⟩ resulting in |Φ+⟩ = (1/√2)*(|00⟩ + |11⟩)
    - 1 initializes |01⟩ resulting in |Ψ+⟩ = (1/√2)*(|01⟩ + |10⟩)
    - 2 initializes |10⟩ resulting in |Ψ-⟩ = (1/√2)*(|00⟩ - |11⟩)
    - 3 initializes |11⟩ resulting in |Φ-⟩ = (1/√2)*(|01⟩ - |10⟩)
  """

  bell_state = basis_state(2, state)
  bell_state.hadamard(0)  # apply Hadamard gate to first qubit
  bell_state.cnot(0, 1)  # apply CNOT gate

  return bell_state


def bell_state_circuit(state: Literal[0, 1, 2, 3]) -> QuantumCircuit:
  """
  Initialize a quantum circuit in a Bell state using Qiskit

  Args:
    state (str): Bell state to initialize the qubit in. Can be 0, 1, 2, or 3.
    - 0 initializes |00⟩ resulting in |Φ+⟩ = (1/√2)*(|00⟩ + |11⟩)
    - 1 initializes |01⟩ resulting in |Ψ+⟩ = (1/√2)*(|01⟩ + |10⟩)
    - 2 initializes |10⟩ resulting in |Ψ-⟩ = (1/√2)*(|00⟩ - |11⟩)
    - 3 initializes |11⟩ resulting in |Φ-⟩ = (1/√2)*(|01⟩ - |10⟩)
  """

  qc = QuantumCircuit(2, 2)
  qc.initialize(f"{state:0{2}b}", [0, 1])

  qc.h(0)  # apply Hadamard gate to first qubit
  qc.cx(0, 1)  # apply CNOT gate

  return qc


def dump_qubit_state(state: NDArray[np.complexfloating], decimals: int = 3) -> str:
  terms = []
  dim = len(state)
  num_qubits = int(np.log2(dim))
  for i in range(dim):
    if abs(state[i]) > 1e-10:  # Ignore very small amplitudes
      basis = f"|{i:0{num_qubits}b}⟩"
      terms.append(f"({state[i]:.{decimals}f}){basis}")

  return " + ".join(terms)
