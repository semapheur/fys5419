from collections import Counter
from typing import Literal

from numba import jit
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from typehints import PolarAngle, AzimuthalAngle

BELL_STATES = [
  np.array([1, 0, 0, 1]) / np.sqrt(2),  # |Φ+⟩
  np.array([0, 1, 1, 0]) / np.sqrt(2),  # |Ψ+⟩
  np.array([1, 0, 0, -1]) / np.sqrt(2),  # |Φ-⟩
  np.array([0, 1, -1, 0]) / np.sqrt(2),  # |Ψ-⟩
]


class Qubit:
  """Represents a single qubit as a 2D complex vector:
  |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩

  Attributes:
    _state (NDArray[np.complex128]): The quantum state vector of the qubit
  """

  def __init__(self, theta: PolarAngle, phi: AzimuthalAngle):
    """Initialize a qubit with given angles in the Bloch sphere.

    Args:
      theta (float): Polar angle in radians [0, π]
      phi (float): Azimuthal angle in radians [0, 2π]
    """

    if theta == 0:  # Explicitly set |0> state
      state = np.array([1, 0], dtype=np.complex128)
    elif theta == np.pi:  # Explicitly set |1> state
      state = np.array([0, np.exp(1j * phi)], dtype=np.complex128)
    else:  # General case
      state = np.array(
        [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=np.complex128
      )

    self.state = state

  @property
  def state(self) -> NDArray[np.complex128]:
    return self._state

  @state.setter
  def state(self, state: NDArray[np.complex128]):
    self._state = state

  def get_bloch_angles(self) -> tuple[PolarAngle, AzimuthalAngle]:
    """Get the Bloch sphere angles (θ, φ) representing the current state.

    Returns:
      Tuple of (theta, phi) in radians
    """
    theta = 2 * np.arccos(np.abs(self.state[0]))
    if np.isclose(theta, 0) or np.isclose(theta, np.pi):
      phi = 0  # Convention: φ is undefined at poles, set to 0
    else:
      phi = (np.angle(self.state[1]) - np.angle(self.state[0])) % (2 * np.pi)
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
    state = [1, 0] if outcome == 0 else [0, 1]
    self.state = np.array(state, dtype=np.complex128)

  def measure(self, shots: int) -> list[int]:
    """Simulate measurements on the qubit.

    Args:
      shots (int): Number of measurements to perform

    Returns:
      List of measured outcomes
    """

    return [int(np.random.choice(2, p=self.get_probabilities())) for _ in range(shots)]

  def hadamard_gate(self):
    """Apply Hadamard gate: Creates superposition."""
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    self.state = hadamard @ self.state

  def phase_gate(self, alpha: float):
    """
    Apply phase gate with given angle.

    Args:
      alpha (float): Phase angle in radians
    """
    phase = np.array([[1, 0], [0, np.exp(1j * alpha)]])
    self.state = phase @ self.state

  def x_gate(self):
    """Apply Pauli-X gate: Bit flip."""
    x = np.array([[0, 1], [1, 0]])
    self.state = x @ self.state

  def y_gate(self):
    """Apply Pauli-Y gate."""
    y = np.array([[0, -1j], [1j, 0]])
    self.state = y @ self.state

  def z_gate(self):
    """Apply Pauli-Z gate: Phase flip."""
    z = np.array([[1, 0], [0, -1]])
    self.state = z @ self.state

  def dump_state(self, basis: Literal["x", "y", "z"] = "z", decimals: int = 3) -> str:
    """Return a string representation of the current qubit in the specified basis."""
    if basis == "x":
      x = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
      state = x @ self.state
      basis_states = ("|+⟩", "|-⟩")

    elif basis == "y":
      y = np.array([[1, -1j], [1j, 1]]) / np.sqrt(2)
      state = y @ self.state
      basis_states = ("|i⟩", "|-i⟩")

    elif basis == "z":
      state = self.state
      basis_states = ("|0⟩", "|1⟩")

    return f"({state[0]:.{decimals}f}){basis_states[0]} + ({state[1]:.{decimals}f}){basis_states[1]}"

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


class NQubitState:
  """Represents an n-qubit quantum system as a 2^n dimensional complex vector:
  |ψ⟩ = Σ αᵢ|i⟩ where Σ|αᵢ|² = 1

  The basis states are indexed in binary order, e.g. for 3 qubits:
  |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩

  Attributes:
    num_qubits (int): Number of qubits in the system
    dim (int): Dimension of the quantum state vector
    state (NDArray[np.complex128]): The quantum state vector of dimension 2^n
  """

  def __init__(
    self,
    num_qubits: int,
    state: NDArray[np.complex128] | None = None,
    basis_state: int | None = 0,
  ):
    """Initialize an n-qubit system.

    Args:
      num_qubits (int): Number of qubits in the system
      state (NDArray[np.complex128]|None): Optional initial state vector of dimension 2^n
      basis_state (int|None): Optional computational basis state to initialize to (default |0...0⟩)
    """

    self.num_qubits = num_qubits
    self.dim = 2**num_qubits

    if state is not None:
      if state.shape != (self.dim,):
        raise ValueError(f"State vector must be {self.dim}-dimensional")
      self._validate_and_set_state(state)
      return

    # Initialize to specified basis state (default |0...0⟩)
    state = np.zeros(self.dim, dtype=np.complex128)
    state[basis_state] = 1.0
    self._state = state

  def copy(self) -> "NQubitState":
    new_state = NQubitState(self.num_qubits, state=self.state.copy())
    return new_state

  def _validate_and_set_state(self, state: NDArray[np.complex128]):
    """Validate and set the quantum state vector.

    Args:
      state (NDArray[np.complex128]): The quantum state vector to validate and set

    Raises:
      ValueError: If the state is not normalized
    """
    # Ensure complex type
    state = state.astype(np.complex128)

    # Check normalization
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0, rtol=1e-6):
      raise ValueError(f"State vector not normalized. Norm: {norm}")

    self._state = state

  @property
  def state(self) -> NDArray[np.complex128]:
    return self._state

  @state.setter
  def state(self, state: NDArray[np.complex128]):
    self._validate_and_set_state(state)

  def get_probabilities(self) -> NDArray[np.float64]:
    return np.square(np.abs(self.state))

  def get_reduced_density_matrix(
    self, target_qubits: list[int]
  ) -> NDArray[np.complex128]:
    """Calculate the reduced density matrix for specified qubits.

    Args:
      target_qubits (list[int]): List of qubit indices to keep (others are traced out)

    Returns:
      NDArray[np.complex128]: Reduced density matrix for the specified qubits
    """

    # Convert state vector to density matrix
    density_matrix = np.outer(self.state, np.conj(self.state))

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

    outcomes = _measure_shots(self.state.copy(), targets, random_outcomes, self.dim)

    result = [tuple(outcome) for outcome in outcomes]
    if not as_dict:
      return result

    outcome_strings = ["".join(map(str, outcome)) for outcome in result]
    return dict(sorted(Counter(outcome_strings).items()))

  def apply_unary_gate(self, gate: NDArray[np.complex128], target: int):
    """Apply a unary quantum gate to the specified qubit.

    Args:
      gate (NDArray[np.complex128]): 2x2 unitary matrix representing the gate
      target (int): Index of target qubit"""

    if gate.shape != (2, 2):
      raise ValueError("Gate must be a 2x2 matrix")

    # Construct full gate matrix using tensor products
    full_gate = np.array([[1]], dtype=np.complex128)
    for i in range(self.num_qubits):
      full_gate = np.kron(full_gate, gate if i == target else np.eye(2)).astype(
        np.complex128
      )

    self.state = full_gate @ self.state

  def apply_controlled_gate(
    self, gate: NDArray[np.complex128], control: int, target: int
  ):
    """Apply a controlled single-qubit gate.

    Args:
      gate (NDArray[np.complex128]): 2x2 unitary matrix representing the gate
      control (int): Index of control qubit
      target (int): Index of target qubit"""

    if gate.shape != (2, 2):
      raise ValueError("Gate must be a 2x2 matrix")

    # Initialize controlled gate
    controlled_gate = np.eye(self.dim, dtype=np.complex128)

    for i in range(self.dim):
      if (i >> target) & 1:  # Apply gate if target qubit is 1
        control_bit = (i >> control) & 1  # Get bit value of control qubit
        i_modified = i ^ (1 << control)  # Index of flipped control qubit

        controlled_gate[i, i] = gate[control_bit, control_bit]
        controlled_gate[i, i_modified] = gate[control_bit, 1 - control_bit]

    self.state = controlled_gate @ self.state

  def hadamard_gate(self, target: int):
    """Apply a Hadamard gate to the specified qubit(s).

    Args:
      target (int): Index of target qubit"""

    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    self.apply_unary_gate(H, target)

  def x_gate(self, target: int):
    """Apply a Pauli-X gate to the specified qubit(s).

    Args:
      target (int): Index of target qubit"""

    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    self.apply_unary_gate(x, target)

  def y_gate(self, target: int):
    """
    Apply a Pauli-Y gate to the specified qubit.

    Args:
      target (int): Index of target qubit"""

    y = np.array([[0, -1j], [1j, 0]])
    self.apply_unary_gate(y, target)

  def z_gate(self, target: int):
    """Apply a Pauli-Z gate to the specified qubit(s).

    Args:
      target (int): Index of target qubit"""

    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    self.apply_unary_gate(z, target)

  def rotation_x_gate(self, theta: float, target: int):
    """Apply a rotation around the X axis.

    Args:
      theta (float): Rotation angle in radians
      target (int): Index of target qubit"""

    R_x = np.array(
      [
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)],
      ]
    )
    self.apply_unary_gate(R_x, target)

  def rotation_y_gate(self, theta: float, target: int):
    """Apply a rotation around the Y axis.

    Args:
      theta (float): Rotation angle in radians
      target (int): Index of target qubit"""

    R_y = np.array(
      [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]]
    )
    self.apply_unary_gate(R_y, target)

  def rotation_z_gate(self, theta: float, target: int):
    """Apply a rotation around the Z axis.

    Args:
      theta (float): Rotation angle in radians
      target (int): Index of target qubit"""

    R_z = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
    self.apply_unary_gate(R_z, target)

  def s_gate(self, target: int):
    """Apply the S gate to the specified qubit(s).

    Args:
      target (int): Index of target qubit"""

    s = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    self.apply_unary_gate(s, target)

  def s_gate_dagger(self, target: int):
    """Apply the S† gate to the specified qubit(s).

    Args:
      target (int): Index of target qubit"""

    s_dagger = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    self.apply_unary_gate(s_dagger, target)

  def cnot_gate(self, control: int, target: int):
    """
    Apply the controlled-NOT (CNOT) gate with specified control qubit.

    Args:
      control (int): Index of control qubit
      target (int): Index of target qubit"""

    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    self.apply_controlled_gate(x, control, target)

  def swap_gate(self, qubit1: int, qubit2: int):
    """Apply a SWAP gate between two qubits using three CNOT gates

    Args:
      qubit1 (int): Index of first qubit
      qubit2 (int): Index of second qubit

    Raises:
      ValueError: If qubit1 and qubit2 are the same
    """

    if qubit1 == qubit2:
      raise ValueError("Qubits must be different")

    self.cnot_gate(qubit1, qubit2)
    self.cnot_gate(qubit2, qubit1)
    self.cnot_gate(qubit1, qubit2)

  def dump_state(self, decimals: int = 3) -> str:
    """Print the current state in the computational basis."""
    terms = []
    for i in range(self.dim):
      if abs(self.state[i]) > 1e-10:  # Ignore very small amplitudes
        basis = f"|{i:0{self.num_qubits}b}⟩"
        terms.append(f"({self.state[i]:.{decimals}f}){basis}")

    return " + ".join(terms)

  def __repr__(self) -> str:
    return (
      f"NQubitSystem(num_qubits={self.num_qubits}, dim={self.dim}, state={self.state})"
    )


@jit(nopython=True)
def _measure_shots(
  state: NDArray[np.complex128],
  targets: list[int],
  random_outcomes: NDArray[np.float64],
  dim: int,
):
  """JIT-compiled function to perform quantum measurements.

  Args:
    state (np.ndarray[np.complex128]): The quantum state vector
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


def create_bell_state(state: Literal[0, 1, 2, 3]) -> NQubitState:
  """
  Create a qubit in a Bell state using Hadamard and CNOT gates.

  Args:
    state (int): Bell state to initialize the qubit in. Can be 0, 1, 2, or 3.
    - 0 initializes |00⟩ resulting in |Φ+⟩ = (1/√2)|00⟩ + (1/√2)|11⟩
    - 1 initializes |01⟩ resulting in |Ψ+⟩ = (1/√2)|01⟩ + (1/√2)|10⟩
    - 2 initializes |10⟩ resulting in |Ψ-⟩ = (1/√2)|00⟩ - (1/√2)|11⟩
    - 3 initializes |11⟩ resulting in |Φ-⟩ = (1/√2)|01⟩ - (1/√2)|10⟩
  """

  bell_state = NQubitState(2, basis_state=state)
  bell_state.hadamard_gate(0)  # Apply Hadamard gate to first qubit
  bell_state.cnot_gate(0, 1)  # Apply CNOT gate

  return bell_state


def bell_state_circuit(state: Literal[0, 1, 2, 3]) -> QuantumCircuit:
  """
  Initialize a quantum circuit in a Bell state.

  Args:
    state (str): Bell state to initialize the qubit in. Can be 0, 1, 2, or 3.
    - 0 initializes |00⟩ resulting in |Φ+⟩ = (1/√2)|00⟩ + (1/√2)|11⟩
    - 1 initializes |01⟩ resulting in |Ψ+⟩ = (1/√2)|01⟩ + (1/√2)|10⟩
    - 2 initializes |10⟩ resulting in |Ψ-⟩ = (1/√2)|00⟩ - (1/√2)|11⟩
    - 3 initializes |11⟩ resulting in |Φ-⟩ = (1/√2)|01⟩ - (1/√2)|10⟩
  """

  qc = QuantumCircuit(2, 2)
  qc.initialize(f"{state:0{2}b}", [0, 1])

  # Apply Hadamard gate and CNOT to create |Φ+⟩
  qc.h(0)  # Apply Hadamard gate to first qubit
  qc.cx(0, 1)  # Apply CNOT gate

  if state == "Φ-":
    qc.z(0)  # Apply Z gate to first qubit
  elif state == "Ψ+":
    qc.x(1)  # Apply X gate to second qubit
  elif state == "Ψ-":
    qc.x(0)  # Apply X gate to first qubit
    qc.z(0)  # Apply Z gate to first qubit

  return qc


def dump_qubit_state(state: NDArray[np.complex128], decimals: int = 3) -> str:
  terms = []
  dim = len(state)
  num_qubits = int(np.log2(dim))
  for i in range(dim):
    if abs(state[i]) > 1e-10:  # Ignore very small amplitudes
      basis = f"|{i:0{num_qubits}b}⟩"
      terms.append(f"({state[i]:.{decimals}f}){basis}")

  return " + ".join(terms)
