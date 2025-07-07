from typing import Callable, cast

import numpy as np
from numpy.typing import NDArray
import scipy as sp
from tqdm import tqdm

from circuit import QuantumCircuit
from gate import Gate, hadamard_gate, controlled_gate, fourier_transform
from qubit import NQubitState, basis_state


def random_unitary_matrix(num_qubits: int):
  return sp.stats.unitary_group.rvs(2**num_qubits)


def phase_probability(
  phase: float, precision: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """
  Calculate the probability distribution for a phase measurement of a given precision.

  Arg:
    phase (float): The phase to calculate the distribution for.
    precision (int): The number of bits to use for the calculation.

  Returns:
    y_scaled (NDArray[np.float64]): The scaled y-values of the distribution, from 0 to 1.
    probability (NDArray[np.float64]): The probability at each y-value.
  """

  N = 2**precision
  y = np.arange(N)
  y_scaled = y / N

  numerator = 1 - np.exp(1j * 2 * np.pi * (y - N * phase))
  denominator = 1 - np.exp(1j * 2 * np.pi * (y - N * phase) / N)

  probability = np.abs(numerator / denominator) ** 2 / N**2

  return y_scaled, probability


def phase_kick_gate(psi: NQubitState, u: Gate, num_qubits: int) -> NQubitState:
  """
  Applies a sequence of phase kick operations to the input quantum state.

  Args:
    psi (NQubitState): The input quantum state to which the phase kick operations are applied.
    u (Gate): A unitary gate representing the phase kick operator.
    num_qubits (int): The number of qubits in the quantum state.

  Returns:
    NQubitState: The quantum state after applying the phase kick operations.
  """

  h = hadamard_gate()
  # hadamard_gate(num_qubits) is memory inefficient for large num_qubits

  # Apply Hadamard gates
  for i in range(num_qubits):
    psi = cast(NQubitState, h(psi, i))

  u_power = u.copy()
  for j in reversed(range(num_qubits)):
    controlled_u = controlled_gate(j, num_qubits, u_power)
    psi = cast(NQubitState, controlled_u(psi, j))
    u_power = cast(Gate, u_power(u_power))

  return psi


def qpe(unitary: Gate, eigenstate: NQubitState, precision: int) -> tuple[float, float]:
  """
  Perform Quantum Phase Estimation (QPE) on a given eigenstate with a specified unitary operator.
  This implementation is memory inefficient for large precision due to costly Kronecker products.

  Args:
    unitary (Gate): A unitary gate representing the operator whose eigenphase is to be estimated.
    eigenstate (NQubitState): The eigenstate of the unitary operator `u`.
    precision (int): The number of qubits used for precision in the phase estimation.

  Returns:
    Tuple[float, float]: Estimated phase as a binary fraction and the probability
                         (max_prob) of the measured state.
  """

  iqft = fourier_transform(precision, inverse=True)

  psi = cast(NQubitState, basis_state(precision, 0) * eigenstate)
  psi = phase_kick_gate(psi, unitary, precision)
  psi = cast(NQubitState, iqft(psi))

  total_qubits = precision + eigenstate.num_qubits
  precision_register = list(range(eigenstate.num_qubits, total_qubits))
  max_state, max_prob = psi.get_max_probability(precision_register)
  phase_estimate = max_state / (1 << precision)

  return phase_estimate, max_prob


def qpe_circuit(
  unitary: Gate, eigenstate: NQubitState, precision: int
) -> tuple[float, float]:
  """
  Perform Quantum Phase Estimation (QPE) on a given eigenstate with a specified unitary operator using a quantum circuit.
  This implementation only works for 2x2 gates.

  Args:
    u (Gate): A unitary 2x2 gate representing the operator whose eigenphase is to be estimated.
    eigenstate (NQubitState): The eigenstate of the unitary operator `u`.
    precision (int): The number of qubits used for precision in the phase estimation.

  Returns:
    Tuple[float, float]: Estimated phase as a binary fraction and the probability
                         (max_prob) of the measured state.
  """
  qc = QuantumCircuit("qpe")
  q0 = qc.add_register(basis_state(precision, 0), name="q0")
  q1 = qc.add_register(eigenstate, name="q1")

  # Apply Hadamard gates
  qc.h(q0)

  u_power = unitary.copy()
  for j in reversed(q0):
    qc.apply_controlled_gate(u_power, j, q1[0], f"u^{{2^{j}}}")
    u_power = cast(Gate, u_power @ u_power)  # cast(Gate, u_power(u_power))

  # Apply inverse fourier transform
  qc.qft(q0, inverse=True)

  # Measure
  max_state, max_prob = qc.max_probability(range(len(q1), qc.num_qubits))
  phase_estimate = max_state / (1 << len(q0))

  return phase_estimate, max_prob


def qpe_measurements(
  unitary: Gate, eigenstate: NQubitState, precision: int, shots: int
) -> dict[str, int]:
  """
  Perform Quantum Phase Estimation (QPE) on a given eigenstate with a specified unitary operator
  and return the measurement results.

  Args:
    unitary (Gate): A unitary 2x2 gate representing the operator whose eigenphase is to be estimated.
    eigenstate (NQubitState): The eigenstate of the unitary operator `u`.
    precision (int): The number of qubits used for precision in the phase estimation.
    shots (int): The number of measurements to perform.

  Returns:
    dict[str, int]: A dictionary mapping measurement outcomes to their counts
  """
  iqft = fourier_transform(precision, inverse=True)

  psi = cast(NQubitState, basis_state(precision, 0) * eigenstate)
  psi = phase_kick_gate(psi, unitary, precision)
  psi = cast(NQubitState, iqft(psi))

  total_qubits = precision + eigenstate.num_qubits
  precision_register = list(range(eigenstate.num_qubits, total_qubits))

  return psi.measure_simple(shots, precision_register)


def qpe_energy(
  hamiltonian: Callable[[float], NDArray[np.float64]],
  parameters: NDArray[np.float64],
  precision: int,
  eigenindex: int = 0,
  time_scale: float | None = 1.0,
):
  """
  Estimate energy eigenvalues of a Hamiltonian using Quantum Phase Estimation (QPE).

  Args:
    hamiltonian (Callable[[float], NDArray[np.float64]]): A function returning the Hamiltonian matrix for a given parameter.
    parameters (NDArray[np.float64]): Array of parameters for which to estimate the eigenvalues.
    precision (int): Number of qubits used for precision in phase estimation.
    eigenindex (int, optional): Index of the eigenvalue to estimate. Defaults to 0.
    time_scale (float | None, optional): Scaling factor for time evolution. If None, it is automatically computed. Defaults to 1.0.

  Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]: Estimated energies from QPE and their corresponding errors compared to true energies.
  """

  qpe_energies = np.zeros(len(parameters))
  true_energies = np.zeros(len(parameters))
  phases = np.zeros(len(parameters))
  time_scales = np.zeros(len(parameters))

  for i, lmb in enumerate(tqdm(parameters)):
    h = hamiltonian(lmb)

    eigenvalues, eigenvectors = np.linalg.eigh(h)
    true_energies[i] = eigenvalues[eigenindex]

    if time_scale is None:
      spectral_range = np.max(eigenvalues) - np.min(eigenvalues)
      time_scale = 2 * np.pi / spectral_range

    u = Gate(sp.linalg.expm(1.0j * h * time_scale))
    eigenstate = NQubitState(eigenvectors[:, eigenindex])
    phase_estimate, _ = qpe(u, eigenstate, precision)
    phases[i] = phase_estimate
    time_scales[i] = time_scale

  # Resolve phase discontinuities
  unwrapped_phases = np.unwrap(2 * np.pi * phases)
  for i in range(len(parameters)):
    qpe_energies[i] = unwrapped_phases[i] / time_scales[i]

  errors = np.abs(qpe_energies - true_energies)

  return qpe_energies, errors


def print_qpe_result(target_phase: float, phase_estimate: float, probability: float):
  print(f"Phase: {target_phase:.3f}")
  print(
    f"Phase estimate: {phase_estimate:.3f} (error: {np.abs(phase_estimate - target_phase):.2e})"
  )
  print(f"Probability: {probability:.3f}")
