from functools import partial
from typing import Callable

import numpy as np
import qiskit as qk
from numpy.typing import NDArray
from optimize import Optimizer, minimize_energy
from qubit import NQubitState, basis_state
from scipy.optimize import minimize
from tqdm import tqdm


def prepare_ansatz(angles: NDArray[np.float64]) -> NQubitState:
  """
  Prepare a VQE ansatz state for an arbitrary number of qubits using rotation X and Y gates.

  Args:
    angles (NDArray[np.float64]): Parameter vector [theta_0, phi_0, theta_1, phi_1, ..., theta_n, phi_n]

  Returns:
    NQubitState: VQE ansatz state
  """

  if len(angles) % 2 != 0:
    raise ValueError("Number of angles must be even")

  num_qubits = len(angles) // 2

  # Initialize the ansatz state
  qubit = basis_state(num_qubits, 0)

  # Apply rotation X and Y gates
  for i in range(num_qubits):
    theta_i = angles[2 * i]
    phi_i = angles[2 * i + 1]
    qubit.rotation_x(theta_i, i)
    qubit.rotation_y(phi_i, i)

  if num_qubits == 1:  # Special case for single qubit
    return qubit

  # Apply CNOT gates to entangle adjacent qubits
  for i in range(num_qubits - 1):
    qubit.cnot(i, i + 1)

  return qubit


def qiskit_prepare_ansatz(angles: NDArray[np.float64]) -> qk.QuantumCircuit:
  """
  Prepare a VQE ansatz state for an arbitrary number of qubits using rotation X and Y gates.

  Args:
    angles (NDArray[np.float64]): Parameter vector [theta_0, phi_0, theta_1, phi_1, ..., theta_n, phi_n]

  Returns:
    QuantumCircuit: VQE ansatz state
  """

  if len(angles) % 2 != 0:
    raise ValueError("Number of angles must be even")

  num_qubits = len(angles) // 2

  # Initialize the ansatz state
  circuit = qk.QuantumCircuit(num_qubits)

  # Apply rotation X and Y gates
  for i in range(num_qubits):
    theta_i = angles[2 * i]
    phi_i = angles[2 * i + 1]
    circuit.rx(theta_i, i)
    circuit.ry(phi_i, i)

  if num_qubits == 1:  # Special case for single qubit
    return circuit

  # Apply CNOT gates to entangle adjacent qubits
  for i in range(num_qubits - 1):
    circuit.cx(i, i + 1)

  return circuit


def measurement_expectation(
  counts: dict[str, int], shots: int, qubit_index: int = -1
) -> float:
  """
  Calculate expectation value from measurement counts.

  Args:
    counts (dict[str, int]): Dictionary of measurement outcomes and their counts
    shots (int): Total number of shots
    qubit_index (int): Index of the qubit to compute expectation value for (default: last qubit).

  Returns:
    float: Expectation value in [-1, 1]
  """
  expectation = sum(
    (1 if outcome[qubit_index] == "0" else -1) * count
    for outcome, count in counts.items()
  )

  return expectation / shots


def measure_pauli(state: NQubitState, pauli: str, shots: int) -> float:
  """
  Measure a qubit state for a given Pauli string

  Args:
    qubit (NQubitState): Qubit state to measure
    pauli (str): Pauli string
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """

  n = state.num_qubits
  pauli = pauli.upper().ljust(n, "I")
  if len(pauli) != n:
    raise ValueError(f"Pauli string longer than {n} qubits")

  psi = state.copy()
  mask = 0
  for q, p in enumerate(pauli):
    if p == "X":
      psi.hadamard(q)
    elif p == "Y":
      psi.s_dagger(q)
      psi.hadamard(q)
    elif p not in "IZ":
      raise ValueError(f"Invalid Pauli character {p!r}")

    if p != "I":
      mask |= 1 << (n - 1 - q)

  probs = psi.get_probabilities()
  signs = np.array([1 - 2 * (bin(i & mask).count("1") & 1) for i in range(len(probs))])
  samples = np.random.choice(len(probs), size=shots, p=probs / probs.sum())
  return float(signs[samples].mean())


def _term(pauli: str):
  def fn(qubit: NQubitState, shots: int) -> float:
    return measure_pauli(qubit, pauli, shots)

  fn.__name__ = f"measure_{pauli.lower()}"
  return fn


measure_z, measure_zz, measure_xx, measure_yy = (
  _term(p) for p in ("Z", "ZZ", "XX", "YY")
)
measure_zizi, measure_xixi, measure_xiix = _term("ZIZI"), _term("XIXI"), _term("XIIX")
measure_ixxi, measure_ixix, measure_iixx = _term("IXXI"), _term("IXIX"), _term("IIXX")
measure_yiyi, measure_yiiy, measure_iyyi = _term("YIYI"), _term("YIIY"), _term("IYYI")
measure_iyiy, measure_iiyy = _term("IYIY"), _term("IIYY")


def measure_nth_z(qubit: NQubitState, z_index: int, shots: int) -> float:
  """
  Measure a qubit state in the I⊗...⊗I⊗Z⊗I⊗...⊗I basis, where Z is in the z_index position

  Args:
    qubit (NQubitState): Qubit state to measure
    z_index (int): Index of Z
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """

  pauli = "I" * z_index + "Z"
  return measure_pauli(qubit, pauli, shots)


def vqe_energies(
  angle_parameters: int,
  energy_fn: Callable,
  lambdas: np.ndarray,
  shots: int,
  max_epochs: int,
  learning_rate: float,
  method: Optimizer,
  verbose: bool = False,
):
  """
  Run VQE for multiple lambda values to find ground state energies.

  Args:
    angle_parameters (int): Number of angle parameters
    energy_fn (Callable): Energy expectation function
    lambdas (np.ndarray): Array of lambda values to evaluate
    shots (int): Number of measurement shots per expectation calculation
    max_epochs (int): Maximum optimization epochs per lambda
    learning_rate (float): Initial learning rate for optimization
    method (Optimizer): Optimization method
    verbose (bool): Whether to print progress information

  Returns:
    Tuple: (energies, epochs_used, optimal_angles)
  """

  partial_energy_fn = partial(energy_fn, shots=shots)
  vqe_energies = np.zeros(len(lambdas))
  epochs = np.zeros(len(lambdas))
  optimal_angles = []

  for i, lmb in enumerate(lambdas):
    if verbose:
      print(f"\nProcessing lambda = {lmb:.4f} ({i + 1}/{len(lambdas)})")

    best_energy = float("inf")
    best_angles = None
    best_epoch = 0

    angles_0 = np.random.uniform(0, np.pi, angle_parameters)
    angles, energy, _, epoch = minimize_energy(
      angles_0,
      partial_energy_fn,
      lmb,
      learning_rate=learning_rate,
      epochs=max_epochs,
      method=method,
      verbose=verbose,
    )

    if energy < best_energy:
      best_energy = energy
      best_angles = angles
      best_epoch = epoch

    vqe_energies[i] = best_energy
    epochs[i] = best_epoch
    optimal_angles.append(best_angles)

  return vqe_energies, epochs, optimal_angles


def scipy_vqe_energies(
  angle_parameters: int,
  energy_fn: Callable,
  lambdas: np.ndarray,
  shots: int,
  method: str = "Powell",
  max_iterations: int = 1000,
  tolerance: float = 1e-5,
  verbose: bool = False,
):
  """
  Run VQE for multiple lambda values to find ground state energies using scipy.optimize.minimize.

  Args:
    angle_parameters (int): Number of angle parameters
    energy_fn (Callable): Energy expectation function
    lambdas (np.ndarray): Array of lambda values to evaluate
    shots (int): Number of measurement shots per expectation calculation
    method (str): Optimization method, defaults to "Powell"
    max_iterations (int): Maximum optimization iterations, defaults to 1000
    tolerance (float): Optimization tolerance, defaults to 1e-5
    verbose (bool): Whether to print optimization progress

  Returns:
    np.ndarray: Array of minimum energies
  """
  partial_energy_fn = partial(energy_fn, shots=shots)

  min_energies = np.zeros(len(lambdas))
  for i, lmb in enumerate(tqdm(lambdas)):
    angles_0 = np.random.uniform(0, np.pi, angle_parameters)
    result = minimize(
      fun=partial_energy_fn,
      x0=angles_0,
      args=lmb,
      method=method,
      options={"maxiter": max_iterations},
      tol=tolerance,
    )
    min_energies[i] = result.fun
    if verbose:
      print(result.message)

  return min_energies
