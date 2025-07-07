from functools import partial
from typing import cast, Callable

import numpy as np
from numpy.typing import NDArray
import qiskit as qk
from scipy.optimize import minimize
from tqdm import tqdm

from qubit import NQubitState, basis_state
from optimize import minimize_energy, Optimizer


def prepare_ansatz(angles: NDArray[np.float64]) -> NQubitState:
  """
  Prepare a VQE ansatz state for an arbitrary number of qubits using rotation X and Y gates.

  Args:
    angles (NDArray[np.float64]): Parameter vector [theta_0, phi_0, theta_1, phi_1, ..., theta_n, phi_n]
    num_qubits (int): Number of qubits in the ansatz state

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
    num_qubits (int): Number of qubits in the ansatz state

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


def measure_z(qubit: NQubitState, shots: int) -> float:
  """
  Measure a qubit state in the Z⊗I⊗...⊗I basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """
  outcome = cast(dict[str, int], qubit.measure(list(range(qubit.num_qubits)), shots))
  return measurement_expectation(outcome, shots)


def measure_nth_z(qubit: NQubitState, z_index: int, shots: int) -> float:
  """
  Measure a qubit state in the I⊗...⊗I⊗Z⊗I⊗...⊗I basis, where Z is in the z_index position

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """
  if z_index >= qubit.num_qubits:
    raise ValueError("Qubit index out of range")

  qubit.swap(0, z_index)

  outcome = cast(dict[str, int], qubit.measure(list(range(qubit.num_qubits)), shots))
  return measurement_expectation(outcome, shots)


def measure_zz(qubit: NQubitState, shots: int) -> float:
  """
  Measure a two-qubit state in the Z⊗Z⊗I⊗...⊗I basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """
  qubit.cnot(1, 0)

  outcome = cast(dict[str, int], qubit.measure(list(range(qubit.num_qubits)), shots))
  return measurement_expectation(outcome, shots)


def measure_xx(qubit: NQubitState, shots: int) -> float:
  """
  Measure a qubit state in the X⊗X⊗I⊗...⊗I basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """
  qubit.hadamard(0)
  qubit.hadamard(1)
  qubit.cnot(1, 0)

  outcome = cast(dict[str, int], qubit.measure(list(range(qubit.num_qubits)), shots))
  return measurement_expectation(outcome, shots)


def measure_yy(qubit: NQubitState, shots: int) -> float:
  """
  Measure a qubit state in the Y⊗Y⊗I⊗...⊗I basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """
  qubit.s_dagger(0)
  qubit.hadamard(0)
  qubit.s_dagger(1)
  qubit.hadamard(1)
  qubit.cnot(1, 0)

  outcome = cast(dict[str, int], qubit.measure(list(range(qubit.num_qubits)), shots))
  return measurement_expectation(outcome, shots)


def measure_zizi(qubit: NQubitState, shots: int) -> float:
  """
  Measure a four-qubit state in the Z⊗I⊗Z⊗I basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """
  qubit.swap(0, 1)
  qubit.cnot(1, 0)

  outcome = cast(dict[str, int], qubit.measure([0, 1, 2, 3], shots))
  return measurement_expectation(outcome, shots)


def measure_xixi(qubit: NQubitState, shots: int) -> float:
  """
  Measure a four-qubit state in the X⊗I⊗X⊗I basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """

  qubit.hadamard(0)
  qubit.hadamard(2)

  qubit.swap(0, 1)
  qubit.cnot(1, 0)
  outcome = cast(dict[str, int], qubit.measure([0, 1, 2, 3], shots))
  return measurement_expectation(outcome, shots)


def measure_xiix(qubit: NQubitState, shots: int) -> float:
  """
  Measure a four-qubit state in the X⊗I⊗I⊗X basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """

  qubit.hadamard(0)
  qubit.swap(2, 3)
  qubit.hadamard(2)

  qubit.swap(0, 1)
  qubit.cnot(1, 0)
  outcome = cast(dict[str, int], qubit.measure([0, 1, 2, 3], shots))
  return measurement_expectation(outcome, shots)


def measure_ixxi(qubit: NQubitState, shots: int) -> float:
  """
  Measure a four-qubit state in the I⊗X⊗X⊗I basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """

  qubit.swap(0, 1)
  qubit.hadamard(0)
  qubit.hadamard(2)

  qubit.swap(0, 1)
  qubit.cnot(1, 0)
  outcome = cast(dict[str, int], qubit.measure([0, 1, 2, 3], shots))
  return measurement_expectation(outcome, shots)


def measure_ixix(qubit: NQubitState, shots: int) -> float:
  """
  Measure a four-qubit state in the I⊗X⊗I⊗X basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """

  qubit.swap(0, 1)
  qubit.hadamard(0)
  qubit.hadamard(2)

  qubit.swap(0, 1)
  qubit.cnot(1, 0)
  outcome = cast(dict[str, int], qubit.measure([0, 1, 2, 3], shots))
  return measurement_expectation(outcome, shots)


def measure_iixx(qubit: NQubitState, shots: int) -> float:
  """
  Measure a four-qubit state in the I⊗I⊗X⊗X basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """

  qubit.hadamard(2)
  qubit.hadamard(3)
  qubit.swap(0, 2)

  outcome = cast(dict[str, int], qubit.measure([0, 1, 2, 3], shots))
  return measurement_expectation(outcome, shots)


def measure_yiyi(qubit: NQubitState, shots: int) -> float:
  """
  Measure a four-qubit state in the Y⊗I⊗Y⊗I basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """

  qubit.s_dagger(0)
  qubit.hadamard(0)
  qubit.s_dagger(2)
  qubit.hadamard(2)

  qubit.swap(0, 1)
  qubit.cnot(1, 0)
  outcome = cast(dict[str, int], qubit.measure([0, 1, 2, 3], shots))
  return measurement_expectation(outcome, shots)


def measure_yiiy(qubit: NQubitState, shots: int) -> float:
  """
  Measure a four-qubit state in the Y⊗I⊗I⊗Y basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """

  qubit.s_dagger(0)
  qubit.hadamard(0)
  qubit.swap(2, 3)
  qubit.s_dagger(2)
  qubit.hadamard(2)

  qubit.swap(0, 1)
  qubit.cnot(1, 0)
  outcome = cast(dict[str, int], qubit.measure([0, 1, 2, 3], shots))
  return measurement_expectation(outcome, shots)


def measure_iyyi(qubit: NQubitState, shots: int) -> float:
  """
  Measure a four-qubit state in the I⊗Y⊗Y⊗I basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """

  qubit.swap(0, 1)
  qubit.s_dagger(0)
  qubit.hadamard(0)
  qubit.s_dagger(2)
  qubit.hadamard(2)

  qubit.swap(0, 1)
  qubit.cnot(1, 0)
  outcome = cast(dict[str, int], qubit.measure([0, 1, 2, 3], shots))
  return measurement_expectation(outcome, shots)


def measure_iyiy(qubit: NQubitState, shots: int) -> float:
  """
  Measure a four-qubit state in the I⊗Y⊗I⊗Y basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """

  qubit.swap(0, 1)
  qubit.s_dagger(0)
  qubit.hadamard(0)
  qubit.s_dagger(2)
  qubit.hadamard(2)

  qubit.swap(0, 1)
  qubit.cnot(1, 0)
  outcome = cast(dict[str, int], qubit.measure([0, 1, 2, 3], shots))
  return measurement_expectation(outcome, shots)


def measure_iiyy(qubit: NQubitState, shots: int) -> float:
  """
  Measure a four-qubit state in the I⊗I⊗Y⊗Y basis

  Args:
    qubit (NQubitState): Qubit state to measure
    shots (int): Number of measurement shots

  Returns:
    float: Expectation value
  """

  qubit.s_dagger(2)
  qubit.hadamard(2)
  qubit.s_dagger(3)
  qubit.hadamard(3)
  qubit.swap(0, 2)

  outcome = cast(dict[str, int], qubit.measure([0, 1, 2, 3], shots))
  return measurement_expectation(outcome, shots)


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
