from typing import cast

import numpy as np
from numpy.typing import NDArray
import qiskit_aer
import qiskit as qk

from qubit import basis_state
from vqe_utils import measurement_expectation

E_1 = 0
E_2 = 4
V_11 = 3
V_22 = -3
V_12 = 0.2

epsilon = (E_1 + E_2) / 2
omega = (E_1 - E_2) / 2
c = (V_11 + V_22) / 2
omega_z = (V_11 - V_22) / 2
omega_x = V_12


def hamiltonian(interaction_strength: float) -> NDArray[np.float64]:
  """
  Construct the Hamiltonian matrix for a two-level quantum system.

  Args:
    interaction_strength (float): Interaction strength parameter that scales the interacting part of the Hamiltonian.

  Returns:
    NDArray[np.float64]: A 2x2 Hamiltonian matrix representing the quantum system.
  """

  I_2 = np.eye(2)
  sigma_x = np.array([[0, 1], [1, 0]])
  sigma_z = np.array([[1, 0], [0, -1]])

  H_0 = epsilon * I_2 + omega * sigma_z
  H_1 = c * I_2 + omega_z * sigma_z + omega_x * sigma_x

  H = H_0 + interaction_strength * H_1

  return H


def analytic_energies(lambdas: NDArray[np.float64]) -> NDArray[np.float64]:
  """
  Calculate the analytic energies of a single-qubit Hamiltonian for a set of interaction strengths.

  Args:
    lambdas (NDArray[np.float64]): Array of interaction strengths

  Returns:
    NDArray[np.float64]: 2-column array of analytic energies, sorted in ascending order for each interaction strength
  """

  root = np.sqrt(np.square(omega + lambdas * omega_z) + np.square(lambdas * omega_x))

  E_minus = epsilon + lambdas * c - root
  E_plus = epsilon + lambdas * c + root

  return np.column_stack((E_minus, E_plus))


def numeric_energies(lambdas: NDArray[np.float64]) -> NDArray[np.float64]:
  """
  Calculate the exact energies of a single-qubit Hamiltonian for a set of interaction strengths.

  Args:
    lambdas (NDArray[np.float64]): Array of interaction strengths

  Returns:
    NDArray[np.float64]: 2-column array of exact energies, sorted in ascending order for each interaction strength
  """

  return np.array([np.linalg.eigvalsh(hamiltonian(lmb)) for lmb in lambdas])


def prepare_ansatz(angles: NDArray[np.float64]):
  """
  Prepare a single-qubit ansatz state using two angles.

  Args:
    angles (NDArray[np.float64]): 2-element array of angles (theta, phi) in radians

  Returns:
    NQubitState: Single-qubit ansatz state
  """
  theta, phi = angles
  qubit = basis_state(1, 0)
  qubit.rotation_x(theta, 0)
  qubit.rotation_y(phi, 0)
  return qubit


def energy_expectation(
  angles: NDArray[np.float64], interaction_strength: float, shots: int
) -> float:
  """
  Calculate energy expectation value for a single-qubit Hamiltonian of the form
  H = epsilon * I + omega * Z + c * I + omega_z * Z + omega_x * X

  Args:
    angles (NDArray[np.float64]): Parameter vector [theta, phi]
    interaction_strength (float): Interaction strength parameter
    shots (int): Number of measurement shots

  Returns:
    float: Energy expectation value
  """

  # Prepare ansatz
  ansatz = prepare_ansatz(angles)

  # Measure in Z-basis
  qubit = ansatz.copy()
  outcomes = cast(dict[str, int], qubit.measure([0], shots))
  measure_z = measurement_expectation(outcomes, shots)

  # Measure in X-basis
  qubit = ansatz.copy()
  qubit.hadamard(0)  # Rotate to X-basis
  outcomes = cast(dict[str, int], qubit.measure([0], shots))
  measure_x = measurement_expectation(outcomes, shots)

  # Calculate expectation value
  exp_val_z = (omega + interaction_strength * omega_z) * measure_z
  exp_val_x = interaction_strength * omega_x * measure_x
  exp_val_i = epsilon + c * interaction_strength
  exp_val = exp_val_z + exp_val_x + exp_val_i
  return exp_val


def qiskit_energy_expectation(
  angles: np.ndarray, interaction_strength: float, shots: int
) -> float:
  # Prepare ansatz
  theta, phi = angles
  circuit = qk.QuantumCircuit(1)
  circuit.rx(theta, 0)
  circuit.ry(phi, 0)

  # Z-basis measurement circuit
  circuit_z = circuit.copy()
  creg_z = qk.ClassicalRegister(1, "z")
  circuit_z.add_register(creg_z)
  circuit_z.measure(0, creg_z)

  # X-basis measurement circuit
  circuit_x = circuit.copy()
  creg_x = qk.ClassicalRegister(1, "x")
  circuit_x.add_register(creg_x)
  circuit_x.h(0)
  circuit_x.measure(0, creg_x)

  # Execute circuits
  hamiltonian_circuits = [circuit_z, circuit_x]

  simulator = qiskit_aer.Aer.get_backend("qasm_simulator")
  job = simulator.run(hamiltonian_circuits, shots=shots)
  result = job.result()

  # Z-basis measurement
  counts_z = result.get_counts(0)
  measure_z = measurement_expectation(counts_z, shots)

  # X-basis measurement
  counts_x = result.get_counts(1)
  measure_x = measurement_expectation(counts_x, shots)

  # Calculate expectation value
  exp_val_z = (omega + interaction_strength * omega_z) * measure_z
  exp_val_x = interaction_strength * omega_x * measure_x
  exp_val_i = epsilon + c * interaction_strength
  exp_val = exp_val_z + exp_val_x + exp_val_i
  return exp_val
