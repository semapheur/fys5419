from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
import qiskit_aer
import qiskit as qk

from qubit import NQubitState, basis_state
from vqe_utils import (
  measure_z,
  measurement_expectation,
  measure_nth_z,
  measure_zz,
  measure_xx,
)

sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])

H_x = 2.0
H_z = 3.0

eps_00 = 0.0
eps_01 = 2.5
eps_10 = 6.5
eps_11 = 7.0

eps_ii = (eps_00 + eps_01 + eps_10 + eps_11) / 4.0
eps_zi = (eps_00 + eps_01 - eps_10 - eps_11) / 4.0
eps_iz = (eps_00 - eps_01 + eps_10 - eps_11) / 4.0
eps_zz = (eps_00 - eps_01 - eps_10 + eps_11) / 4.0


def hamiltonian(interaction_strength: float) -> NDArray[np.float64]:
  """Construct the Hamiltonian matrix for a two-qubit quantum system."""
  H_I = H_z * np.kron(sigma_z, sigma_z) + H_x * np.kron(sigma_x, sigma_x)
  H_0 = np.diag([eps_00, eps_01, eps_10, eps_11])
  H = H_0 + interaction_strength * H_I

  return H


def trace_out(state: NDArray[np.complex128], index: int):
  """
  Calculate the reduced density matrix of a two-qubit state by tracing out a specified qubit.
  Based from https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/Programs/LipkinModel/two_qubit_VQE.ipynb

  Args:
    state (NDArray[np.complex128]): State vector of the quantum system.
    index (int): Index of the qubit to trace out.

  Returns:
    NDArray[np.complex128]: Reduced density matrix after tracing out the specified qubit.
  """

  state_dim = state.shape[0]
  if state_dim != 4:
    raise ValueError("State vector must have dimension 4")

  if index < 0 or index > 1:
    raise ValueError("Index must be 0 or 1")

  I2 = np.eye(2)
  ket0 = np.array([1.0, 0.0])
  ket1 = np.array([0.0, 1.0])

  density = np.outer(state, np.conj(state))
  if index == 0:
    op0 = np.kron(ket0, I2)
    op1 = np.kron(ket1, I2)
  else:
    op0 = np.kron(I2, ket0)
    op1 = np.kron(I2, ket1)

  return op0.conj() @ density @ op0.T + op1.conj() @ density @ op1.T


def exact_energies_and_entropies(lambdas: NDArray[np.float64]):
  """
  Calculate the exact energies and von Neumann entropies for a two-qubit system Hamiltonian.
  Based from https://github.com/CompPhysics/QuantumComputingMachineLearning/blob/gh-pages/doc/Programs/LipkinModel/two_qubit_VQE.ipynb


  Args:
    lambdas (NDArray[np.float64]): Array of interaction strengths.

  Returns:
    Tuple[NDArray[np.float64], NDArray[np.float64]]:
      - A 2D array of exact energies, with each row corresponding to a lambda value.
      - A 2D array of von Neumann entropies, with each row corresponding to a lambda value.
  """

  energies = np.zeros((len(lambdas), 4))
  entropies = np.zeros((len(lambdas), 4))

  for i, lmb in enumerate(lambdas):
    H = hamiltonian(lmb)
    eigenvalues, eigenvectors = np.linalg.eig(H)

    permute = eigenvalues.argsort()
    energies[i] = eigenvalues[permute]
    eigenvectors = eigenvectors[:, permute]

    # Calculate von Neumann entropy
    for j in range(4):
      sub_density = trace_out(eigenvectors[:, j], 0)
      eigenvals_density = np.linalg.eigvalsh(sub_density)
      eigenvals_density = np.ma.masked_equal(eigenvals_density, 0.0).compressed()
      entropies[i, j] = -np.sum(eigenvals_density * np.log2(eigenvals_density))

  return energies, entropies


def prepare_ansatz(angles: NDArray[np.float64]) -> NQubitState:
  theta_0, phi_0, theta_1, phi_1 = angles
  qubit = basis_state(2, 0)
  qubit.rotation_x(theta_0, 0)
  qubit.rotation_y(phi_0, 0)
  qubit.rotation_x(theta_1, 1)
  qubit.rotation_y(phi_1, 1)
  qubit.cnot(0, 1)  # entangle qubits

  return qubit


def energy_expectation(
  angles: NDArray[np.float64], interaction_strength: float, shots: int
) -> float:
  """
  Calculate energy expectation value of a two-qubit Hamiltonian

  The Hamiltonian is of the form:
  H = eps_II * I⊗I + eps_IZ * I⊗Z + eps_ZI * Z⊗I + (eps_ZZ + lmb * H_z) * Z⊗Z + lmb * H_x * X⊗X

  Args:
    angles (NDArray[np.float64]): Parameter vector [theta_0, phi_0, theta_1, phi_1]
    lmb (float): Interaction strength parameter
    shots (int): Number of measurement shots

  Returns:
    float: Energy expectation value
  """

  # Prepare ansatz
  ansatz = prepare_ansatz(angles)

  with ThreadPoolExecutor(max_workers=4) as executor:  # Parallelize
    future_zi = executor.submit(measure_z, ansatz.copy(), shots)  # Measure Z⊗I
    future_iz = executor.submit(measure_nth_z, ansatz.copy(), 1, shots)  # Measure I⊗Z
    future_zz = executor.submit(measure_zz, ansatz.copy(), shots)  # Measure Z⊗Z
    future_xx = executor.submit(measure_xx, ansatz.copy(), shots)  # Measure X⊗X

    try:
      expectation_zi = future_zi.result()
      expectation_iz = future_iz.result()
      expectation_zz = future_zz.result()
      expectation_xx = future_xx.result()
    except Exception as e:
      raise RuntimeError(f"Error in parallel measurement: {e}")

  # Calculate expectation value
  exp_val = (
    eps_ii
    + eps_zi * expectation_zi
    + eps_iz * expectation_iz
    + (eps_zz + interaction_strength * H_z) * expectation_zz
    + (interaction_strength * H_x * expectation_xx)
  )

  return exp_val


def qiskit_ansatz(angles: NDArray[np.float64]) -> qk.QuantumCircuit:
  theta_0, phi_0, theta_1, phi_1 = angles
  circuit = qk.QuantumCircuit(2)
  circuit.rx(theta_0, 0)
  circuit.ry(phi_0, 0)
  circuit.rx(theta_1, 1)
  circuit.ry(phi_1, 1)
  circuit.cx(0, 1)
  return circuit


def qiskit_energy_expectation(
  angles: NDArray[np.float64], interaction_strength: float, shots: int
) -> float:
  """
  Calculate energy expectation value of a two-qubit Hamiltonian using Qiskit

  The Hamiltonian is of the form:
  H = eps_II * I⊗I + eps_IZ * I⊗Z + eps_ZI * Z⊗I + (eps_ZZ + lmb * H_z) * Z⊗Z + lmb * H_x * X⊗X

  Args:
    angles (NDArray[np.float64]): Parameter vector [theta_0, phi_0, theta_1, phi_1]
    lmb (float): Interaction strength parameter
    shots (int): Number of measurement shots

  Returns:
    float: Energy expectation value
  """

  circuit = qiskit_ansatz(angles)

  # Z⊗I measurement circuit
  circuit_zi = circuit.copy()
  creg_iz = qk.ClassicalRegister(2, "zi")
  circuit_zi.add_register(creg_iz)
  circuit_zi.measure([0, 1], creg_iz)

  # I⊗Z measurement circuit
  circuit_iz = circuit.copy()
  creg_iz = qk.ClassicalRegister(2, "iz")
  circuit_iz.add_register(creg_iz)
  circuit_iz.swap(0, 1)
  circuit_iz.measure([0, 1], creg_iz)

  # Z⊗Z measurement circuit
  circuit_zz = circuit.copy()
  creg_zz = qk.ClassicalRegister(2, "zz")
  circuit_zz.add_register(creg_zz)
  circuit_zz.cx(1, 0)
  circuit_zz.measure([0, 1], creg_zz)

  # X⊗X measurement circuit
  circuit_xx = circuit.copy()
  creg_xx = qk.ClassicalRegister(2, "xx")
  circuit_xx.add_register(creg_xx)
  circuit_xx.h(0)
  circuit_xx.h(1)
  circuit_xx.cx(1, 0)
  circuit_xx.measure([0, 1], creg_xx)

  # Execute circuits
  measurement_circuits = [circuit_zi, circuit_iz, circuit_zz, circuit_xx]

  simulator = qiskit_aer.Aer.get_backend("qasm_simulator")
  transpiled_circuits = qk.transpile(measurement_circuits, simulator)
  job = simulator.run(transpiled_circuits, shots=shots)
  result = job.result()

  # Calculate expectation value
  measurements = [
    measurement_expectation(result.get_counts(i), shots)
    for i in range(len(measurement_circuits))
  ]

  exp_val = (
    eps_ii
    + eps_zi * measurements[0]
    + eps_iz * measurements[1]
    + (eps_zz + interaction_strength * H_z) * measurements[2]
    + (interaction_strength * H_x * measurements[3])
  )

  return exp_val
