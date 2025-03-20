import concurrent.futures

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

from vqe_utils import (
  prepare_ansatz,
  measure_nth_z,
  measure_z,
  measure_xx,
  measure_yy,
  qiskit_prepare_ansatz,
)


def hamiltonian(N: int, eps: float, V: float, W: float) -> NDArray[np.float64]:
  """
  Construct the Hamiltonian matrix for the Lipkin model.

  Args:
    N (int): Number of particles in the system.
    eps (float):Energy level spacing.
    V (float): Interaction strength of the Lipkin model.
    W (float): Interaction strength of the pairing force.

  Returns
    H (ndarray): The Hamiltonian matrix.
  """
  j = N / 2
  j_z = np.arange(-j, j + 1, 1)

  dim = len(j_z)
  H = np.zeros((dim, dim))

  # Diagonal elements
  np.fill_diagonal(H, j_z * eps + W * (j**2 - j_z**2))

  # Off-diagonal elements
  for i in range(dim - 2):
    jz = j_z[i]
    coeff = (
      0.5
      * V
      * np.sqrt((j * (j + 1) - jz * (jz + 1)) * (j * (j + 1) - (jz + 1) * (jz + 2)))
    )
    H[i, i + 2] = coeff
    H[i + 2, i] = coeff

  return H


def exact_energies(
  N: int, eps: float, W: float, V_array: NDArray[np.float64]
) -> NDArray[np.float64]:
  """
  Calculate the exact energies of the Lipkin model for a set of interaction strengths.

  Args:
    N (int): Number of particles in the system.
    eps (float): Energy level spacing.
    W (float): Interaction strength of the pairing force.
    V_array (NDArray[np.float64]): Array of interaction strengths of the Lipkin model.

  Returns:
    NDArray[np.float64]: 2D array of exact energies, sorted in ascending order for each interaction strength.
  """
  return np.array([np.linalg.eigvalsh(hamiltonian(N, eps, v, W)) for v in V_array])


def energy_expectation_two_fermions(
  angles: NDArray[np.float64],
  interaction_strength: float,
  shots: int,
  eps: float,
  W: float,
) -> float:
  """
  Calculate energy expectation value of the Lipkin model for two fermions.

  Args:
    eps (float): Energy level spacing.
    V (float): Interaction strength of the Lipkin model.
    W (float): Interaction strength of the pairing force.
    angles (NDArray[np.float64]): Parameter vector [theta_0, phi_0, theta_1, phi_1] used to prepare the ansatz state.
    shots (int): Number of measurement shots.

  Returns:
    float: Energy expectation value.
  """

  # Prepare ansatz
  qubit = prepare_ansatz(angles)

  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    future_zi = executor.submit(measure_z, qubit.copy(), shots)  # Measure Z⊗I
    future_iz = executor.submit(measure_nth_z, qubit.copy(), 1, shots)  # Measure I⊗Z
    future_xx = executor.submit(measure_xx, qubit.copy(), shots)  # Measure X⊗X
    future_yy = executor.submit(measure_yy, qubit.copy(), shots)  # Measure Y⊗Y

    try:
      expectation_iz = future_iz.result()
      expectation_zi = future_zi.result()
      expectation_xx = future_xx.result()
      expectation_yy = future_yy.result()
    except Exception as e:
      raise RuntimeError(f"Error in parallel measurement: {str(e)}")

  # Calculate expectation value
  exp_val = (
    0.5 * eps * (expectation_iz + expectation_zi)
    + 0.5 * (interaction_strength + W) * expectation_xx
    + 0.5 * (W - interaction_strength) * expectation_yy
  )

  return exp_val


def qiskit_energy_expectation_two_fermions(
  angles: NDArray[np.float64],
  interaction_strength: float,
  shots: int,
  eps: float,
  W: float,
) -> float:
  """
  Calculate energy expectation value of the Lipkin model for two fermions.

  Args:
    eps (float): Energy level spacing.
    V (float): Interaction strength of the Lipkin model.
    W (float): Interaction strength of the pairing force.
    angles (NDArray[np.float64]): Parameter vector [theta_0, phi_0, theta_1, phi_1] used to prepare the ansatz state.
    shots (int): Number of measurement shots.

  Returns:
    float: Energy expectation value.
  """

  # Prepare ansatz
  circuit = qiskit_prepare_ansatz(angles)

  # Construct Hamiltonian using Qiskit's SparsePauliOp
  pauli_strings = ["IZ", "ZI", "XX", "YY"]
  coeffs = [
    0.5 * eps,
    0.5 * eps,
    0.5 * (interaction_strength + W),
    0.5 * (W - interaction_strength),
  ]
  hamiltonian = SparsePauliOp(pauli_strings, coeffs)

  # Calculate expectation value
  estimator = Estimator()
  job = estimator.run(circuit, observables=hamiltonian, shots=shots)
  exp_val = job.result().values[0]

  return exp_val


def energy_expectation_four_fermions(
  eps: float, W: float, V: float, shots: int, angles: NDArray[np.float64]
) -> float:
  """
  Calculate energy expectation value of the Lipkin model for four fermions.

  Args:
    eps (float): Energy level spacing.
    V (float): Interaction strength of the Lipkin model.
    W (float): Interaction strength of the pairing force.
    angles (NDArray[np.float64]): Parameter vector [theta_0, phi_0, theta_1, phi_1, theta_2, phi_2, theta_3, phi_3] used to prepare the ansatz state.
    shots (int): Number of measurement shots.

  Returns:
    float: Energy expectation value.
  """

  # Prepare ansatz
  qubit = prepare_ansatz(angles)

  return 0.0


def qiskit_energy_expectation_four_fermions(
  angles: NDArray[np.float64],
  interaction_strength: float,
  shots: int,
  eps: float,
  W: float,
) -> float:
  """
  Calculate energy expectation value of the Lipkin model for four fermions.

  Args:
    eps (float): Energy level spacing.
    V (float): Interaction strength of the Lipkin model.
    W (float): Interaction strength of the pairing force.
    angles (NDArray[np.float64]): Parameter vector [theta_0, phi_0, theta_1, phi_1, theta_2, phi_2, theta_3, phi_3] used to prepare the ansatz state.
    shots (int): Number of measurement shots.

  Returns:
    float: Energy expectation value.
  """

  # Prepare ansatz
  circuit = qiskit_prepare_ansatz(angles)

  # Construct Hamiltonian using Qiskit's SparsePauliOp
  z_coeff = 0.5 * eps
  xx_coeff = 0.5 * (W - interaction_strength)
  yy_coeff = 0.5 * (W + interaction_strength)

  hamiltionian_terms = [
    ("ZIII", z_coeff),
    ("IZII", z_coeff),
    ("IIZI", z_coeff),
    ("IIIZ", z_coeff),
    ("XXII", xx_coeff),
    ("XIXI", xx_coeff),
    ("XIIX", xx_coeff),
    ("IXXI", xx_coeff),
    ("IXIX", xx_coeff),
    ("IIXX", xx_coeff),
    ("YYII", yy_coeff),
    ("YIYI", yy_coeff),
    ("YIIY", yy_coeff),
    ("IYYI", yy_coeff),
    ("IYIY", yy_coeff),
    ("IIYY", yy_coeff),
  ]
  pauli_strings, coeffs = zip(*hamiltionian_terms)

  hamiltonian = SparsePauliOp(list(pauli_strings), list(coeffs))

  # Calculate expectation value
  estimator = Estimator()
  job = estimator.run(circuit, observables=hamiltonian, shots=shots)
  exp_val = job.result().values[0]

  return exp_val
