from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

from vqe_utils import (
  measure_iixx,
  measure_iiyy,
  measure_ixix,
  measure_ixxi,
  measure_iyiy,
  measure_iyyi,
  measure_xiix,
  measure_xixi,
  measure_yiiy,
  measure_yiyi,
  prepare_ansatz,
  measure_nth_z,
  measure_z,
  measure_xx,
  measure_yy,
  qiskit_prepare_ansatz,
)


def hamiltonian(
  N: int,
  eps: float,
  W: float,
  V: float,
) -> NDArray[np.float64]:
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
  return np.array([np.linalg.eigvalsh(hamiltonian(N, eps, W, v)) for v in V_array])


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
    angles (NDArray[np.float64]): Parameter vector [theta_0, phi_0, theta_1, phi_1] used to prepare the ansatz state.
    interaction_strength (float): Interaction strength of the Lipkin model.
    shots (int): Number of measurement shots.
    eps (float): Energy level spacing.
    W (float): Interaction strength of the pairing force.

  Returns:
    float: Energy expectation value.
  """

  # Prepare ansatz
  qubit = prepare_ansatz(angles)

  with ThreadPoolExecutor(max_workers=4) as executor:  # Parallelize
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
    + 0.5 * (W + interaction_strength) * expectation_xx
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
  angles: NDArray[np.float64],
  interaction_strength: float,
  shots: int,
  eps: float,
  W: float,
) -> float:
  """
  Calculate energy expectation value of the Lipkin model for four fermions.

  Args:
    angles (NDArray[np.float64]): Parameter vector [theta_0, phi_0, theta_1, phi_1, theta_2, phi_2, theta_3, phi_3] used to prepare the ansatz state.
    interaction_strength (float): Interaction strength of the Lipkin model.
    shots (int): Number of measurement shots.
    eps (float): Energy level spacing.
    W (float): Interaction strength of the pairing force.

  Returns:
    float: Energy expectation value.
  """

  # Prepare ansatz
  qubit = prepare_ansatz(angles)

  with ThreadPoolExecutor(max_workers=4) as executor:  # Parallelize
    future_ziii = executor.submit(measure_z, qubit.copy(), shots)
    future_izii = executor.submit(measure_nth_z, qubit.copy(), 1, shots)
    future_iziz = executor.submit(measure_nth_z, qubit.copy(), 2, shots)
    future_zzii = executor.submit(measure_nth_z, qubit.copy(), 3, shots)

    future_xxii = executor.submit(measure_xx, qubit.copy(), shots)
    future_xixi = executor.submit(measure_xixi, qubit.copy(), shots)
    future_xiix = executor.submit(measure_xiix, qubit.copy(), shots)
    future_ixxi = executor.submit(measure_ixxi, qubit.copy(), shots)
    future_ixix = executor.submit(measure_ixix, qubit.copy(), shots)
    future_iixx = executor.submit(measure_iixx, qubit.copy(), shots)

    future_yyii = executor.submit(measure_yy, qubit.copy(), shots)
    future_yiyi = executor.submit(measure_yiyi, qubit.copy(), shots)
    future_yiiy = executor.submit(measure_yiiy, qubit.copy(), shots)
    future_iyyi = executor.submit(measure_iyyi, qubit.copy(), shots)
    future_iyiy = executor.submit(measure_iyiy, qubit.copy(), shots)
    future_iiyy = executor.submit(measure_iiyy, qubit.copy(), shots)

    try:
      expectation_ziii = future_ziii.result()
      expectation_izii = future_izii.result()
      expectation_iizi = future_iziz.result()
      expectation_iiiz = future_zzii.result()

      expectation_xxii = future_xxii.result()
      expectation_xixi = future_xixi.result()
      expectation_xiix = future_xiix.result()
      expectation_ixxi = future_ixxi.result()
      expectation_ixix = future_ixix.result()
      expectation_iixx = future_iixx.result()

      expectation_yyii = future_yyii.result()
      expectation_yiyi = future_yiyi.result()
      expectation_yiiy = future_yiiy.result()
      expectation_iyyi = future_iyyi.result()
      expectation_iyiy = future_iyiy.result()
      expectation_iiyy = future_iiyy.result()
    except Exception as e:
      raise RuntimeError(f"Error in parallel execution: {e}")

  # Calculate energy expectation value
  exp_val = (
    (0.5 * eps)
    * (expectation_ziii + expectation_izii + expectation_iizi + expectation_iiiz)
    + (0.5 * (W - interaction_strength))
    * (
      expectation_xxii
      + expectation_xixi
      + expectation_xiix
      + expectation_ixxi
      + expectation_ixix
      + expectation_iixx
    )
    + (0.5 * (W + interaction_strength))
    * (
      expectation_yyii
      + expectation_yiyi
      + expectation_yiiy
      + expectation_iyyi
      + expectation_iyiy
      + expectation_iiyy
    )
  )

  return exp_val


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
    angles (NDArray[np.float64]): Parameter vector [theta_0, phi_0, theta_1, phi_1, theta_2, phi_2, theta_3, phi_3] used to prepare the ansatz state.
    interaction_strength (float): Interaction strength of the Lipkin model.
    shots (int): Number of measurement shots.
    eps (float): Energy level spacing.
    W (float): Interaction strength of the pairing force.

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
