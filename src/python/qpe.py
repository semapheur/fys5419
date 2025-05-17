from typing import cast
import numpy as np
import scipy as sp

from bitutils import decimal_to_bits, bits_to_binary_fraction
from gate import Gate, hadamard_gate, controlled_gate
from qubit import NQubitState, basis_state


def random_unitary_matrix(num_qubits: int):
  return sp.stats.unitary_group.rvs(2**num_qubits)


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

  h = hadamard_gate(num_qubits)
  psi = cast(NQubitState, h(psi, 0))

  for i, j in enumerate(range(num_qubits - 1, -1, -1)):
    u_power = u

    for _ in range(i):
      u_power = cast(Gate, u_power(u_power, 0))

    controlled_u = controlled_gate(j, num_qubits, u_power)
    psi = cast(NQubitState, controlled_u(psi, j))

  return psi


def qpe_test(u: Gate, eigenstate: NQubitState, precision: int) -> tuple[float, float]:
  """
  Perform Quantum Phase Estimation (QPE) on a given eigenstate with a specified unitary operator.

  Args:
    u (Gate): A unitary gate representing the operator whose eigenphase is to be estimated.
    eigenstate (NQubitState): The eigenstate of the unitary operator `u`.
    precision (int): The number of qubits used for precision in the phase estimation.

  Returns:
    Tuple[float, float]: Estimated phase as a binary fraction and the probability
                         (max_prob) of the measured state.
  """

  psi = cast(NQubitState, basis_state(precision, 0) * eigenstate)
  psi = phase_kick_gate(psi, u, precision)

  max_state, max_prob = psi.get_max_probability()

  phase_estimate = bits_to_binary_fraction(decimal_to_bits(max_state, precision))

  return phase_estimate, max_prob
