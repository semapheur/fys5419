from concurrent.futures import ThreadPoolExecutor
import time
from typing import cast, Callable, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

type Optimizer = Literal["adam", "rmsprop", "adagrad", "sgd"]

OPTIMIZER_PARAMS = {
  "adam": {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
  "rmsprop": {"decay_rate": 0.9, "epsilon": 1e-8},
  "adagrad": {"epsilon": 1e-8},
  "sgd": {"momentum": 0.9},
}


class OptimizerState(TypedDict, total=False):
  squared_grad: NDArray[np.float64] | None
  grad_accumulator: NDArray[np.float64] | None
  momentum: float | None
  momentums: NDArray[np.float64] | None
  velocities: NDArray[np.float64] | None


def parameter_shift_gradient(
  angles: NDArray[np.float64],
  interaction_strength: float,
  energy_fn: Callable[[NDArray[np.float64], float], float],
) -> NDArray[np.float64]:
  """
  Calculate gradient using parameter shift rule.

  Args:
    angles (NDArray[np.float64],]): Parameter vector [theta, phi]
    interaction_strength (float): Interaction strength parameter
    energy_fn (Callable): Energy calculation function

  Returns:
    NDArray[np.float64],: Gradient vector
  """

  grad = np.zeros_like(angles)
  shift = np.pi / 2

  shifts: list[tuple[NDArray[np.float64], float]] = []
  for i in range(len(angles)):
    angles_plus = angles.copy()
    angles_plus[i] += shift
    shifts.append((angles_plus, interaction_strength))

    angles_minus = angles.copy()
    angles_minus[i] -= shift
    shifts.append((angles_minus, interaction_strength))

  with ThreadPoolExecutor() as executor:  # Parallelize
    energies = list(executor.map(lambda args: energy_fn(*args), shifts))

  for i, (energy_plus, energy_minus) in enumerate(zip(energies[::2], energies[1::2])):
    grad[i] = (energy_plus - energy_minus) / (2 * shift)

  return grad


def minimize_energy(
  angles_0: NDArray[np.float64],
  energy_fn: Callable[[NDArray[np.float64], float], float],
  lmb: float,
  learning_rate: float = 0.01,
  epochs: int = 1000,
  tolerance: float = 1e-6,
  method: Optimizer = "adam",
  line_search: bool = False,
  early_stop: int = 50,
  verbose: bool = True,
) -> tuple[NDArray[np.float64], float, float, int]:
  """
  Minimize energy function using gradient descent optimization.

  Args:
    angles_0 (NDArray[np.float64]): Initial parameter vector [theta, phi]
    energy_fn (Callable[[NDArray[np.float64], float], float]): Energy expectation function
    lmb (float): Interaction strength parameter
    learning_rate (float): Initial learning rate
    epochs (int): Maximum number of iterations
    tolerance (float): Convergence threshold for energy change
    method (str): Optimization method: "adam", "sgd", "rmsprop", "adagrad"
    line_search (bool): Whether to use backtracking line search
    early_stop (int): Number of epochs with no improvement before early stopping
    adaptive_lr (bool): Whether to use adaptive learning rate
    verbose (bool): Whether to print optimization progress

  Returns:
    Tuple: (optimized_angles, final_energy, energy_change, epochs_used)
  """

  angles = angles_0.copy()
  epoch = 0
  delta_energy = float("inf")
  min_energy = float("inf")
  no_improvement_epochs = 0
  start_time = time.time()

  energy = energy_fn(angles, lmb)
  energies = [energy]
  angle_store = [angles.copy()]

  optimizer_state = initialize_optimizer_state(method, angles.shape[0])

  # Track when to log progress
  log_interval = max(1, epochs // 10) if verbose else epochs

  while epoch < epochs and delta_energy > tolerance:
    # Calculate gradient
    grad = parameter_shift_gradient(angles, lmb, energy_fn)

    # Update learning rate and optimizer state
    update, optimizer_state = optimizer_update(
      method, grad, optimizer_state, learning_rate, epoch, **OPTIMIZER_PARAMS[method]
    )

    step_size = 1.0
    if line_search:
      step_size = backtracking_line_search(angles, update, energy_fn, lmb, energy)

    angles += step_size * update

    # Update energy and delta_energy
    energy_new = energy_fn(angles, lmb)
    delta_energy = np.abs(energy_new - energy)
    energy = energy_new

    energies.append(energy)
    angle_store.append(angles.copy())

    if energy < min_energy:
      min_energy = energy
      no_improvement_epochs = 0
    else:
      no_improvement_epochs += 1

    if no_improvement_epochs >= early_stop:
      if verbose:
        print(f"Early stopping after {epoch} epochs with no improvement")
      break

    epoch += 1

    if verbose and (epoch % log_interval == 0 or epoch == epochs):
      print(f"Epoch {epoch}: Energy = {energy:.6f}, Change = {delta_energy:.6f}")

  final_energy = energy_fn(angles, lmb)
  computation_time = time.time() - start_time

  if verbose:
    print(
      f"Optimization completed after {epoch} epochs ({computation_time:.2f} seconds)"
    )
    print(f"Final energy: {final_energy:.8f}")
    print(f"Final angles: {angles}")

  return angles, energy, delta_energy, epoch


def initialize_optimizer_state(method: Optimizer, dim: int) -> OptimizerState:
  if method == "adam":
    return OptimizerState(momentums=np.zeros(dim), velocities=np.zeros(dim))
  elif method == "rmsprop":
    return OptimizerState(squared_grad=np.zeros(dim))
  elif method == "adagrad":
    return OptimizerState(grad_accumulator=np.zeros(dim))
  elif method == "sgd":
    return OptimizerState(velocities=np.zeros(dim))
  else:
    return {}


def optimizer_update(
  method: Optimizer,
  gradient: NDArray[np.float64],
  state: OptimizerState,
  learning_rate: float,
  epoch: int,
  **kwargs,
):
  if method == "adam":
    beta1 = cast(float, kwargs.get("beta1", 0.9))
    beta2 = cast(float, kwargs.get("beta2", 0.999))
    epsilon = cast(float, kwargs.get("epsilon", 1e-8))

    momentums = cast(NDArray[np.float64], state["momentums"])
    velocities = cast(NDArray[np.float64], state["velocities"])

    momentums = beta1 * momentums + (1 - beta1) * gradient
    velocities = beta2 * velocities + (1 - beta2) * np.square(gradient)

    momentums_hat = momentums / (1 - beta1 ** (epoch + 1))
    velocities_hat = velocities / (1 - beta2 ** (epoch + 1))

    update = -learning_rate * momentums_hat / (np.sqrt(velocities_hat) + epsilon)
    state["momentums"] = momentums
    state["velocities"] = velocities

  elif method == "rmsprop":
    decay_rate = cast(float, kwargs.get("decay_rate", 0.9))
    epsilon = cast(float, kwargs.get("epsilon", 1e-8))
    squared_grad = cast(NDArray[np.float64], state["squared_grad"])

    squared_grad = decay_rate * squared_grad + (1 - decay_rate) * np.square(gradient)
    update = -learning_rate * gradient / (np.sqrt(squared_grad) + epsilon)

    state["squared_grad"] = squared_grad

  elif method == "adagrad":
    epsilon = cast(float, kwargs.get("epsilon", 1e-8))
    grad_accumulator = cast(NDArray[np.float64], state["grad_accumulator"])

    grad_accumulator += np.square(gradient)
    update = -learning_rate * gradient / (np.sqrt(grad_accumulator) + epsilon)

    state["grad_accumulator"] = grad_accumulator

  elif method == "sgd":
    momentum = cast(float, kwargs.get("momentum"))
    velocities = cast(NDArray[np.float64], state["velocities"])

    adapted_lr = learning_rate / (1 + 0.1 * epoch)
    velocities = momentum * velocities - adapted_lr * gradient
    update = velocities

    state["velocities"] = velocities

  else:  # Default to gradient descent
    update = -learning_rate * gradient

  return update, state


def backtracking_line_search(
  angles: NDArray[np.float64],
  update: NDArray[np.float64],
  energy_fn: Callable[[NDArray[np.float64], float], float],
  lmb: float,
  current_energy: float,
  alpha: float = 0.1,
  beta: float = 0.7,
  c: float = 1e-4,
  max_iterations: int = 10,
) -> float:
  """
  Perform backtracking line search to determine the optimal step size.

  Args:
    angles (NDArray[np.float64]): Current parameter angles.
    update (NDArray[np.float64]): Update direction vector.
    energy_fn (Callable[[NDArray[np.float64], float], float]): Function to compute energy given angles and lambda.
    lmb (float): Lambda parameter for energy function.
    current_energy (float): Current energy value.
    alpha (float, optional): Initial step size. Default is 0.1.
    beta (float, optional): Factor to decrease step size. Default is 0.7.
    c (float, optional): Armijo condition constant. Default is 1e-4.
    max_iterations (int, optional): Maximum number of iterations. Default is 10.

  Returns:
    float: Optimal step size satisfying the Armijo condition.
  """

  step_size = alpha
  grad_dot_update = np.sum(update * update)  # Update is the gradient

  for _ in range(max_iterations):
    new_angles = angles + step_size * update
    new_energy = energy_fn(new_angles, lmb)

    # Armijo condition
    if new_energy < current_energy + c * step_size * grad_dot_update:
      return step_size

    step_size *= beta

  return step_size
