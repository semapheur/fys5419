import fractions
import math
import random

from bitutils import bits_to_decimal, bits_to_fraction, bit_product
from circuit import QuantumCircuit
from qubit import QubitRegister, basis_state


def is_prime(x: int) -> bool:
  """Check if x is a prime number."""

  if x < 2:
    return False

  if x == 2:
    return True

  if x % 2 == 0:
    return False

  for i in range(3, int(math.sqrt(x)) + 1, 2):
    if x % i == 0:
      return False

  return True


def is_coprime(a: int, b: int) -> bool:
  """Check if a and b are coprime."""

  return math.gcd(a, b) == 1


def get_odd_compound(lower: int, upper: int, max_attempts: int = 10000) -> int | None:
  """Generate a random odd compound number within the specified inclusive range."""

  if upper % 2 == 0:
    upper -= 1

  if upper < 9:
    raise ValueError(
      f"Upper bound must be at least 9 (smallest odd compound). Got {upper}"
    )

  if lower % 2 == 0:
    lower += 1

  lower = max(lower, 9)

  if lower > upper:
    raise ValueError(f"Lower bound must be less than upper bound: {lower} > {upper}")

  # For small ranges, precompute odd compounds
  range_size = (upper - lower) // 2 + 1
  if range_size <= 100:
    candidates = [n for n in range(lower, upper + 1, 2) if not is_prime(n)]
    if len(candidates) == 0:
      raise ValueError(f"Unable to find odd compound in the range {lower} to {upper}")
    return random.choice(candidates)

  # Otherwise, use a random number generator
  for _ in range(max_attempts):
    n = random.randrange(lower, upper + 1, 2)
    if not is_prime(n):
      return n

  # Fallback to linear search
  for i in range(lower, upper + 1, 2):
    if not is_prime(i):
      return i

  raise ValueError(f"Unable to find odd compound in the range {lower} to {upper}")


def get_coprime(upper: int, max_attempts: int = 10000) -> int:
  """Generate a random integer coprime to the given upper limit."""

  if upper <= 2:
    raise ValueError(f"Upper limit must be at least 3. Got {upper}")

  for _ in range(max_attempts):
    n = random.randint(3, upper - 1)
    if is_coprime(n, upper):
      return n

  for n in range(3, upper):
    if is_coprime(n, upper):
      return n

  raise ValueError(f"Unable to find coprime in the range 3 to {upper}")


def multiplicative_order(n: int, modulus: int) -> int:
  """Compute the multiplicative order of n modulo modulus.
  That is, the smallest positive integer k such that n^k = 1 mod modulus."""

  order = 1
  current = n % modulus
  while current != 1:
    current = (current * n) % modulus
    order += 1

    if order > modulus:
      raise ValueError(f"Order is greater than modulus: {order} > {modulus}")

  return order


def modular_inverse(a: int, m: int) -> int:
  """Modular inverse of a mod m is the number a^{-1} such that a * a^{-1} = 1 mod m."""

  def extended_euclidean_algorithm(a: int, b: int) -> tuple[int, int, int]:
    if a == 0:
      return b, 0, 1

    gcd, y, x = extended_euclidean_algorithm(b % a, a)
    return gcd, x - (b // a) * y, y

  gcd, x, _ = extended_euclidean_algorithm(a, m)

  if gcd != 1:
    raise ValueError(f"Modular inverse does not exist: {a} and {m} are not coprime")

  return x % m


def precompute_angles(a: int, n: int) -> list[float]:
  s = bin(a)[2:].zfill(n)

  angles = [0.0] * n
  for i in range(n):
    for j in range(i, n):
      if s[j] == "1":
        angles[n - i - 1] += 2 ** (-(j - i))

    angles[n - i - 1] *= math.pi

  return angles


def add_fourier(
  qc: QuantumCircuit,
  reg: QubitRegister,
  a: int,
  n: int,
  factor: float,
):
  """Add in Fourier space"""

  angles = precompute_angles(a, n)
  for i in range(n):
    qc.p(reg[i], factor * angles[i])


def cadd_fourier(
  qc: QuantumCircuit,
  reg: QubitRegister,
  control: int,
  a: int,
  n: int,
  factor: float,
):
  """Controlled add in Fourier space"""

  angles = precompute_angles(a, n)
  for i in range(n):
    qc.cp(control, reg[i], factor * angles[i])


def ccadd_fourier(
  qc: QuantumCircuit,
  reg: QubitRegister,
  control1: int,
  control2: int,
  a: int,
  n: int,
  factor: float,
):
  """Double-controlled add in Fourier space"""

  angles = precompute_angles(a, n)
  for i in range(n):
    qc.ccp(control1, control2, reg[i], factor * angles[i])


def ccadd_mod(
  qc: QuantumCircuit,
  reg: QubitRegister,
  control1: int,
  control2: int,
  aux: int,
  a: int,
  number: int,
  n: int,
):
  """Double-controlled modular addition"""

  ccadd_fourier(qc, reg, control1, control2, a, n, factor=1.0)
  add_fourier(qc, reg, a, n, factor=-1.0)
  qc.qft(reg, inverse=True, flip=False)
  qc.cx(reg[n - 1], aux)
  qc.qft(reg, inverse=False, flip=False)
  cadd_fourier(qc, reg, aux, number, n, factor=1.0)

  ccadd_fourier(qc, reg, control1, control2, a, n, factor=-1.0)
  qc.qft(reg, inverse=True, flip=False)
  qc.x(reg[n - 1])
  qc.cx(reg[n - 1], aux)
  qc.x(reg[n - 1])
  qc.qft(reg, inverse=False, flip=False)
  ccadd_fourier(qc, reg, control1, control2, a, n, factor=1.0)


def ccadd_mod_inverse(
  qc: QuantumCircuit,
  reg: QubitRegister,
  control1: int,
  control2: int,
  aux: int,
  a: int,
  number: int,
  n: int,
):
  ccadd_fourier(qc, reg, control1, control2, a, n, factor=-1.0)
  qc.qft(reg, inverse=True, flip=False)
  qc.x(reg[n - 1])
  qc.cx(reg[n - 1], aux)
  qc.x(reg[n - 1])
  qc.qft(reg, inverse=False, flip=False)
  ccadd_fourier(qc, reg, control1, control2, a, n, factor=1.0)

  cadd_fourier(qc, reg, aux, number, n, factor=-1.0)
  qc.qft(reg, inverse=True, flip=False)
  qc.cx(reg[n - 1], aux)
  add_fourier(qc, reg, number, n, factor=1.0)
  ccadd_fourier(qc, reg, control1, control2, a, n, factor=-1.0)


def cmul_mod(
  qc: QuantumCircuit,
  control: int,
  reg: QubitRegister,
  aux: QubitRegister,
  a: int,
  number: int,
  n: int,
):
  qc.qft(aux, flip=False)
  for i in range(n):
    ccadd_mod(
      qc, aux, reg[i], control, aux[n + 1], (1 << i) * a % number, number, n + 1
    )
  qc.qft(reg, inverse=True, flip=False)

  for i in range(n):
    qc.cswap(control, reg[i], aux[i])
  a_inv = modular_inverse(a, number)

  qc.qft(aux, inverse=False, flip=False)
  for i in range(n - 1, -1, -1):
    ccadd_mod_inverse(
      qc, aux, reg[i], control, aux[n + 1], (1 << i) * a_inv % number, number, n + 1
    )

  qc.qft(aux, inverse=True, flip=False)


def order_finding(a: int, n: int) -> tuple[int, int]:
  """Run Shor's algorithm for order finding.

  Args
  a (int): The base of the exponentiation.
  n (int): The modulus.

  Returns
  tuple[int, int]: A tuple (r1, r2) such that r1 and r2 are both candidates for the order of a modulo n.
  """

  n_bits = n.bit_length()

  qc = QuantumCircuit("order_finding")
  aux = qc.add_register(basis_state(n_bits + 2, n_bits + 2), name="q0")
  up = qc.add_register(basis_state(n_bits * 2, n_bits * 2), name="q1")
  down = qc.add_register(basis_state(n_bits, n_bits), name="q2")

  _ = modular_inverse(a, n)

  qc.h(up)
  qc.x(down[0])
  for i in range(n_bits * 2):
    cmul_mod(qc, up[i], down, aux, int(a ** (2**i)), n, n_bits)

  qc.qft(up, inverse=True, flip=True)
  qc.qft(down, inverse=True, flip=False)

  total_prob = 0.0
  for bits in bit_product(n_bits * 4 + 2):
    prob = qc.state.get_probability(bits)
    if prob > 0.1:
      bitslice = bits[n_bits + 2 : n_bits + 2 + n_bits * 2][::-1]
      decimal = bits_to_decimal(bitslice)
      phase = bits_to_fraction(bitslice)

      r = fractions.Fraction(phase).limit_denominator(8).denominator
      guesses = (math.gcd(a ** (r // 2) - 1, n), math.gcd(a ** (r // 2) + 1, n))
      print(f"Final x: {decimal}, phase: {phase}, prob: {prob}, factors: {guesses}")

    total_prob += prob
    if total_prob > 0.999:
      break

    print(qc.stats())

  return guesses


def classic_shor(n: int, max_attempts: int = 100) -> tuple[int, int]:
  """Factorize a number using the classical part of Shor's algorithm."""

  for _ in range(max_attempts):
    try:
      a = get_coprime(n)

      gcd_check = math.gcd(a, n)
      if gcd_check > 1:
        return gcd_check, n // gcd_check

      r = multiplicative_order(a, n)
      if r % 2 != 0:
        continue

      half_power = pow(a, r // 2, n)
      if half_power == n - 1:
        continue

      factor1 = math.gcd(half_power + 1, n)
      factor2 = math.gcd(half_power - 1, n)

      if factor1 > 1 and factor1 < n:
        return factor1, n // factor1

      if factor2 > 1 and factor2 < n:
        return factor2, n // factor2

    except Exception:
      continue

  raise ValueError(f"Factorization of {n} failed after {max_attempts} attempts")
