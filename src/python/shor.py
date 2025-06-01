import math
import random


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
