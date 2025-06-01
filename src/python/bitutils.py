import itertools
from typing import cast, Iterable, Literal, Sequence

type Bit = Literal[0, 1]
type Bits = Sequence[Bit]


def is_bitstring(s: str) -> bool:
  """Check if a string is a bitstring."""

  return all(c in "01" for c in s)


def bits_to_decimal(bits: Bits) -> int:
  """Convert a list of bits to a decimal integer."""

  return sum(cast(bool, bit * (1 << len(bits) - i - 1)) for i, bit in enumerate(bits))


def decimal_to_bits(val: int, nbits: int) -> Bits:
  """Convert a decimal integer to a list of bits."""

  return [cast(Bit, int(bit)) for bit in f"{val:0{nbits}b}"]


def bits_to_fraction(bits: Bits) -> float:
  """Convert a list of bits to a binary fraction."""

  result = 0.0
  factor = 0.5
  for bit in bits:
    result += bit * factor
    factor *= 0.5
  return result


def bit_product(num_bits: int) -> Iterable[Bits]:
  """Produce the iterable cartesian product of num_bits bits."""

  for bits in itertools.product((0, 1), repeat=num_bits):
    yield cast(Bits, bits)
