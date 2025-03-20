import itertools
from typing import cast, Iterable, Literal

type Bit = Literal[0, 1]
type Bits = Iterable[Bit]


def bits_to_decimal(bits: Bits) -> int:
  """Convert a list of bits to a decimal integer."""

  bits = list(bits)
  return sum(cast(bool, bit * (1 << len(bits) - i - 1)) for i, bit in enumerate(bits))


def decimal_to_bits(val: int, nbits: int) -> Bits:
  """Convert a decimal integer to a list of bits."""

  return [cast(Bit, int(bit)) for bit in f"{val:0{nbits}b}"]


def bit_product(num_bits: int) -> Iterable[Bits]:
  """Produce the iterable cartesian product of num_bits bits."""

  for bits in itertools.product((0, 1), repeat=num_bits):
    yield cast(Bits, bits)
