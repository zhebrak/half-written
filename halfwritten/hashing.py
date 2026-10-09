"""Deterministic hashes (Python's own hash() of a string changes per process), for seeded draws in the simulator."""
import hashlib

_MASK = (1 << 64) - 1


def stable_key(*parts) -> int:
    """A 64-bit integer from the parts' string forms, the same in every process."""
    return int.from_bytes(hashlib.blake2b(":".join(map(str, parts)).encode(), digest_size=8).digest(), "little")


def mix(x: int) -> int:
    """splitmix64: a cheap, well-spread integer hash for per-token draws."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)
