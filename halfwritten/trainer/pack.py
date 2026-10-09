"""Micro-batches under a token budget: first-fit decreasing over sequence lengths (prompt + answer). Deterministic, so
two arms that consume the same answers train them in the same micro-batches. A sequence longer than the budget gets a
micro-batch of its own."""


def pack(lengths: list[int], budget: int) -> list[list[int]]:
    """Indices of `lengths` grouped into bins whose padded size (members x longest) stays within `budget`."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    bins: list[list[int]] = []
    longest: list[int] = []
    for i in order:
        for b, members in enumerate(bins):
            if (len(members) + 1) * longest[b] <= budget:
                members.append(i)
                break
        else:
            bins.append([i])
            longest.append(lengths[i])
    return bins
