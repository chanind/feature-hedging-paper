import math
from collections.abc import Generator, Sequence
from decimal import Decimal
from typing import TypeVar

import torch
from tqdm import tqdm

T = TypeVar("T")
DEFAULT_DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_DEVICE = torch.device(DEFAULT_DEVICE_STR)


def cos_sims(mat1: torch.Tensor, mat2: torch.Tensor):
    """
    Calculate the cosine similarity between each row of mat1 and each row of mat2.

    Args:
        mat1: A tensor of shape (n_rows1, n_cols1).
        mat2: A tensor of shape (n_rows2, n_cols2).

    Returns:
        A tensor of shape (n_rows1, n_rows2) containing the cosine similarity between each row of mat1 and each row of mat2.
    """
    mat1_normed = mat1 / (mat1.norm(dim=0, keepdim=True))
    mat2_normed = mat2 / (mat2.norm(dim=0, keepdim=True))

    return mat1_normed.T @ mat2_normed


def dtypify(dtype_str: str) -> torch.dtype:
    return getattr(torch, dtype_str)


def batchify(
    data: Sequence[T], batch_size: int, show_progress: bool = False
) -> Generator[Sequence[T], None, None]:
    """Generate batches from data. If show_progress is True, display a progress bar."""

    for i in tqdm(
        range(0, len(data), batch_size),
        total=(len(data) // batch_size + (len(data) % batch_size != 0)),
        disable=not show_progress,
    ):
        yield data[i : i + batch_size]


def untuple_tensor(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


class Tween:
    def __init__(self, start: float, end: float, n_steps: int, start_step: int = 0):
        self.start = start
        self.end = end
        self.n_steps = n_steps
        self.current = start
        self.start_step = start_step

    def __call__(self, step: int) -> float:
        if step < self.start_step:
            return self.start
        elif step >= self.start_step + self.n_steps:
            return self.end
        else:
            return (
                self.start
                + (self.end - self.start) * (step - self.start_step) / self.n_steps
            )


def listify(x: T | list[T]) -> list[T]:
    return x if isinstance(x, list) else [x]


# Copied from https://github.com/azaitsev/millify/blob/master/millify/__init__.py


def remove_exponent(d):
    """Remove exponent."""
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()


def millify(n, precision=0, drop_nulls=True, prefixes=[]):
    """Humanize number."""
    millnames = ["", "k", "M", "B", "T", "P", "E", "Z", "Y"]
    if prefixes:
        millnames = [""]
        millnames.extend(prefixes)
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )
    result = "{:.{precision}f}".format(n / 10 ** (3 * millidx), precision=precision)
    if drop_nulls:
        result = remove_exponent(Decimal(result))
    return f"{result}{millnames[millidx]}"
