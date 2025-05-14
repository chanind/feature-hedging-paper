from abc import ABC, abstractmethod
from pathlib import Path

from sae_lens import SAE


class Eval(ABC):
    @abstractmethod
    def has_eval_run(self, results_dir: Path) -> bool:
        pass

    @abstractmethod
    def run(self, sae: SAE, results_dir: Path, shared_dir: Path) -> None:
        pass
