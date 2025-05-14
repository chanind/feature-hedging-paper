import traceback
from pathlib import Path

from sae_lens import SAE

from hedging_paper.evals.absorption_eval import AbsorptionEval
from hedging_paper.evals.autointerp_eval import AutointerpEval
from hedging_paper.evals.eval import Eval
from hedging_paper.evals.pos_sparse_probing_eval import POSSparseProbingEval
from hedging_paper.evals.scr_eval import SCREval
from hedging_paper.evals.sparse_probing_eval import SparseProbingEval
from hedging_paper.evals.tpp_eval import TPPEval

ALL_EVALS = [
    AbsorptionEval(),
    SparseProbingEval(),
    SCREval(),
    TPPEval(),
    AutointerpEval(),
    POSSparseProbingEval(),
]


def run_eval(
    eval: Eval, sae: SAE, results_dir: Path, shared_dir: Path, force: bool = False
) -> None:
    if eval.has_eval_run(results_dir) and not force:
        print(f"Skipping {eval.__class__.__name__} because it has already been run")
        return
    print(f"Running {eval.__class__.__name__}")
    eval.run(sae, results_dir, shared_dir)


def run_all_evals(
    sae: SAE,
    results_dir: Path,
    shared_dir: Path,
    evals: list[Eval] | None = None,
    force: bool = False,
    crash_on_error: bool = False,
) -> None:
    if evals is None:
        evals = ALL_EVALS
    for eval in evals:
        try:
            run_eval(
                eval, sae, results_dir=results_dir, shared_dir=shared_dir, force=force
            )
        except Exception as e:
            if crash_on_error:
                raise e
            print(f"Error running {eval.__class__.__name__}:")
            print(traceback.format_exc())
