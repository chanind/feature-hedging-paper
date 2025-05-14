from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sae_bench.evals.absorption.common import PROBES_DIR
from sae_bench.evals.absorption.feature_absorption import (
    FEATURE_ABSORPTION_EXPERIMENT_NAME,
    run_feature_absortion_experiment,
)
from sae_bench.evals.absorption.k_sparse_probing import (
    SPARSE_PROBING_EXPERIMENT_NAME,
    run_k_sparse_probing_experiment,
)
from sae_bench.evals.absorption.main import _aggregate_results_df
from sae_bench.evals.absorption.probing import LinearProbe
from sae_bench.sae_bench_utils import general_utils
from sae_lens import SAE
from torch.nn.modules.linear import Linear
from transformer_lens import HookedTransformer

from hedging_paper.evals.eval import Eval
from hedging_paper.evals.util import glob_matches
from hedging_paper.util import DEFAULT_DEVICE_STR


@dataclass
class AbsorptionEvalOptions:
    device: str | None = None
    llm_dtype: str = "float32"
    sae_name: str = "sae"
    force_rerun: bool = False
    max_k_value: int = 10
    f1_jump_threshold: float = 0.03
    prompt_template: str = "{word} has the first letter:"
    prompt_token_pos: int = -6
    k_sparse_probe_l1_decay: float = 0.01
    k_sparse_probe_batch_size: int = 4096
    k_sparse_probe_num_epochs: int = 50
    llm_batch_size: int = 10


class AbsorptionEval(Eval):
    def __init__(self, options: AbsorptionEvalOptions | None = None):
        self.options = options or AbsorptionEvalOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        return glob_matches(results_dir, "**/feature_absorption/**/*.parquet")

    def run(self, sae: SAE, results_dir: Path, shared_dir: Path) -> None:
        eval_absorption(
            sae, self.options, results_dir, probes_dir=shared_dir / "probes"
        )


def eval_absorption(
    sae: SAE,
    options: AbsorptionEvalOptions,
    results_dir: str | Path,
    probes_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    modified version of sae_bench.evals.absorption.main.run_eval
    allows us to save the results to a different directory (should probaby submit a PR to sae_bench)
    """

    sae = sae.to(options.device or DEFAULT_DEVICE_STR)
    sae.fold_W_dec_norm()
    # obnoxiously, torch now won't let us load probes without this
    with torch.serialization.safe_globals([LinearProbe, Linear]):  # type: ignore
        llm_dtype = general_utils.str_to_dtype(options.llm_dtype)
        model = HookedTransformer.from_pretrained_no_processing(
            sae.cfg.model_name, device=options.device, dtype=llm_dtype
        )
        sparse_probing_dir = Path(results_dir) / SPARSE_PROBING_EXPERIMENT_NAME
        feature_absorption_dir = Path(results_dir) / FEATURE_ABSORPTION_EXPERIMENT_NAME
        run_k_sparse_probing_experiment(
            model=model,
            sae=sae,
            layer=sae.cfg.hook_layer,
            sae_name=options.sae_name,
            force=options.force_rerun,
            max_k_value=options.max_k_value,
            f1_jump_threshold=options.f1_jump_threshold,
            prompt_template=options.prompt_template,
            prompt_token_pos=options.prompt_token_pos,
            device=options.device or DEFAULT_DEVICE_STR,
            k_sparse_probe_l1_decay=options.k_sparse_probe_l1_decay,
            k_sparse_probe_batch_size=options.k_sparse_probe_batch_size,
            k_sparse_probe_num_epochs=options.k_sparse_probe_num_epochs,
            probes_dir=probes_dir or PROBES_DIR,
            experiment_dir=sparse_probing_dir,
        )
        raw_df = run_feature_absortion_experiment(
            model=model,
            sae=sae,
            layer=sae.cfg.hook_layer,
            sae_name=options.sae_name,
            force=options.force_rerun,
            max_k_value=options.max_k_value,
            feature_split_f1_jump_threshold=options.f1_jump_threshold,
            prompt_template=options.prompt_template,
            prompt_token_pos=options.prompt_token_pos,
            batch_size=options.llm_batch_size,
            device=options.device or DEFAULT_DEVICE_STR,
            experiment_dir=feature_absorption_dir,
            sparse_probing_experiment_dir=sparse_probing_dir,
            probes_dir=probes_dir or PROBES_DIR,
        )
        agg_df = _aggregate_results_df(raw_df)
    return agg_df
