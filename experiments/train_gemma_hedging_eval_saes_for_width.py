import time
from pathlib import Path

import fire

from hedging_paper.saes.base_sae import (
    BaseSAERunnerConfig,
)
from hedging_paper.scripts.train_hedging_eval_saelens_sae import (
    TrainHedgingEvalSaelensSaeOptions,
    train_hedging_eval_saelens_sae,
)

EXPERIMENT_NAME = Path(__file__).name
LAYER = 12
TRAINING_TOKENS = 250_000_000
BATCH_SIZE = 4096
D_IN = 2304
LR = 7e-5
TOTAL_STEPS = TRAINING_TOKENS // BATCH_SIZE
NEW_LATENTS = 64

WIDTHS = [128, 256, 1024, 4096, 16384, 65536]
SEEDS = [0, 1, 2, 3]
# SEEDS = [0]
L1 = 10.0


def run_experiment(
    base_output_path: str,
    shared_path: str,
):
    script_configs: list[TrainHedgingEvalSaelensSaeOptions] = []

    for width in WIDTHS:
        for seed in SEEDS:
            cfg = BaseSAERunnerConfig(
                model_name="google/gemma-2-2b",
                model_class_name="AutoModelForCausalLM",
                hook_name=f"model.layers.{LAYER}",
                hook_layer=LAYER,
                dataset_path="chanind/pile-uncopyrighted-gemma-1024-abbrv-1B",
                is_dataset_tokenized=True,
                streaming=False,
                context_size=1024,
                d_in=D_IN,
                d_sae=width,
                expansion_factor=None,
                training_tokens=TRAINING_TOKENS,
                init_encoder_as_decoder_transpose=True,
                scale_sparsity_penalty_by_decoder_norm=True,
                decoder_heuristic_init=False,
                normalize_sae_decoder=False,
                device="cuda",
                run_name=f"hedging-width-{width}-l1-{L1}-seed-{seed}-{time.strftime('%Y-%m-%dT%H:%M:%S')}",
                lr_warm_up_steps=0,
                lr_decay_steps=0,
                l1_warm_up_steps=10_000,
                l1_coefficient=L1,
                lr=LR,
                n_batches_in_buffer=64,
                train_batch_size_tokens=BATCH_SIZE,
                store_batch_size_prompts=12,
                eval_batch_size_prompts=6,
                wandb_log_frequency=100,
                eval_every_n_wandb_logs=50,
                autocast_lm=True,
                autocast=True,
                n_checkpoints=2,
                exclude_special_tokens=True,
                seed=seed,
            )
            script_configs.append(
                TrainHedgingEvalSaelensSaeOptions(
                    base_sae_cfg=cfg,
                    base_output_path=f"{base_output_path}/hedging-widths/gemma-2-2b/layer-12/l1-{L1}/width-{width}/seed-{seed}",
                    new_latents=NEW_LATENTS,
                    shared_path=shared_path,
                    run_evals=False,
                    reset_l1_warm_up_steps=False,
                    min_l1_during_secondary_warm_up=min(L1, 10.0),
                )
            )

    for script_config in script_configs:
        train_hedging_eval_saelens_sae(script_config)


if __name__ == "__main__":
    fire.Fire(run_experiment)
