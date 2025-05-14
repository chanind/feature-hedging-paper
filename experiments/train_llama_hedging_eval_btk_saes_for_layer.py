import time
from pathlib import Path

import fire

from hedging_paper.saes.base_sae import (
    BaseSAERunnerConfig,
)
from hedging_paper.scripts.train_hedging_eval_batch_topk_sae import (
    TrainHedgingEvalBatchTopkSaeOptions,
    train_hedging_eval_batch_topk_sae,
)

EXPERIMENT_NAME = Path(__file__).name
TRAINING_TOKENS = 250_000_000
BATCH_SIZE = 4096
D_IN = 2048
LR = 3e-4
TOTAL_STEPS = TRAINING_TOKENS // BATCH_SIZE
NEW_LATENTS = 64

WIDTH = 8192
SEEDS = [0, 1, 2, 3]
# SEEDS = [0]
K = 25
LAYERS = [0, 7, 14]


def run_experiment(
    base_output_path: str,
    shared_path: str,
):
    script_configs: list[TrainHedgingEvalBatchTopkSaeOptions] = []

    for layer in LAYERS:
        for seed in SEEDS:
            cfg = BaseSAERunnerConfig(
                architecture="topk",
                activation_fn_kwargs={"k": K},
                model_name="meta-llama/Llama-3.2-1B",
                model_class_name="AutoModelForCausalLM",
                hook_name=f"model.layers.{layer}",
                hook_layer=layer,
                dataset_path="chanind/pile-uncopyrighted-llama-3_2-1024-abbrv-1B",
                is_dataset_tokenized=True,
                streaming=False,
                context_size=1024,
                d_in=D_IN,
                d_sae=WIDTH,
                expansion_factor=None,
                training_tokens=TRAINING_TOKENS,
                init_encoder_as_decoder_transpose=True,
                scale_sparsity_penalty_by_decoder_norm=True,
                decoder_heuristic_init=True,
                normalize_sae_decoder=False,
                device="cuda",
                run_name=f"llama-hl-btk-width-{WIDTH}-k-{K}-layer-{layer}-seed-{seed}-{time.strftime('%Y-%m-%dT%H:%M:%S')}",
                lr_warm_up_steps=0,
                lr_decay_steps=0,
                l1_warm_up_steps=0,
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
                TrainHedgingEvalBatchTopkSaeOptions(
                    base_sae_cfg=cfg,
                    base_output_path=f"{base_output_path}/hedging-layers/llama-3.2-1b-btk/layer-{layer}/k-{K}/width-{WIDTH}/seed-{seed}",
                    new_latents=NEW_LATENTS,
                    shared_path=shared_path,
                    run_evals=False,
                )
            )

    for script_config in script_configs:
        train_hedging_eval_batch_topk_sae(script_config)


if __name__ == "__main__":
    fire.Fire(run_experiment)
