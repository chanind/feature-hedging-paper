import time
from pathlib import Path

from hedging_paper.saes.base_sae import (
    BaseSAERunnerConfig,
)
from hedging_paper.scripts.train_hedging_eval_batch_topk_sae import (
    TrainHedgingEvalBatchTopkSaeOptions,
    train_hedging_eval_batch_topk_sae,
)

EXPERIMENT_NAME = Path(__file__).name
LAYER = 12
TRAINING_TOKENS = 250_000_000
BATCH_SIZE = 4096
D_IN = 2304
LR = 3e-4
TOTAL_STEPS = TRAINING_TOKENS // BATCH_SIZE
NEW_LATENTS = 64

WIDTHS = [8192]
NEW_LATENTS = [1, 4, 16, 64, 128, 256, 512, 1024]
SEEDS = [0]
KS = [25]

BASE_PATH = (
    "/home/dev/project-storage/saes/paper/hedging-nlatents/gemma-2-2b-btk/layer-12"
)

script_configs: list[TrainHedgingEvalBatchTopkSaeOptions] = []

for width in WIDTHS:
    for k in KS:
        for nlatents in NEW_LATENTS:
            for seed in SEEDS:
                cfg = BaseSAERunnerConfig(
                    architecture="topk",
                    activation_fn_kwargs={"k": k},
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
                    decoder_heuristic_init=True,
                    normalize_sae_decoder=False,
                    device="cuda",
                    run_name=f"btk-hedging-width-{width}-k-{k}-nlatents-{nlatents}-seed-{seed}-{time.strftime('%Y-%m-%dT%H:%M:%S')}",
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
                        base_output_path=f"{BASE_PATH}/k-{k}/width-{width}/nlatents-{nlatents}/seed-{seed}",
                        new_latents=nlatents,
                        shared_path="/home/dev/project-storage/shared",
                        run_evals=False,
                    )
                )

if __name__ == "__main__":
    for script_config in script_configs:
        train_hedging_eval_batch_topk_sae(script_config)
