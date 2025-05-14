import time
from pathlib import Path

from hedging_paper.saes.matryoshka_sae import (
    MatryoshkaSAERunnerConfig,
)
from hedging_paper.scripts.train_batch_topk_matryoshka_sae import (
    TrainBatchTopkMatryoshkaSaeOptions,
    train_batch_topk_matryoshka_sae,
)

EXPERIMENT_NAME = Path(__file__).name
LAYER = 12
TRAINING_TOKENS = 500_000_000
D_IN = 2304
BATCH_SIZE = 4096
D_SAE = 16 * 2048
LR = 1e-4
AUX = 1 / 32
KS = [40]
ALL_STEPS = [
    D_SAE // 256,  # 128
    D_SAE // 64,  # 512
    D_SAE // 16,  # 2048
    D_SAE // 4,  # 8192
    D_SAE,  # 32768
]
TOTAL_STEPS = TRAINING_TOKENS // BATCH_SIZE

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MULTIPLIERS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]
COMPOUND_MULTIPLIERS = [True]

script_configs: list[TrainBatchTopkMatryoshkaSaeOptions] = []

for seed in SEEDS:
    for k in KS:
        for compound_multipliers in COMPOUND_MULTIPLIERS:
            for multiplier in MULTIPLIERS:
                steps = ALL_STEPS[:-1]
                multipliers = [multiplier] * len(steps)
                if compound_multipliers:
                    multipliers = []
                    cur_multiplier = 1.0
                    for i in range(len(steps)):
                        cur_multiplier *= multiplier
                        multipliers.insert(0, cur_multiplier)
                compound_suffix = "-compound" if compound_multipliers else ""

                cfg = MatryoshkaSAERunnerConfig(
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
                    d_sae=D_SAE,
                    expansion_factor=None,
                    training_tokens=TRAINING_TOKENS,
                    init_encoder_as_decoder_transpose=True,
                    scale_sparsity_penalty_by_decoder_norm=False,
                    decoder_heuristic_init=True,
                    normalize_sae_decoder=False,
                    device="cuda",
                    run_name=f"btk-balanced-matyoshka-4x-layer-{LAYER}-mul-{multiplier}{compound_suffix}-k-{k}-seed-{seed}-{time.strftime('%Y-%m-%dT%H:%M:%S')}",
                    lr_warm_up_steps=0,
                    lr_decay_steps=0,
                    aux_coefficient=AUX,
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
                    # matryoshka stuff
                    matryoshka_steps=steps,
                    include_matryoshka_inner_l1_loss=False,
                    normalize_losses_by_num_matryoshka_steps=multiplier > 1,
                    use_matryoshka_aux_loss=True,
                    skip_outer_loss=False,
                    matryoshka_inner_loss_multipliers=multipliers,
                )
                script_configs.append(
                    TrainBatchTopkMatryoshkaSaeOptions(
                        sae_cfg=cfg,
                        output_path=f"/home/dev/project-storage/saes/pile/gemma-2-2b-btk-balanced-4x-matryoshka/multiplier-{multiplier}{compound_suffix}/seed-{seed}",
                        shared_path="/home/dev/project-storage/shared",
                    )
                )


if __name__ == "__main__":
    for script_config in script_configs:
        train_batch_topk_matryoshka_sae(script_config)
