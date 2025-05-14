import os
import shutil
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from sae_dashboard.data_writing_fns import save_feature_centric_vis
from sae_dashboard.sae_vis_data import SaeVisConfig, SaeVisData
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_lens import SAE, ActivationsStore
from tqdm import tqdm
from transformer_lens import HookedTransformer

from hedging_paper.util import DEFAULT_DEVICE_STR, dtypify


@dataclass(frozen=True)
class BuildSAEDashboardOptions:
    n_prompts: int = 2500
    store_batch_size_prompts: int = 32
    n_batches_in_buffer: int = 4
    minibatch_size_features: int = 512
    minibatch_size_tokens: int = 512
    quantile_feature_batch_size: int = 16
    device: str | None = None
    dtype: str = "float32"
    context_size: int = 128
    zip_output: bool = True
    features: Iterable[int] | None = None
    ignore_special_tokens: bool = True


def generate_dashboard_data(
    sae: SAE,
    model: HookedTransformer | None = None,
    device: str | None = None,
    dtype: str = "float32",
    n_prompts: int = 5000,
    minibatch_size_features: int = 1024,
    minibatch_size_tokens: int = 1024,
    store_batch_size_prompts: int = 16,
    n_batches_in_buffer: int = 8,
    context_size: int = 256,
    ignore_special_tokens: bool = True,
    features: Iterable[int] | None = None,
    quantile_feature_batch_size: int = 16,
) -> SaeVisData:
    if model is None:
        model = HookedTransformer.from_pretrained(
            sae.cfg.model_name,
            device=device,
            dtype=dtype,
            **sae.cfg.model_from_pretrained_kwargs,
        )
    sae = sae.to(device, dtype=dtypify(dtype))
    model = model.to(device).to(dtypify(dtype))  # type: ignore

    sae.fold_W_dec_norm()

    assert model is not None
    assert model.tokenizer is not None
    activations_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        context_size=context_size,
        store_batch_size_prompts=store_batch_size_prompts,
        n_batches_in_buffer=n_batches_in_buffer,
        device=device or DEFAULT_DEVICE_STR,
    )

    all_tokens_list = []
    pbar = tqdm(range(n_prompts))
    for _ in pbar:
        batch_tokens = activations_store.get_batch_tokens().cpu()
        batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
            : batch_tokens.shape[0]
        ]
        all_tokens_list.append(batch_tokens)
    token_dataset = torch.cat(all_tokens_list, dim=0)
    token_dataset = token_dataset[torch.randperm(token_dataset.shape[0])]

    feature_vis_config = SaeVisConfig(
        hook_point=sae.cfg.hook_name,
        features=features or range(sae.cfg.d_sae),
        minibatch_size_features=minibatch_size_features,
        minibatch_size_tokens=minibatch_size_tokens,
        verbose=True,
        device=device or DEFAULT_DEVICE_STR,
        # cache_dir=Path("/content/cache"),
        dtype=dtype,
        ignore_tokens=set(model.tokenizer.all_special_ids)
        if ignore_special_tokens
        else set(),
        quantile_feature_batch_size=quantile_feature_batch_size,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        data = SaeVisRunner(feature_vis_config).run(
            encoder=sae,
            model=model,
            tokens=token_dataset,
        )
    return data


def save_feature_dashboards(
    data: SaeVisData, output_path: Path | str, zip_output: bool = False
) -> str:
    output_path = Path(output_path)
    filename = output_path / "dashboard.html"
    output_dir = filename.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_feature_centric_vis(sae_vis_data=data, filename=filename, separate_files=True)

    if zip_output:
        output_path = Path(output_path)

        # Create zip file
        with zipfile.ZipFile(
            str(output_path) + ".zip", "w", zipfile.ZIP_DEFLATED
        ) as zipf:
            # Walk through directory and add files to zip
            for root, dirs, files in os.walk(str(output_path)):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, str(output_path))
                    zipf.write(file_path, arcname)

        # Delete original directory after zipping

        shutil.rmtree(str(output_path))

    return str(output_path) + (".zip" if zip_output else "")


def build_sae_dashboard(
    sae: SAE,
    output_path: Path | str,
    options: BuildSAEDashboardOptions = BuildSAEDashboardOptions(),
    model: HookedTransformer | None = None,
) -> str:
    """
    Helper function to build and save a feature dashboard in one step.
    """
    data = generate_dashboard_data(
        sae=sae,
        model=model,
        n_prompts=options.n_prompts,
        store_batch_size_prompts=options.store_batch_size_prompts,
        n_batches_in_buffer=options.n_batches_in_buffer,
        minibatch_size_features=options.minibatch_size_features,
        minibatch_size_tokens=options.minibatch_size_tokens,
        device=options.device,
        dtype=options.dtype,
        context_size=options.context_size,
        features=options.features,
        ignore_special_tokens=options.ignore_special_tokens,
        quantile_feature_batch_size=options.quantile_feature_batch_size,
    )
    return save_feature_dashboards(data, output_path, options.zip_output)
