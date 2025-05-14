from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nltk
import torch
from dataclasses_json import DataClassJsonMixin
from nltk.tokenize.treebank import TreebankWordDetokenizer
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


@dataclass
class TaggedToken(DataClassJsonMixin):
    """Represents a word with its part-of-speech tag and token information."""

    word: str
    tag: str
    token_positions: list[int]
    char_span: tuple[int, int]


@dataclass
class TaggedSentence(DataClassJsonMixin):
    """Represents a sentence with its tagged tokens."""

    sentence: str
    tagged_tokens: list[TaggedToken]


@dataclass
class POSMetadata(DataClassJsonMixin):
    """Metadata for POS activations."""

    pos_tags: list[str]
    pos_indices: dict[str, list[int]]
    tagged_sentences: list[TaggedSentence]


@dataclass
class POSActivations:
    """Container for activations by part of speech."""

    activations: torch.Tensor  # Shape: [num_samples, hidden_dim]
    pos_tags: list[str]  # List of all unique POS tags
    pos_indices: dict[
        str, list[int]
    ]  # Mapping from POS tag to indices in activations tensor
    tagged_sentences: list[TaggedSentence]  # Original tagged sentences


def download_nltk_data() -> None:
    """Download required NLTK data if not already present."""
    nltk.download("treebank")


def reconstruct_sentence(tagged_words: list[tuple[str, str]]) -> str:
    """
    Reconstruct a readable sentence from tagged words using TreebankWordDetokenizer.

    Args:
        tagged_words: list of (word, tag) tuples

    Returns:
        A properly formatted sentence string
    """
    detokenizer = TreebankWordDetokenizer()
    words = [word for word, _ in tagged_words]
    return detokenizer.detokenize(words)


def find_token_positions(
    sentence: str,
    tagged_words: list[tuple[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> list[TaggedToken]:
    """
    Find token positions for each tagged word in the sentence.

    Args:
        sentence: The reconstructed sentence
        tagged_words: list of (word, tag) tuples
        tokenizer: The tokenizer to use

    Returns:
        list of TaggedToken objects containing word, tag, and token positions
    """
    # Tokenize the full sentence
    tokenized = tokenizer(sentence, return_offsets_mapping=True)
    # Cast to BatchEncoding to help type checker
    tokenized_batch = cast(BatchEncoding, tokenized)

    # Extract input_ids and convert to list to satisfy type checker
    input_ids = tokenized_batch["input_ids"]
    if not isinstance(input_ids, list):
        input_ids = input_ids.tolist()  # type: ignore

    # Extract offset_mapping and ensure it's a list
    offset_mapping = tokenized_batch["offset_mapping"]
    if not isinstance(offset_mapping, list):
        offset_mapping = offset_mapping.tolist()  # type: ignore

    result = []
    current_pos = 0

    for word, tag in tagged_words:
        # Find the word in the sentence starting from current_pos
        word_start = sentence.find(word, current_pos)
        if word_start == -1:
            # Skip if word not found (shouldn't happen with properly reconstructed sentence)
            continue

        word_end = word_start + len(word)
        current_pos = word_end

        # Find which tokens overlap with this word
        token_positions = []
        for i, (start, end) in enumerate(offset_mapping):
            # Check if this token overlaps with the word
            if end > word_start and start < word_end:
                token_positions.append(i)

        result.append(
            TaggedToken(
                word=word,
                tag=tag,
                token_positions=token_positions,
                char_span=(word_start, word_end),
            )
        )

    return result


def get_treebank_tagged_sents() -> list[list[tuple[str, str]]]:
    """
    Get tagged sentences from the NLTK treebank corpus.

    Returns:
        List of sentences, where each sentence is a list of (word, tag) tuples
    """
    # Ensure treebank data is downloaded
    download_nltk_data()
    return nltk.corpus.treebank.tagged_sents()


def create_pos_dataset(
    tokenizer: PreTrainedTokenizerBase,
    tagged_sents: list[list[tuple[str, str]]],
) -> list[TaggedSentence]:
    """
    Create a dataset from tagged sentences.

    Args:
        tokenizer: The tokenizer to use
        tagged_sents: Optional list of tagged sentences. If None, uses NLTK treebank.

    Returns:
        list of TaggedSentence objects containing sentence and tagged token information
    """

    dataset = []
    for tagged_words in tagged_sents:
        # Reconstruct the sentence using TreebankWordDetokenizer
        sentence = reconstruct_sentence(tagged_words)

        # Find token positions for each word
        tagged_tokens = find_token_positions(sentence, tagged_words, tokenizer)

        dataset.append(TaggedSentence(sentence=sentence, tagged_tokens=tagged_tokens))

    return dataset


@torch.inference_mode()
def process_tagged_sentences_for_activations(
    model: HookedTransformer,
    hook_name: str,
    tagged_sentences: list[TaggedSentence],
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
) -> POSActivations:
    """
    Process a list of tagged sentences and collect activations for each part of speech.
    Only collects activations for the final token of each word.

    Args:
        model: The HookedTransformer model to run
        hook_name: The name of the hook to collect activations from
        tagged_sentences: List of TaggedSentence objects
        device: Device to run model on

    Returns:
        POSActivations object containing:
        - activations tensor [num_samples, hidden_dim]
        - list of unique POS tags
        - mapping from POS tag to indices in activations tensor
        - tagged sentences with token position information
    """
    # Prepare containers for activations
    all_activations = []
    all_pos_tags = set()
    pos_to_indices: dict[str, list[int]] = {}

    # Track current index in the activations tensor
    current_idx = 0

    for tagged_sentence in tqdm(tagged_sentences):
        # Tokenize the sentence
        if model.tokenizer is None:
            raise ValueError("Model tokenizer is None - model must have a tokenizer")

        inputs = model.tokenizer(tagged_sentence.sentence, return_tensors="pt").to(
            device
        )

        # Run the model and collect activations
        _, cache = model.run_with_cache(
            inputs["input_ids"],
            names_filter=[hook_name],
        )

        # Get activations from the specified hook
        activations = cache[hook_name][0].cpu()  # [seq_len, hidden_dim]

        # Map each token to its POS tag
        for tagged_token in tagged_sentence.tagged_tokens:
            tag = tagged_token.tag
            token_positions = tagged_token.token_positions

            # Skip if there are no token positions
            if not token_positions:
                continue

            # Only use the final token position for this word
            final_pos = token_positions[-1]

            if final_pos >= activations.shape[0]:
                raise ValueError(
                    f"Token position {final_pos} is out of bounds for activations shape {activations.shape}. "
                    f"This should never happen - check tokenization and model processing."
                )

            all_pos_tags.add(tag)
            all_activations.append(activations[final_pos])

            if tag not in pos_to_indices:
                pos_to_indices[tag] = []
            pos_to_indices[tag].append(current_idx)
            current_idx += 1

    # Convert to tensor
    all_activations_tensor = (
        torch.stack(all_activations)
        if all_activations
        else torch.zeros((0, model.cfg.d_model))
    )
    pos_tags_list = sorted(list(all_pos_tags))

    return POSActivations(
        activations=all_activations_tensor,
        pos_tags=pos_tags_list,
        pos_indices=pos_to_indices,
        tagged_sentences=tagged_sentences,
    )


@torch.inference_mode()
def collect_pos_activations(
    model_name: str,
    hook_name: str,
    from_pretrained_kwargs: dict[str, Any],
    cache_dir: Path | str | None = None,
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    force_recompute: bool = False,
) -> POSActivations:
    """
    Run a model on the treebank sentences and collect activations for each part of speech.
    Only collects activations for the final token of each word.

    Args:
        model: The HookedTransformer model to run
        hook_name: The name of the hook to collect activations from
        model_name: Name of the model, used for caching
        cache_dir: Optional directory to cache activations
        device: Device to run model on

    Returns:
        POSActivations object containing:
        - activations tensor [num_samples, hidden_dim]
        - list of unique POS tags
        - mapping from POS tag to indices in activations tensor
        - tagged sentences with token position information
    """
    activations_file: Path | None = None
    metadata_file: Path | None = None

    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    # Check if cache exists and load it if it does
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        activations_file = (
            cache_dir
            / f"pos_activations_{model_name}_{hook_name.replace('.', '_')}.safetensors"
        )
        metadata_file = (
            cache_dir / f"pos_metadata_{model_name}_{hook_name.replace('.', '_')}.json"
        )

        if activations_file.exists() and metadata_file.exists() and not force_recompute:
            print(f"Loading cached activations from {activations_file}")
            with open(metadata_file) as f:
                metadata = POSMetadata.from_json(f.read())
            activations = load_file(activations_file)["activations"]
            return POSActivations(
                activations=activations,
                pos_tags=metadata.pos_tags,
                pos_indices=metadata.pos_indices,
                tagged_sentences=metadata.tagged_sentences,
            )

    model = HookedTransformer.from_pretrained(
        model_name, device=device, **from_pretrained_kwargs
    )
    # Get the tagged sentences
    if model.tokenizer is None:
        raise ValueError("Model tokenizer is None - model must have a tokenizer")

    tagged_sents = get_treebank_tagged_sents()
    tagged_sentences = create_pos_dataset(model.tokenizer, tagged_sents)

    # Process the tagged sentences to get activations
    result = process_tagged_sentences_for_activations(
        model=model,
        hook_name=hook_name,
        tagged_sentences=tagged_sentences,
        device=device,
    )

    # Save to cache if requested
    if (
        cache_dir is not None
        and activations_file is not None
        and metadata_file is not None
    ):
        metadata = POSMetadata(
            pos_tags=result.pos_tags,
            pos_indices=result.pos_indices,
            tagged_sentences=result.tagged_sentences,
        )
        with open(metadata_file, "w") as f:
            f.write(metadata.to_json())
        # Use safetensors for the large tensor
        save_file({"activations": result.activations}, activations_file)

    return result
