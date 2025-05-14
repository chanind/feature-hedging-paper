from unittest.mock import patch

import pytest
import torch

from hedging_paper.sparse_probing.parts_of_speech import (
    TaggedSentence,
    TaggedToken,
    create_pos_dataset,
    download_nltk_data,
    find_token_positions,
    process_tagged_sentences_for_activations,
    reconstruct_sentence,
)


def test_download_nltk_data_calls_nltk_download():
    """Test that download_nltk_data calls nltk.download with the correct argument."""
    with patch("nltk.download") as mock_download:
        download_nltk_data()
        mock_download.assert_called_once_with("treebank")


def test_reconstruct_sentence_properly_formats_sentence():
    """Test that reconstruct_sentence correctly formats a sentence from tagged words."""
    tagged_words = [
        ("The", "DT"),
        ("quick", "JJ"),
        ("brown", "JJ"),
        ("fox", "NN"),
        ("jumps", "VBZ"),
        ("over", "IN"),
        ("the", "DT"),
        ("lazy", "JJ"),
        ("dog", "NN"),
        (".", "."),
    ]

    result = reconstruct_sentence(tagged_words)

    # The TreebankWordDetokenizer should produce a properly formatted sentence
    assert result == "The quick brown fox jumps over the lazy dog."


def test_find_token_positions_maps_words_to_tokens(gpt2_tokenizer):
    """Test that find_token_positions correctly maps words to token positions."""
    sentence = "The cat sat on the mat."
    tagged_words = [
        ("The", "DT"),
        ("cat", "NN"),
        ("sat", "VBD"),
        ("on", "IN"),
        ("the", "DT"),
        ("mat", "NN"),
        (".", "."),
    ]

    result = find_token_positions(sentence, tagged_words, gpt2_tokenizer)

    # Check that we have the correct number of results
    assert len(result) == len(tagged_words)

    # Check that each word has the expected token position and tag
    assert result[0].word == "The"
    assert result[0].tag == "DT"
    assert len(result[0].token_positions) == 1  # "The" should be a single token

    assert result[1].word == "cat"
    assert result[1].tag == "NN"

    assert result[-1].word == "."
    assert result[-1].tag == "."

    # Verify that token positions are sequential and don't overlap
    all_positions = []
    for item in result:
        all_positions.extend(item.token_positions)
    assert sorted(all_positions) == all_positions  # Positions should be in order
    assert len(all_positions) == len(set(all_positions))  # No duplicates


def test_find_token_positions_tokens_match_words(gpt2_tokenizer):
    """Test that the tokens at the positions specified for each word detokenize to the expected word."""
    sentence = "The quick brown fox jumps over the lazy dog."
    tagged_words = [
        ("The", "DT"),
        ("quick", "JJ"),
        ("brown", "JJ"),
        ("fox", "NN"),
        ("jumps", "VBZ"),
        ("over", "IN"),
        ("the", "DT"),
        ("lazy", "JJ"),
        ("dog", "NN"),
        (".", "."),
    ]

    result = find_token_positions(sentence, tagged_words, gpt2_tokenizer)

    # For each word, check that the tokens at the specified positions detokenize to the expected word
    for item in result:
        word = item.word
        token_positions = item.token_positions

        # Get the tokens for this word
        tokenized = gpt2_tokenizer(sentence, add_special_tokens=False)
        token_ids = tokenized["input_ids"]
        if not isinstance(token_ids, list):
            token_ids = token_ids.tolist()

        # Extract just the token IDs for this word
        word_token_ids = [token_ids[pos] for pos in token_positions]

        # Decode these tokens back to text
        decoded_word = gpt2_tokenizer.decode(word_token_ids).strip()

        # For some tokenizers, we need to handle whitespace differently
        # The decoded word should either match exactly or contain the original word
        assert word in decoded_word or decoded_word in word, (
            f"Expected '{word}' but got '{decoded_word}'"
        )


def test_find_token_positions_handles_subword_tokenization(gpt2_tokenizer):
    """Test that find_token_positions correctly handles subword tokenization."""
    sentence = "Transformers models are powerful."
    tagged_words = [
        ("Transformers", "NNS"),
        ("models", "NNS"),
        ("are", "VBP"),
        ("powerful", "JJ"),
        (".", "."),
    ]

    result = find_token_positions(sentence, tagged_words, gpt2_tokenizer)

    # Check that "Transformers" might be split into multiple tokens
    assert result[0].word == "Transformers"
    assert len(result[0].token_positions) >= 1  # May be split into multiple tokens

    # Check that all words are found
    words = [item.word for item in result]
    assert words == [word for word, _ in tagged_words]

    # Check that character spans are correct
    assert result[0].char_span == (0, 12)  # "Transformers"
    assert result[1].char_span == (13, 19)  # "models"
    assert result[2].char_span == (20, 23)  # "are"
    assert result[3].char_span == (24, 32)  # "powerful"
    assert result[4].char_span == (32, 33)  # "."


def test_create_pos_dataset_returns_expected_structure(gpt2_tokenizer):
    """Test that create_pos_dataset returns data with the expected structure."""
    # Mock the treebank corpus to return a small set of tagged sentences
    mock_tagged_sents = [
        [("The", "DT"), ("cat", "NN"), ("sat", "VBD"), (".", ".")],
        [("A", "DT"), ("dog", "NN"), ("barked", "VBD"), (".", ".")],
    ]

    dataset = create_pos_dataset(gpt2_tokenizer, mock_tagged_sents)

    # Check the dataset structure
    assert len(dataset) == 2
    assert isinstance(dataset[0], TaggedSentence)
    assert isinstance(dataset[0].tagged_tokens[0], TaggedToken)

    # First sentence should be about a cat
    assert "cat" in dataset[0].sentence
    # Second sentence should be about a dog
    assert "dog" in dataset[1].sentence

    # Check the structure of tagged_tokens
    assert len(dataset[0].tagged_tokens) == 4  # 4 tokens in first sentence

    token = dataset[0].tagged_tokens[0]
    assert isinstance(token.word, str)
    assert isinstance(token.tag, str)
    assert isinstance(token.token_positions, list)
    assert isinstance(token.char_span, tuple)

    # Check that the first word in the first sentence is "The" with tag "DT"
    assert token.word == "The"
    assert token.tag == "DT"


@pytest.mark.parametrize(
    "sentence,tagged_words",
    [
        (
            "The quick brown fox jumps over the lazy dog.",
            [
                ("The", "DT"),
                ("quick", "JJ"),
                ("brown", "JJ"),
                ("fox", "NN"),
                ("jumps", "VBZ"),
                ("over", "IN"),
                ("the", "DT"),
                ("lazy", "JJ"),
                ("dog", "NN"),
                (".", "."),
            ],
        ),
        (
            "Don't go there!",
            [
                ("Do", "VB"),
                ("n't", "RB"),
                ("go", "VB"),
                ("there", "RB"),
                ("!", "."),
            ],
        ),
    ],
)
def test_find_token_positions_with_different_sentences(
    sentence: str, tagged_words: list[tuple[str, str]], gpt2_tokenizer
):
    """Test find_token_positions with different sentence structures."""
    result = find_token_positions(sentence, tagged_words, gpt2_tokenizer)

    # Check that all words are found
    assert len(result) == len(tagged_words)

    # Check that words and tags match
    for i, item in enumerate(result):
        assert item.word == tagged_words[i][0]
        assert item.tag == tagged_words[i][1]

        # Each word should have at least one token position
        assert len(item.token_positions) >= 1

        # Character span should be valid
        start, end = item.char_span
        assert 0 <= start < end <= len(sentence)
        assert sentence[start:end] == item.word


@pytest.fixture
def simple_tagged_sentence(gpt2_tokenizer):
    """Create a simple tagged sentence for testing."""
    sentence = "The cat sat on the mat."
    tagged_words = [
        ("The", "DT"),
        ("cat", "NN"),
        ("sat", "VBD"),
        ("on", "IN"),
        ("the", "DT"),
        ("mat", "NN"),
        (".", "."),
    ]
    tagged_tokens = find_token_positions(sentence, tagged_words, gpt2_tokenizer)
    return TaggedSentence(sentence=sentence, tagged_tokens=tagged_tokens)


def test_process_tagged_sentences_simple(gpt2_model, simple_tagged_sentence):
    """Test that activations are collected for each POS tag in a simple sentence."""
    hook_name = "blocks.0.hook_resid_post"

    pos_activations = process_tagged_sentences_for_activations(
        model=gpt2_model,
        hook_name=hook_name,
        tagged_sentences=[simple_tagged_sentence],
        device="cpu",
    )

    # Check basic properties
    assert len(pos_activations.activations) > 0
    assert len(pos_activations.pos_tags) > 0
    assert set(pos_activations.pos_tags) == {"DT", "NN", "VBD", "IN", "."}

    # Check that each POS tag has at least one activation
    for tag in pos_activations.pos_tags:
        assert tag in pos_activations.pos_indices
        assert len(pos_activations.pos_indices[tag]) > 0

    # The total number of activations should equal the sum of indices for each tag
    total_indices = sum(
        len(indices) for indices in pos_activations.pos_indices.values()
    )
    assert len(pos_activations.activations) == total_indices


def test_process_tagged_sentences_manual_check(gpt2_model, simple_tagged_sentence):
    """Test that activations match the expected values from manual calculation."""
    hook_name = "blocks.0.hook_resid_post"

    # First, run the function and get the activations
    pos_activations = process_tagged_sentences_for_activations(
        model=gpt2_model,
        hook_name=hook_name,
        tagged_sentences=[simple_tagged_sentence],
        device="cpu",
    )

    # Now manually run the model and extract activations for comparison
    inputs = gpt2_model.tokenizer(simple_tagged_sentence.sentence, return_tensors="pt")
    _, cache = gpt2_model.run_with_cache(inputs["input_ids"], names_filter=[hook_name])
    manual_activations = cache[hook_name][0].cpu()

    # Find the final token position for a specific word to check
    test_word_idx = 1  # "cat" (second word)
    test_token = simple_tagged_sentence.tagged_tokens[test_word_idx]
    final_token_pos = test_token.token_positions[-1]

    # Find the corresponding activation in our results
    tag = test_token.tag  # Should be "NN"
    tag_indices = pos_activations.pos_indices[tag]

    # We need to find which index in tag_indices corresponds to our test word
    # This is a bit complex since we don't have a direct mapping in the results
    # Let's check that at least one of the activations for this tag matches

    # Get the actual activation for this token from the manual calculation
    expected_activation = manual_activations[final_token_pos]

    # Check if any of the activations for this tag match the expected activation
    matching_found = False
    for idx in tag_indices:
        actual_activation = pos_activations.activations[idx]
        if torch.allclose(actual_activation, expected_activation):
            matching_found = True
            break

    assert matching_found, "Could not find matching activation for the test word"


def test_process_empty_dataset(gpt2_model):
    """Test handling of empty input."""
    hook_name = "blocks.0.hook_resid_post"

    pos_activations = process_tagged_sentences_for_activations(
        model=gpt2_model,
        hook_name=hook_name,
        tagged_sentences=[],
        device="cpu",
    )

    assert len(pos_activations.activations) == 0
    assert len(pos_activations.pos_tags) == 0
    assert pos_activations.pos_indices == {}


def test_invalid_hook_name(gpt2_model, simple_tagged_sentence):
    """Test behavior with invalid hook name."""
    hook_name = "nonexistent_hook"

    with pytest.raises(KeyError):
        process_tagged_sentences_for_activations(
            model=gpt2_model,
            hook_name=hook_name,
            tagged_sentences=[simple_tagged_sentence],
            device="cpu",
        )
