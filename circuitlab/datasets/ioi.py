"""Reproducible GPT-2 tokenized datasets for Indirect Object Identification.

The clean prompt has the form ``A and B ... B gave ... to`` and requires A as
its continuation. The paired ABC corruption replaces A with a third distinct
name C while preserving the grammatical subject B and token positions.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch

from circuitlab.model import get_model


NAMES = (
    "Mary", "John", "Alice", "Bob", "Sarah", "James", "Emma", "David",
    "Laura", "Michael", "Anna", "Peter", "Susan", "Robert", "Linda", "Paul",
)
PLACES = (
    "store", "park", "restaurant", "school", "office", "garden",
    "museum", "library", "station", "beach", "market", "cinema",
)
OBJECTS = (
    "drink", "book", "gift", "letter", "snack", "ball", "map", "note",
    "flower", "ticket", "coffee", "photo",
)

# Each template ends immediately before the name the model must supply.
TEMPLATES = (
    "When {io} and {subject} went to the {place}, {subject} gave a {object} to",
    "After {io} and {subject} visited the {place}, {subject} handed a {object} to",
    "While {io} and {subject} were at the {place}, {subject} passed a {object} to",
    "When {io} and {subject} arrived at the {place}, {subject} brought a {object} to",
    "After {io} met {subject} at the {place}, {subject} sent a {object} to",
    "When {io} and {subject} left the {place}, {subject} offered a {object} to",
    "After {io} and {subject} walked through the {place}, {subject} showed a {object} to",
    "When {io} and {subject} sat near the {place}, {subject} read a {object} to",
    "After {io} and {subject} talked at the {place}, {subject} delivered a {object} to",
    "When {io} and {subject} stopped by the {place}, {subject} lent a {object} to",
    "After {io} and {subject} returned from the {place}, {subject} gave a {object} to",
    "When {io} and {subject} waited outside the {place}, {subject} brought a {object} to",
)


@dataclass
class IOIDataset:
    """A paired clean/ABC-corrupted IOI dataset ready for later interventions."""

    clean_prompts: list[str]
    corrupted_prompts: list[str]
    clean_tokens: torch.Tensor
    corrupted_tokens: torch.Tensor
    io_names: list[str]
    subject_names: list[str]
    io_token_ids: torch.Tensor
    subject_token_ids: torch.Tensor
    seed: int
    metadata: dict[str, Any]

    @property
    def dataset_id(self) -> str:
        """Stable identifier reserved for later dataset registration code."""
        return str(self.metadata["dataset_id"])

    def __len__(self) -> int:
        return len(self.clean_prompts)


def _single_name_token_id(name: str) -> int:
    """Return GPT-2's single leading-space token for a proper name."""
    model = get_model()
    tokens = model.to_tokens(f" {name}", prepend_bos=False, move_to_device=False)
    if tokens.numel() != 1:
        raise ValueError(
            f"Name {name!r} does not map to one GPT-2 leading-space token: {tokens.tolist()}"
        )
    token_id = int(tokens.item())
    if not 0 <= token_id < model.cfg.d_vocab:
        raise ValueError(f"GPT-2 token ID out of range for {name!r}: {token_id}")
    return token_id


def _tokenize_prompts(prompts: list[str]) -> torch.Tensor:
    """Tokenize on CPU; this deliberately does not execute a model forward pass."""
    model = get_model()
    return model.to_tokens(prompts, prepend_bos=True, move_to_device=False).cpu()


def make_ioi_dataset(n: int = 100, seed: int = 0) -> IOIDataset:
    """Generate ``n`` reproducible clean/ABC-corrupted GPT-2 IOI examples.

    Each clean prompt uses names A and B. Its corruption uses C and B, where
    A, B, and C are distinct. This keeps B as the grammatical subject in both
    prompts and preserves name-token positions for later activation patching.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    rng = random.Random(seed)
    clean_prompts: list[str] = []
    corrupted_prompts: list[str] = []
    io_names: list[str] = []
    subject_names: list[str] = []
    corrupted_io_names: list[str] = []
    template_indices: list[int] = []

    for _ in range(n):
        io_name, subject_name, corrupt_io_name = rng.sample(NAMES, 3)
        template_index = rng.randrange(len(TEMPLATES))
        template = TEMPLATES[template_index]
        place = rng.choice(PLACES)
        item = rng.choice(OBJECTS)

        clean_prompts.append(template.format(io=io_name, subject=subject_name, place=place, object=item))
        corrupted_prompts.append(
            template.format(io=corrupt_io_name, subject=subject_name, place=place, object=item)
        )
        io_names.append(io_name)
        subject_names.append(subject_name)
        corrupted_io_names.append(corrupt_io_name)
        template_indices.append(template_index)

    clean_tokens = _tokenize_prompts(clean_prompts)
    corrupted_tokens = _tokenize_prompts(corrupted_prompts)
    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError("Clean and corrupted token tensors must have equal shapes.")

    io_token_ids = torch.tensor([_single_name_token_id(name) for name in io_names], dtype=torch.long)
    subject_token_ids = torch.tensor(
        [_single_name_token_id(name) for name in subject_names], dtype=torch.long
    )
    corrupted_io_token_ids = torch.tensor(
        [_single_name_token_id(name) for name in corrupted_io_names], dtype=torch.long
    )
    model = get_model()
    # GPT-2 uses its end-of-text token for both BOS and padding. Compute each
    # unpadded length directly rather than inferring it from padded batch IDs.
    prompt_lengths = [
        model.to_tokens(prompt, prepend_bos=True, move_to_device=False).shape[1]
        for prompt in clean_prompts
    ]

    return IOIDataset(
        clean_prompts=clean_prompts,
        corrupted_prompts=corrupted_prompts,
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        io_names=io_names,
        subject_names=subject_names,
        io_token_ids=io_token_ids,
        subject_token_ids=subject_token_ids,
        seed=seed,
        metadata={
            "task": "ioi",
            "dataset_id": f"ioi-{seed}-{n}",
            "template_indices": template_indices,
            "corrupted_io_names": corrupted_io_names,
            "corrupted_io_token_ids": corrupted_io_token_ids,
            "prompt_lengths": prompt_lengths,
            "answer_positions": [length - 1 for length in prompt_lengths],
            "prepend_bos": True,
            "padding_side": model.tokenizer.padding_side,
        },
    )
