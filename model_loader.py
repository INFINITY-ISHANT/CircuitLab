"""One shared GPT-2 Small instance per Python process.

This module only loads the model; it never performs an experimental forward pass.
Callers should not move the shared model to another device or enable training mode.
"""

from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

_model: "HookedTransformer | None" = None
_model_lock = Lock()


def get_model() -> "HookedTransformer":
    """Load once, select CUDA when available, and return the model in eval mode.

    Imports and checkpoint loading are lazy. Failed loads are not cached, so a
    later call can retry. The lock prevents concurrent callers loading twice.
    """
    global _model
    with _model_lock:
        if _model is None:
            import torch
            from transformer_lens import HookedTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = HookedTransformer.from_pretrained("gpt2", device=device)
            model.eval()
            _model = model
        else:
            _model.eval()
        return _model
