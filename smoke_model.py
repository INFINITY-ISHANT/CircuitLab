"""Run with: python smoke_model.py (downloads GPT-2 on first use)."""

from model_loader import get_model

PROMPT = "When Mary and John went to the store, John gave a drink to"


def main() -> None:
    import torch

    model = get_model()
    print(f"Chosen device: {model.cfg.device}", flush=True)
    assert get_model() is model, "Loader did not reuse the model"
    assert not model.training, "Model must be in eval mode"

    tokens = model.to_tokens(PROMPT)
    with torch.inference_mode():
        logits = model(tokens, return_type="logits")

    expected = (tokens.shape[0], tokens.shape[1], model.cfg.d_vocab)
    assert tokens.ndim == 2 and tokens.shape[0] == 1
    assert logits.ndim == 3 and tuple(logits.shape) == expected, (
        f"Expected logits shape {expected}, got {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all().item(), "Non-finite logits"
    print(f"Logits shape [batch, sequence, vocab]: {tuple(logits.shape)}")
    print("Validation forward calls: 1 (batch size 1)")
    print("Top 5 next-token predictions:")
    values, indices = logits[0, -1].softmax(dim=-1).topk(5)
    for token_id, probability in zip(indices.tolist(), values.tolist()):
        token = model.to_string([token_id])
        print(f"  {token!r:20} id={token_id:5d} probability={probability:.4f}")


if __name__ == "__main__":
    main()
