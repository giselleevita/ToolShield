"""Regression coverage for experiment seed propagation."""

from unittest.mock import patch

from toolshield.models.transformer import TransformerClassifier


def test_transformer_prefers_explicit_seed() -> None:
    with patch("toolshield.models.transformer.torch.manual_seed") as manual_seed:
        model = TransformerClassifier({"seed": 7, "random_state": 99})

    assert model.seed == 7
    manual_seed.assert_called_with(7)


def test_transformer_accepts_sklearn_random_state() -> None:
    with patch("toolshield.models.transformer.torch.manual_seed") as manual_seed:
        model = TransformerClassifier({"random_state": 11})

    assert model.seed == 11
    manual_seed.assert_called_with(11)
