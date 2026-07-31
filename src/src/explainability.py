from __future__ import annotations

import numpy as np
import pytest

from src.explainability import (
    ExplainabilityError,
    _normalize_heatmap,
    generate_gradcam,
)


def test_heatmap_normalization_produces_zero_to_one_values() -> None:
    heatmap = np.array(
        [
            [2.0, 4.0],
            [6.0, 10.0],
        ],
        dtype=np.float32,
    )

    normalized = _normalize_heatmap(heatmap)

    assert normalized.shape == (2, 2)
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)


def test_uniform_heatmap_normalizes_to_zero() -> None:
    heatmap = np.full(
        (8, 8),
        fill_value=5.0,
        dtype=np.float32,
    )

    normalized = _normalize_heatmap(heatmap)

    assert normalized.shape == (8, 8)
    assert np.count_nonzero(normalized) == 0


def test_invalid_heatmap_dimensions_are_rejected() -> None:
    with pytest.raises(ExplainabilityError):
        _normalize_heatmap(
            np.zeros((1, 2, 3), dtype=np.float32)
        )


def test_non_finite_heatmap_is_rejected() -> None:
    heatmap = np.zeros((4, 4), dtype=np.float32)
    heatmap[0, 0] = np.nan

    with pytest.raises(ExplainabilityError):
        _normalize_heatmap(heatmap)


def test_real_gradcam_runs_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "RADIANTGUARD_ENABLE_RESEARCH_MODEL",
        "true",
    )

    image = np.linspace(
        0,
        255,
        224 * 224,
        dtype=np.float32,
    ).reshape(224, 224)

    result = generate_gradcam(
        image=image,
        target_label="Pneumonia",
    )

    assert result.label == "Pneumonia"
    assert result.method == "Grad-CAM"
    assert result.target_layer == "features.norm5"

    assert result.heatmap.shape == (224, 224)
    assert result.heatmap.dtype == np.float32
    assert np.isfinite(result.heatmap).all()
    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0

    assert np.isfinite(result.research_score)
    assert 0 <= result.research_score_percent <= 100

    assert any(
        "model influence" in limitation.lower()
        for limitation in result.limitations
    )

    assert any(
        "not confirmed disease location" in limitation.lower()
        for limitation in result.limitations
    )


def test_gradcam_rejects_unknown_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "RADIANTGUARD_ENABLE_RESEARCH_MODEL",
        "true",
    )

    image = np.zeros(
        (224, 224),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="Unknown target label"):
        generate_gradcam(
            image=image,
            target_label="Not A Real Model Label",
        )
