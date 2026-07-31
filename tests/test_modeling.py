import numpy as np
import pytest

from src.modeling import (
    ModelUnavailableError,
    get_model_status,
    get_simulated_demo_result,
    run_research_model,
)


def test_model_is_disabled_until_intentionally_connected() -> None:
    status = get_model_status()

    assert status.available is False
    assert status.version == "not-connected"
    assert status.mode == "Foundation scaffold"


def test_demo_result_is_clearly_labeled_simulated() -> None:
    result = get_simulated_demo_result()

    assert result.is_simulated is True
    assert "SIMULATED" in result.mode
    assert len(result.predictions) > 0

    assert any(
        "No uploaded image was analyzed" in limitation
        for limitation in result.limitations
    )


def test_research_model_refuses_inference_when_unavailable() -> None:
    image = np.zeros((224, 224), dtype=np.uint8)

    with pytest.raises(ModelUnavailableError):
        run_research_model(image)


def test_research_model_rejects_non_array_input() -> None:
    with pytest.raises(TypeError):
        run_research_model("not an image")  # type: ignore[arg-type]


def test_research_model_rejects_empty_image() -> None:
    image = np.array([], dtype=np.uint8)

    with pytest.raises(ValueError):
        run_research_model(image)
