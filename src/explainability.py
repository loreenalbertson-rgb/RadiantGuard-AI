from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.modeling import (
    ModelUnavailableError,
    _INFERENCE_LOCK,
    _load_model,
    _preprocess_image,
    _validate_image,
    get_model_status,
)


class ExplainabilityError(RuntimeError):
    """Raised when an explainability map cannot be generated safely."""


@dataclass(frozen=True)
class ExplainabilityResult:
    """A non-diagnostic model-influence visualization."""

    label: str
    research_score: float
    heatmap: np.ndarray
    method: str
    target_layer: str
    preprocessing: tuple[str, ...]
    limitations: tuple[str, ...]

    @property
    def research_score_percent(self) -> int:
        bounded = min(max(float(self.research_score), 0.0), 1.0)
        return round(bounded * 100)


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalize a two-dimensional heatmap to the range 0–1."""

    array = np.asarray(heatmap, dtype=np.float32)

    if array.ndim != 2:
        raise ExplainabilityError(
            "The explainability heatmap must be two-dimensional."
        )

    if not np.isfinite(array).all():
        raise ExplainabilityError(
            "The explainability heatmap contains non-finite values."
        )

    minimum = float(array.min())
    maximum = float(array.max())

    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)

    normalized = (array - minimum) / (maximum - minimum)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def generate_gradcam(
    image: np.ndarray,
    target_label: str,
) -> ExplainabilityResult:
    """Generate a Grad-CAM model-influence map for one research label.

    The map shows which final convolutional features most influenced the
    selected model output. It does not establish disease location, anatomy,
    causality, clinical relevance, or correctness.
    """

    validated = _validate_image(image)
    status = get_model_status()

    if not status.available:
        raise ModelUnavailableError(status.description)

    if not isinstance(target_label, str) or not target_label.strip():
        raise ValueError("A non-empty target label is required.")

    import torch
    import torch.nn.functional as functional

    tensor, preprocessing = _preprocess_image(validated)
    model = _load_model()

    if target_label not in model.pathologies:
        raise ValueError(
            f"Unknown target label: {target_label!r}. "
            "Choose a label returned by the connected research model."
        )

    target_index = model.pathologies.index(target_label)

    try:
        target_layer = model.features.norm5
    except AttributeError as exc:
        raise ExplainabilityError(
            "The connected model does not expose the expected final "
            "DenseNet feature layer."
        ) from exc

    captured: dict[str, object] = {}

    def capture_activation(_module, _inputs, output) -> None:
        captured["activation"] = output

        def capture_gradient(gradient) -> None:
            captured["gradient"] = gradient

        output.register_hook(capture_gradient)

    hook_handle = target_layer.register_forward_hook(capture_activation)

    try:
        with _INFERENCE_LOCK:
            model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                outputs = model(tensor)

                if outputs.ndim != 2 or outputs.shape[0] != 1:
                    raise ExplainabilityError(
                        f"Unexpected model output shape: {tuple(outputs.shape)}"
                    )

                selected_score = outputs[0, target_index]

                if not torch.isfinite(selected_score):
                    raise ExplainabilityError(
                        "The selected model output is not finite."
                    )

                selected_score.backward()

    finally:
        hook_handle.remove()

    activation = captured.get("activation")
    gradient = captured.get("gradient")

    if activation is None or gradient is None:
        raise ExplainabilityError(
            "The model did not expose the activations and gradients "
            "required for Grad-CAM."
        )

    activation_tensor = activation.detach()[0]
    gradient_tensor = gradient.detach()[0]

    if activation_tensor.ndim != 3 or gradient_tensor.ndim != 3:
        raise ExplainabilityError(
            "Unexpected activation or gradient dimensions."
        )

    if activation_tensor.shape != gradient_tensor.shape:
        raise ExplainabilityError(
            "Activation and gradient shapes do not match."
        )

    channel_weights = gradient_tensor.mean(
        dim=(1, 2),
        keepdim=True,
    )

    coarse_map = torch.relu(
        (channel_weights * activation_tensor).sum(dim=0)
    )

    resized_map = functional.interpolate(
        coarse_map[None, None, ...],
        size=(tensor.shape[-2], tensor.shape[-1]),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    heatmap = _normalize_heatmap(
        resized_map.detach().cpu().numpy()
    )

    limitations = (
        "This Grad-CAM map visualizes model influence, not confirmed disease location.",
        "A highlighted region does not prove that the model used medically valid evidence.",
        "The map may emphasize artifacts, borders, text markers, positioning, or synthetic texture.",
        "Grad-CAM is low resolution and depends on the selected layer and model architecture.",
        "The model score and heatmap may both be wrong, unstable, or dataset-specific.",
        "The image, score, and visualization require independent physician or radiologist review.",
    )

    return ExplainabilityResult(
        label=target_label,
        research_score=float(selected_score.detach().cpu().item()),
        heatmap=heatmap,
        method="Grad-CAM",
        target_layer="features.norm5",
        preprocessing=preprocessing,
        limitations=limitations,
    )
