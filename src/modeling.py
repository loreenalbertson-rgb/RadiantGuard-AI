from __future__ import annotations

import importlib.util
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import numpy as np


class ModelUnavailableError(RuntimeError):
    """Raised when research inference is requested but is not safely available."""


@dataclass(frozen=True)
class ModelStatus:
    name: str
    version: str
    available: bool
    mode: str
    description: str


@dataclass(frozen=True)
class ResearchPrediction:
    label: str
    confidence: float
    explanation: str

    @property
    def confidence_percent(self) -> int:
        """Return a display-safe 0–100 score.

        The underlying value is a research-model score, not a calibrated
        probability or diagnostic confidence estimate.
        """

        bounded = min(max(float(self.confidence), 0.0), 1.0)
        return round(bounded * 100)


@dataclass(frozen=True)
class ResearchModelResult:
    model_name: str
    model_version: str
    mode: str
    predictions: tuple[ResearchPrediction, ...]
    limitations: tuple[str, ...]
    is_simulated: bool
    preprocessing: tuple[str, ...] = ()


FEATURE_FLAG: Final[str] = "RADIANTGUARD_ENABLE_RESEARCH_MODEL"
MODEL_NAME: Final[str] = "TorchXRayVision DenseNet-121"
MODEL_WEIGHTS: Final[str] = "densenet121-res224-all"
MODEL_INPUT_SIZE: Final[int] = 224
MAX_DISPLAY_PREDICTIONS: Final[int] = 6

_INFERENCE_LOCK = threading.Lock()


def _feature_switch_enabled() -> bool:
    value = os.getenv(FEATURE_FLAG, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _optional_dependencies_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchxrayvision") is not None
    )


def get_model_status() -> ModelStatus:
    """Return availability without importing heavyweight model packages.

    The research model remains unavailable unless:
    1. the explicit environment feature switch is enabled, and
    2. the optional model dependencies are installed.
    """

    if not _feature_switch_enabled():
        # Preserve the safe foundation behavior and existing regression tests.
        return ModelStatus(
            name="RadiantGuard Chest Research Model",
            version="not-connected",
            available=False,
            mode="Foundation scaffold",
            description=(
                "The real research-model runner is installed in the codebase, "
                "but its feature switch remains off."
            ),
        )

    if not _optional_dependencies_available():
        return ModelStatus(
            name=MODEL_NAME,
            version=MODEL_WEIGHTS,
            available=False,
            mode="Dependencies unavailable",
            description=(
                "The research feature switch is on, but PyTorch and "
                "TorchXRayVision are not installed in this environment."
            ),
        )

    return ModelStatus(
        name=MODEL_NAME,
        version=MODEL_WEIGHTS,
        available=True,
        mode="Research baseline",
        description=(
            "A public pretrained chest X-ray research baseline is available. "
            "Its output is not clinically validated and is not diagnostic."
        ),
    )


def get_simulated_demo_result() -> ResearchModelResult:
    """Return clearly labeled sample data for interface development.

    This output is not produced from an uploaded image and must never
    be represented as medical-image analysis.
    """

    predictions = (
        ResearchPrediction(
            label="Airspace opacity",
            confidence=0.72,
            explanation=(
                "Simulated example showing how a future research score "
                "could be displayed with uncertainty."
            ),
        ),
        ResearchPrediction(
            label="Pleural effusion",
            confidence=0.18,
            explanation=(
                "Simulated low-score example for interface testing."
            ),
        ),
        ResearchPrediction(
            label="Cardiomegaly",
            confidence=0.10,
            explanation=(
                "Simulated low-score example for interface testing."
            ),
        ),
    )

    limitations = (
        "These values are hard-coded demonstration data.",
        "No uploaded image was analyzed.",
        "This output cannot support diagnosis or treatment.",
        "Future research output must be reviewed by a physician or radiologist.",
    )

    return ResearchModelResult(
        model_name="RadiantGuard Simulated UI Demo",
        model_version="demo-0.1",
        mode="SIMULATED — NOT IMAGE ANALYSIS",
        predictions=predictions,
        limitations=limitations,
        is_simulated=True,
        preprocessing=("No image preprocessing occurred.",),
    )


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("The model input must be a NumPy array.")

    if image.size == 0:
        raise ValueError("The model input cannot be empty.")

    array = np.asarray(image)

    if array.ndim not in {2, 3}:
        raise ValueError(
            "The model input must be a two-dimensional grayscale image "
            "or a three-dimensional color image."
        )

    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("The model input must contain numeric pixel values.")

    if not np.isfinite(array.astype(np.float32, copy=False)).all():
        raise ValueError("The model input contains non-finite pixel values.")

    return array


def _to_grayscale(array: np.ndarray) -> np.ndarray:
    """Convert common image layouts to one two-dimensional grayscale array."""

    if array.ndim == 2:
        return array.astype(np.float32, copy=False)

    # Common channel-last layout: H × W × C.
    if array.shape[-1] in {1, 3, 4}:
        channel_last = array.astype(np.float32, copy=False)

        if channel_last.shape[-1] == 1:
            return channel_last[..., 0]

        rgb = channel_last[..., :3]
        return (
            0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]
        )

    # Common channel-first layout: C × H × W.
    if array.shape[0] in {1, 3, 4}:
        channel_first = array.astype(np.float32, copy=False)

        if channel_first.shape[0] == 1:
            return channel_first[0]

        rgb = channel_first[:3]
        return (
            0.299 * rgb[0]
            + 0.587 * rgb[1]
            + 0.114 * rgb[2]
        )

    raise ValueError(
        "The color image must use a recognized channel-first or "
        "channel-last layout."
    )


def _scale_to_uint8_range(image: np.ndarray) -> tuple[np.ndarray, str]:
    """Scale numeric image values to the 0–255 display range.

    The live application supplies display-ready pixels. This fallback keeps
    the runner deterministic for other callers while documenting any rescale.
    """

    image = image.astype(np.float32, copy=False)
    minimum = float(image.min())
    maximum = float(image.max())

    if 0.0 <= minimum and maximum <= 1.0:
        return image * 255.0, "Scaled normalized 0–1 pixels to 0–255."

    if 0.0 <= minimum and maximum <= 255.0:
        return image, "Used existing 0–255 grayscale display pixels."

    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32), (
            "Input had no dynamic range; converted to a uniform zero image."
        )

    scaled = (image - minimum) / (maximum - minimum)
    return scaled * 255.0, (
        "Min-max scaled pixels to 0–255 for the research baseline."
    )


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the CPU research model and its pretrained weights."""

    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights=MODEL_WEIGHTS)
    model.eval()
    model.to("cpu")
    return model


def _preprocess_image(image: np.ndarray):
    """Apply the preprocessing chain expected by TorchXRayVision."""

    import torch
    import torchxrayvision as xrv

    validated = _validate_image(image)
    grayscale = _to_grayscale(validated)
    scaled, scale_note = _scale_to_uint8_range(grayscale)

    normalized = xrv.datasets.normalize(
        scaled.astype(np.float32, copy=False),
        maxval=255,
    )

    channel_first = normalized[None, ...]
    center_crop = xrv.datasets.XRayCenterCrop()
    resize = xrv.datasets.XRayResizer(MODEL_INPUT_SIZE)

    transformed = center_crop(channel_first)
    transformed = resize(transformed)

    tensor = torch.from_numpy(
        np.ascontiguousarray(transformed)
    ).float().unsqueeze(0)

    notes = (
        "Converted the supplied display image to grayscale.",
        scale_note,
        "Applied TorchXRayVision normalization.",
        "Applied center cropping.",
        f"Resized to 1 × {MODEL_INPUT_SIZE} × {MODEL_INPUT_SIZE} pixels.",
        "Ran inference on CPU.",
    )

    return tensor, notes


def run_research_model(image: np.ndarray) -> ResearchModelResult:
    """Run the public pretrained chest X-ray research baseline.

    This function stays inaccessible until the explicit feature switch is on.
    Scores are research outputs only. They are not diagnoses, calibrated
    probabilities, or a replacement for radiologist interpretation.
    """

    validated = _validate_image(image)
    status = get_model_status()

    if not status.available:
        raise ModelUnavailableError(status.description)

    import torch

    tensor, preprocessing = _preprocess_image(validated)
    model = _load_model()

    with _INFERENCE_LOCK:
        with torch.inference_mode():
            output = model(tensor)[0].detach().cpu().numpy()

    if output.ndim != 1:
        raise RuntimeError(
            f"Unexpected model output shape: {tuple(output.shape)}"
        )

    if len(output) != len(model.pathologies):
        raise RuntimeError(
            "The model output count does not match its pathology labels."
        )

    if not np.isfinite(output).all():
        raise RuntimeError("The model returned non-finite research scores.")

    ranked = sorted(
        zip(model.pathologies, output.tolist(), strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    predictions = tuple(
        ResearchPrediction(
            label=str(label),
            confidence=float(score),
            explanation=(
                "Raw label-specific output from a public pretrained "
                "research model. This score is not a calibrated probability "
                "and does not establish that the finding is present."
            ),
        )
        for label, score in ranked[:MAX_DISPLAY_PREDICTIONS]
    )

    limitations = (
        "This is an unvalidated educational research baseline, not a medical device.",
        "Model scores are not calibrated probabilities or diagnoses.",
        "The model can produce false positives, false negatives, and dataset-specific bias.",
        "Results may be unreliable for synthetic images, pediatric studies, non-frontal views, artifacts, or out-of-distribution inputs.",
        "This build does not localize findings or prove that a highlighted region supports a label.",
        "Every image and result must be independently reviewed by a qualified physician or radiologist.",
    )

    return ResearchModelResult(
        model_name=MODEL_NAME,
        model_version=MODEL_WEIGHTS,
        mode="UNVALIDATED RESEARCH OUTPUT — NOT DIAGNOSTIC",
        predictions=predictions,
        limitations=limitations,
        is_simulated=False,
        preprocessing=preprocessing,
    )
