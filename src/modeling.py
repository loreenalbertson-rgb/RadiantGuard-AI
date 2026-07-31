from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


class ModelUnavailableError(RuntimeError):
    """Raised when clinical inference is requested before a model is connected."""


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
        return round(self.confidence * 100)


@dataclass(frozen=True)
class ResearchModelResult:
    model_name: str
    model_version: str
    mode: str
    predictions: tuple[ResearchPrediction, ...]
    limitations: tuple[str, ...]
    is_simulated: bool


MODEL_NAME: Final[str] = "RadiantGuard Chest Research Model"
MODEL_VERSION: Final[str] = "not-connected"


def get_model_status() -> ModelStatus:
    """
    Return the current research-model availability.

    Clinical inference remains disabled until a documented,
    versioned public research model is intentionally connected.
    """

    return ModelStatus(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        available=False,
        mode="Foundation scaffold",
        description=(
            "The research-model interface is installed, but no clinical "
            "inference model is currently connected."
        ),
    )


def get_simulated_demo_result() -> ResearchModelResult:
    """
    Return clearly labeled sample data for interface development.

    This output is not produced from an uploaded image and must never
    be represented as medical-image analysis.
    """

    predictions = (
        ResearchPrediction(
            label="Airspace opacity",
            confidence=0.72,
            explanation=(
                "Simulated example showing how a future research finding "
                "could be displayed with uncertainty."
            ),
        ),
        ResearchPrediction(
            label="Pleural effusion",
            confidence=0.18,
            explanation=(
                "Simulated low-confidence example for interface testing."
            ),
        ),
        ResearchPrediction(
            label="Cardiomegaly",
            confidence=0.10,
            explanation=(
                "Simulated low-confidence example for interface testing."
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
    )


def run_research_model(image: np.ndarray) -> ResearchModelResult:
    """
    Run the connected research model.

    This function intentionally refuses inference until a real,
    documented research model is installed and validated.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("The model input must be a NumPy array.")

    if image.size == 0:
        raise ValueError("The model input cannot be empty.")

    raise ModelUnavailableError(
        "Clinical inference is disabled because no research model "
        "has been connected yet."
    )
