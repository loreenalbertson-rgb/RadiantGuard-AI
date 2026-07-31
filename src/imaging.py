from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class QualityPreview:
    brightness_score: float
    contrast_score: float
    detail_score: float
    messages: tuple[str, ...]


@dataclass(frozen=True)
class ImageStudy:
    file_name: str
    file_format: str
    display_image: Image.Image
    width: int
    height: int
    metadata: dict[str, str]
    quality: QualityPreview


def _normalise_to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim > 2:
        arr = arr[0]

    low, high = np.percentile(arr, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(arr.min()), float(arr.max())
    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low)
    return np.round(arr * 255.0).astype(np.uint8)


def _technical_quality_preview(image: Image.Image) -> QualityPreview:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    mean = float(gray.mean())
    std = float(gray.std())

    grad_y, grad_x = np.gradient(gray)
    gradient_energy = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))

    brightness_score = max(0.0, 100.0 - abs(mean - 0.5) * 190.0)
    contrast_score = min(100.0, std / 0.25 * 100.0)
    detail_score = min(100.0, gradient_energy / 0.09 * 100.0)

    messages: list[str] = []
    if mean < 0.18:
        messages.append("The displayed image is unusually dark after normalization.")
    elif mean > 0.82:
        messages.append("The displayed image is unusually bright after normalization.")
    if std < 0.08:
        messages.append("The displayed image has low pixel contrast.")
    if gradient_energy < 0.015:
        messages.append("The displayed image contains relatively little edge detail or may appear blurred.")

    return QualityPreview(
        brightness_score=brightness_score,
        contrast_score=contrast_score,
        detail_score=detail_score,
        messages=tuple(messages),
    )


def _safe_text(value: Any, default: str = "Not provided") -> str:
    if value is None:
        return default
    rendered = str(value).strip()
    return rendered or default


def _load_dicom(file_name: str, data: bytes) -> ImageStudy:
    try:
        import pydicom
        from pydicom.pixels import apply_voi_lut
    except ImportError as exc:
        raise RuntimeError("DICOM support requires the pydicom package.") from exc

    dataset = pydicom.dcmread(io.BytesIO(data), force=True)
    if "PixelData" not in dataset:
        raise ValueError("This DICOM file does not contain displayable pixel data.")

    try:
        pixels = dataset.pixel_array
    except Exception as exc:
        raise ValueError(
            "The DICOM pixel data could not be decoded. Some compressed files require an additional decoder."
        ) from exc

    try:
        pixels = apply_voi_lut(pixels, dataset)
    except Exception:
        pass

    if pixels.ndim >= 3:
        pixels = pixels[0]

    normalized = _normalise_to_uint8(pixels)
    if _safe_text(getattr(dataset, "PhotometricInterpretation", None)) == "MONOCHROME1":
        normalized = 255 - normalized

    image = Image.fromarray(normalized, mode="L")
    metadata = {
        "Modality": _safe_text(getattr(dataset, "Modality", None)),
        "Body Part": _safe_text(getattr(dataset, "BodyPartExamined", None)),
        "View Position": _safe_text(getattr(dataset, "ViewPosition", None)),
        "Photometric Interpretation": _safe_text(
            getattr(dataset, "PhotometricInterpretation", None)
        ),
        "Rows": _safe_text(getattr(dataset, "Rows", image.height)),
        "Columns": _safe_text(getattr(dataset, "Columns", image.width)),
        "Bits Stored": _safe_text(getattr(dataset, "BitsStored", None)),
    }
    metadata = {key: value for key, value in metadata.items() if value != "Not provided"}

    return ImageStudy(
        file_name=file_name,
        file_format="DICOM",
        display_image=image,
        width=image.width,
        height=image.height,
        metadata=metadata,
        quality=_technical_quality_preview(image),
    )


def _load_standard_image(file_name: str, data: bytes) -> ImageStudy:
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image.load()
    except Exception as exc:
        raise ValueError("The uploaded file is not a readable PNG or JPEG image.") from exc

    original_format = (image.format or file_name.rsplit(".", 1)[-1]).upper()
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")

    return ImageStudy(
        file_name=file_name,
        file_format=original_format,
        display_image=image,
        width=image.width,
        height=image.height,
        metadata={"Source": "Standard image upload"},
        quality=_technical_quality_preview(image),
    )


def load_uploaded_study(file_name: str, data: bytes) -> ImageStudy:
    if not data:
        raise ValueError("The uploaded file is empty.")

    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    if suffix in {"dcm", "dicom"} or data[128:132] == b"DICM":
        return _load_dicom(file_name, data)
    if suffix in {"png", "jpg", "jpeg"}:
        return _load_standard_image(file_name, data)
    raise ValueError("Unsupported file type. Upload PNG, JPG, JPEG, DCM, or DICOM.")
