from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.explainability import (
    ExplainabilityError,
    ExplainabilityResult,
    generate_gradcam,
)
from src.imaging import ImageStudy, load_uploaded_study
from src.modeling import (
    ModelUnavailableError,
    ResearchModelResult,
    get_model_status,
    get_simulated_demo_result,
    run_research_model,
    _preprocess_image,
)
from src.safety import audit_generated_report
from src.ui import apply_theme, render_footer, render_header, render_metric_card


APP_VERSION = "0.6.0-explainability-export"


st.set_page_config(
    page_title="RadiantGuard AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


def render_sidebar() -> None:
    model_status = get_model_status()

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="brand-mark small">RG</div>
                <div>
                    <div class="sidebar-title">RadiantGuard AI</div>
                    <div class="sidebar-subtitle">Research Sandbox</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**Feature-gated model build**")
        st.caption(f"Version {APP_VERSION}")
        st.progress(78, text="MVP roadmap")

        st.markdown("### Current capabilities")
        st.markdown(
            """
            - Standard image viewing
            - DICOM pixel display
            - Technical image QA preview
            - AI report language audit
            - Protected model interface
            - Feature-gated real model runner
            - Source and suitability screening
            - Downloadable research reports
            - Grad-CAM model-influence maps
            - Adjustable blue X-ray overlays
            - Downloadable explainability graphics
            - Simulated prediction UI
            - Privacy-first metadata display
            """
        )

        st.markdown("### Model status")
        if model_status.available:
            st.success(f"Connected: {model_status.name}")
        else:
            st.info("Clinical inference disabled")
            st.caption(model_status.description)

        st.markdown("### Clinical boundary")
        st.warning(
            "This prototype does not diagnose disease and is not a medical device. "
            "All medical imaging must be reviewed by a qualified physician or radiologist."
        )
        st.caption("Do not upload files containing patient-identifying information.")


def render_empty_analysis() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">⌁</div>
            <h3>Your study workspace is ready</h3>
            <p>
                Upload a de-identified PNG, JPG, or DICOM file to begin
                a non-diagnostic technical review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _study_key(study: ImageStudy) -> str:
    """Create a stable session key for the currently displayed image."""

    image = np.ascontiguousarray(np.asarray(study.display_image))
    digest = hashlib.sha256()
    digest.update(study.file_name.encode("utf-8", errors="ignore"))
    digest.update(str(image.shape).encode("utf-8"))
    digest.update(str(image.dtype).encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _clean_label(label: str) -> str:
    """Convert model labels into a cleaner display format."""

    return label.replace("_", " ").strip()


def _create_analysis_id(study_key: str, created_at: datetime) -> str:
    """Create a reproducible-looking, non-patient analysis identifier."""

    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    short_hash = study_key[:8].upper()
    return f"RG-{timestamp}-{short_hash}"


def _assess_model_suitability(
    image_source: str,
    declared_image_type: str,
) -> tuple[str, str, tuple[str, ...]]:
    """Assess whether an upload fits this baseline's narrow research scope.

    This is a workflow check based only on user-declared source and image type.
    It is not an image interpretation or clinical-quality assessment.
    """

    notes: list[str] = []

    if declared_image_type != "Frontal chest X-ray (PA or AP)":
        return (
            "Outside intended baseline",
            "Do not run",
            (
                "This baseline was selected for frontal chest radiographs.",
                "The declared image type is outside its intended research input.",
                "No model run should be treated as meaningful for this image type.",
            ),
        )

    if image_source == "AI-generated or synthetic demo":
        notes.extend(
            (
                "Synthetic textures may produce arbitrary or misleading model scores.",
                "This source is useful only for software and interface testing.",
            )
        )
        return "Limited demo suitability", "High caution", tuple(notes)

    if image_source == "De-identified public research dataset":
        notes.extend(
            (
                "Dataset provenance and licensing should be documented separately.",
                "Research scores may still reflect dataset-specific bias.",
            )
        )
        return "Research-only eligible", "Caution", tuple(notes)

    if image_source == "De-identified educational sample":
        notes.extend(
            (
                "The sample's provenance and intended teaching use should be verified.",
                "Research scores remain unvalidated and non-diagnostic.",
            )
        )
        return "Research-only eligible", "Caution", tuple(notes)

    notes.extend(
        (
            "The source and acquisition context are not sufficiently documented.",
            "Unknown provenance increases out-of-distribution and privacy risk.",
        )
    )
    return "Uncertain suitability", "High caution", tuple(notes)


def _build_research_report(
    *,
    result: ResearchModelResult,
    study: ImageStudy,
    image_source: str,
    declared_image_type: str,
    suitability: str,
    caution_level: str,
    suitability_notes: tuple[str, ...],
    analysis_id: str,
    created_at: str,
) -> dict[str, object]:
    """Build a machine-readable, non-diagnostic research report."""

    return {
        "report_type": "RadiantGuard AI unvalidated research report",
        "analysis_id": analysis_id,
        "created_at_utc": created_at,
        "application_version": APP_VERSION,
        "clinical_status": (
            "Educational research output only. Not a diagnosis, medical device, "
            "or substitute for physician or radiologist review."
        ),
        "input": {
            "file_name": study.file_name,
            "file_format": study.file_format,
            "dimensions": {
                "width": study.width,
                "height": study.height,
            },
            "declared_image_source": image_source,
            "declared_image_type": declared_image_type,
            "displayed_modality_metadata": study.metadata.get(
                "Modality",
                "Not provided",
            ),
            "displayed_view_metadata": study.metadata.get(
                "View Position",
                "Not provided",
            ),
        },
        "suitability_screen": {
            "status": suitability,
            "caution_level": caution_level,
            "basis": (
                "User-declared source and image type only; "
                "not image interpretation."
            ),
            "notes": list(suitability_notes),
        },
        "model": {
            "name": result.model_name,
            "version": result.model_version,
            "mode": result.mode,
        },
        "research_scores": [
            {
                "label": _clean_label(prediction.label),
                "raw_score": round(float(prediction.confidence), 6),
                "display_score_percent": prediction.confidence_percent,
                "interpretation_boundary": prediction.explanation,
            }
            for prediction in result.predictions
        ],
        "preprocessing": list(result.preprocessing),
        "required_limitations": list(result.limitations),
    }



def _model_input_preview(image: np.ndarray) -> np.ndarray:
    """Return the exact 224×224 grayscale image supplied to the model."""

    tensor, _ = _preprocess_image(image)
    normalized = tensor[0, 0].detach().cpu().numpy().astype(np.float32)

    # TorchXRayVision normalization maps display pixels from 0–255
    # into approximately -1024–1024. Reverse that mapping for display.
    preview = ((normalized + 1024.0) / 2048.0) * 255.0
    return np.clip(preview, 0.0, 255.0).astype(np.uint8)


def _thresholded_influence(
    heatmap: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Suppress weaker values and rescale the remaining influence to 0–1."""

    clipped = np.clip(
        np.asarray(heatmap, dtype=np.float32),
        0.0,
        1.0,
    )

    threshold = min(max(float(threshold), 0.0), 0.95)
    denominator = max(1.0 - threshold, 1e-6)

    return np.clip(
        (clipped - threshold) / denominator,
        0.0,
        1.0,
    )


def _blue_glow_heatmap(
    heatmap: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert a normalized influence map into an icy blue RGB image."""

    influence = _thresholded_influence(heatmap, threshold)
    glow = np.power(influence, 0.72)

    red = 2.0 + (72.0 * glow)
    green = 9.0 + (220.0 * glow)
    blue = 28.0 + (227.0 * glow)

    rgb = np.stack((red, green, blue), axis=-1)
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _blue_glow_overlay(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    opacity: float,
    threshold: float,
) -> np.ndarray:
    """Blend the blue influence map onto the exact model-space input."""

    base = np.asarray(base_image, dtype=np.uint8)

    if base.ndim != 2:
        raise ValueError("The model-space preview must be grayscale.")

    base_rgb = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    heat_rgb = _blue_glow_heatmap(heatmap, threshold).astype(np.float32)
    influence = _thresholded_influence(heatmap, threshold)

    opacity = min(max(float(opacity), 0.0), 1.0)
    alpha = (opacity * np.power(influence, 0.68))[..., None]

    overlay = (base_rgb * (1.0 - alpha)) + (heat_rgb * alpha)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)



def _load_export_font(
    size: int,
    *,
    bold: bool = False,
) -> ImageFont.ImageFont:
    """Load a bundled Linux font when available, with a safe fallback."""

    candidates = (
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        (
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/dejavu/DejaVuSans.ttf"
        ),
    )

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue

    return ImageFont.load_default()


def _wrap_export_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap text to a pixel width for the generated PNG."""

    words = str(text).split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        left, top, right, bottom = draw.textbbox(
            (0, 0),
            candidate,
            font=font,
        )

        if right - left <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def _draw_wrapped_export_text(
    draw: ImageDraw.ImageDraw,
    *,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_spacing: int = 8,
) -> int:
    """Draw wrapped text and return the y-coordinate below the final line."""

    x, y = position
    lines = _wrap_export_text(
        draw,
        text,
        font,
        max_width,
    )

    for line in lines:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
        )
        bbox = draw.textbbox(
            (x, y),
            line,
            font=font,
        )
        y = bbox[3] + line_spacing

    return y


def _build_explainability_png(
    *,
    model_input: np.ndarray,
    heatmap_rgb: np.ndarray,
    overlay: np.ndarray,
    report: dict[str, object],
    explanation: ExplainabilityResult,
    created_at_utc: str,
    opacity_percent: int,
    threshold_percent: int,
) -> bytes:
    """Build a polished three-panel, explicitly non-diagnostic PNG export."""

    canvas_width = 1800
    canvas_height = 1160
    margin = 70
    gap = 34
    panel_width = (
        canvas_width
        - (2 * margin)
        - (2 * gap)
    ) // 3
    panel_top = 265
    panel_height = 650
    image_size = panel_width - 54

    background = (2, 8, 20)
    panel_background = (6, 24, 43)
    panel_inner = (1, 10, 24)
    cyan = (119, 233, 255)
    blue = (72, 169, 255)
    white = (238, 251, 255)
    muted = (158, 190, 205)
    warning_background = (33, 52, 46)
    warning_border = (120, 170, 145)
    warning_text = (233, 249, 239)

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        color=background,
    )
    draw = ImageDraw.Draw(canvas)

    title_font = _load_export_font(48, bold=True)
    subtitle_font = _load_export_font(25, bold=True)
    body_font = _load_export_font(21)
    small_font = _load_export_font(18)
    panel_title_font = _load_export_font(27, bold=True)
    badge_font = _load_export_font(20, bold=True)
    warning_font = _load_export_font(20, bold=True)

    # Header glow and framing.
    draw.rounded_rectangle(
        (35, 30, canvas_width - 35, 225),
        radius=34,
        fill=(4, 19, 36),
        outline=(34, 114, 170),
        width=2,
    )
    draw.ellipse(
        (
            canvas_width - 380,
            -110,
            canvas_width + 70,
            340,
        ),
        outline=(40, 142, 214),
        width=3,
    )
    draw.ellipse(
        (
            canvas_width - 320,
            -50,
            canvas_width + 10,
            280,
        ),
        outline=(27, 88, 139),
        width=2,
    )

    draw.rounded_rectangle(
        (70, 72, 152, 154),
        radius=20,
        fill=(93, 210, 246),
        outline=(217, 250, 255),
        width=2,
    )
    logo_font = _load_export_font(25, bold=True)
    draw.text(
        (91, 96),
        "RG",
        font=logo_font,
        fill=(2, 17, 31),
    )

    draw.text(
        (180, 60),
        "RadiantGuard AI",
        font=title_font,
        fill=white,
    )
    draw.text(
        (182, 124),
        "MODEL-INFLUENCE EXPLAINABILITY EXPORT",
        font=subtitle_font,
        fill=cyan,
    )

    label = _clean_label(explanation.label)
    score_text = f"{explanation.research_score_percent}% research score"
    draw.rounded_rectangle(
        (1195, 72, 1718, 151),
        radius=19,
        fill=(8, 48, 78),
        outline=(72, 169, 255),
        width=2,
    )
    draw.text(
        (1220, 88),
        label,
        font=badge_font,
        fill=white,
    )
    draw.text(
        (1220, 120),
        score_text,
        font=small_font,
        fill=cyan,
    )

    analysis_id = str(report.get("analysis_id", "Unknown"))
    model_info = report.get("model", {})
    if not isinstance(model_info, dict):
        model_info = {}

    model_name = str(model_info.get("name", "Unknown model"))
    model_version = str(model_info.get("version", "Unknown version"))
    input_info = report.get("input", {})
    if not isinstance(input_info, dict):
        input_info = {}

    source_label = str(
        input_info.get(
            "declared_image_source",
            "Source not documented",
        )
    )

    draw.text(
        (72, 188),
        f"Analysis ID: {analysis_id}",
        font=small_font,
        fill=muted,
    )
    draw.text(
        (655, 188),
        f"Generated: {created_at_utc}",
        font=small_font,
        fill=muted,
    )

    panel_data = (
        (
            "Model-space input",
            "Exact 224 × 224 image received by the model",
            model_input,
        ),
        (
            "Blue influence map",
            "Display-colored Grad-CAM values",
            heatmap_rgb,
        ),
        (
            "Adjustable overlay",
            (
                f"Glow {opacity_percent}% · "
                f"suppression {threshold_percent}%"
            ),
            overlay,
        ),
    )

    for index, (heading, caption, array) in enumerate(panel_data):
        left = margin + index * (panel_width + gap)
        right = left + panel_width
        bottom = panel_top + panel_height

        draw.rounded_rectangle(
            (left, panel_top, right, bottom),
            radius=28,
            fill=panel_background,
            outline=(32, 104, 153),
            width=2,
        )
        draw.text(
            (left + 27, panel_top + 25),
            heading,
            font=panel_title_font,
            fill=white,
        )

        image_top = panel_top + 83
        image_left = left + 27
        image_right = image_left + image_size
        image_bottom = image_top + image_size

        draw.rounded_rectangle(
            (
                image_left - 5,
                image_top - 5,
                image_right + 5,
                image_bottom + 5,
            ),
            radius=22,
            fill=panel_inner,
            outline=(43, 135, 190),
            width=2,
        )

        image_array = np.asarray(array)

        if image_array.ndim == 2:
            panel_image = Image.fromarray(
                image_array.astype(np.uint8),
                mode="L",
            ).convert("RGB")
        else:
            panel_image = Image.fromarray(
                image_array.astype(np.uint8),
                mode="RGB",
            )

        panel_image = panel_image.resize(
            (image_size, image_size),
            Image.Resampling.LANCZOS,
        )

        # Rounded image mask.
        mask = Image.new(
            "L",
            (image_size, image_size),
            color=0,
        )
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle(
            (0, 0, image_size - 1, image_size - 1),
            radius=18,
            fill=255,
        )
        canvas.paste(
            panel_image,
            (image_left, image_top),
            mask,
        )

        _draw_wrapped_export_text(
            draw,
            position=(
                left + 27,
                image_bottom + 25,
            ),
            text=caption,
            font=body_font,
            fill=muted,
            max_width=panel_width - 54,
            line_spacing=6,
        )

    warning_top = 950
    draw.rounded_rectangle(
        (
            margin,
            warning_top,
            canvas_width - margin,
            canvas_height - 55,
        ),
        radius=24,
        fill=warning_background,
        outline=warning_border,
        width=2,
    )

    warning = (
        "MODEL-INFLUENCE MAP — NOT DISEASE LOCALIZATION. "
        "Bright regions show what influenced this selected model output. "
        "They do not confirm disease, anatomy, causality, clinical relevance, "
        "or model correctness. Independent physician or radiologist review is required."
    )
    y_after_warning = _draw_wrapped_export_text(
        draw,
        position=(margin + 28, warning_top + 24),
        text=warning,
        font=warning_font,
        fill=warning_text,
        max_width=canvas_width - (2 * margin) - 56,
        line_spacing=9,
    )

    footer = (
        f"{model_name} · {model_version} · {explanation.method} · "
        f"target layer {explanation.target_layer} · Source: {source_label}"
    )
    _draw_wrapped_export_text(
        draw,
        position=(margin + 28, y_after_warning + 2),
        text=footer,
        font=small_font,
        fill=(177, 207, 219),
        max_width=canvas_width - (2 * margin) - 56,
        line_spacing=5,
    )

    output = BytesIO()
    canvas.save(
        output,
        format="PNG",
        optimize=True,
    )
    return output.getvalue()

def _add_explainability_to_report(
    report: dict[str, object],
    record: dict[str, object] | None,
) -> dict[str, object]:
    """Add explainability metadata without embedding image pixels."""

    enriched = json.loads(json.dumps(report))

    if record is None:
        enriched["explainability"] = {
            "status": "not generated",
            "interpretation_boundary": (
                "No model-influence map was generated for this analysis."
            ),
        }
        return enriched

    result = record["result"]

    if not isinstance(result, ExplainabilityResult):
        raise TypeError("Unexpected explainability result type.")

    heatmap = np.ascontiguousarray(result.heatmap, dtype=np.float32)
    heatmap_digest = hashlib.sha256(heatmap.tobytes()).hexdigest()

    enriched["explainability"] = {
        "status": "generated",
        "generated_at_utc": record["created_at_utc"],
        "selected_label": _clean_label(result.label),
        "selected_model_label": result.label,
        "selected_raw_research_score": round(
            float(result.research_score),
            6,
        ),
        "selected_display_score_percent": result.research_score_percent,
        "method": result.method,
        "target_layer": result.target_layer,
        "heatmap_shape": list(heatmap.shape),
        "heatmap_value_range": [
            round(float(heatmap.min()), 6),
            round(float(heatmap.max()), 6),
        ],
        "heatmap_sha256": heatmap_digest,
        "heatmap_pixels_embedded": False,
        "display_parameters": {
            "overlay_glow_strength_percent": record.get(
                "opacity_percent"
            ),
            "weaker_influence_suppression_percent": record.get(
                "threshold_percent"
            ),
            "color_style": "RadiantGuard icy blue glow",
        },
        "interpretation_boundary": (
            "The heatmap visualizes model influence in model space. "
            "It does not confirm disease location, anatomy, causality, "
            "clinical relevance, or model correctness."
        ),
        "limitations": list(result.limitations),
    }

    return enriched


def render_explainability_workspace(
    *,
    result: ResearchModelResult,
    study: ImageStudy,
    report: dict[str, object],
) -> dict[str, object]:
    """Render the Grad-CAM lab and return a report enriched with its metadata."""

    st.markdown("### Explainability lab")
    st.error(
        "MODEL-INFLUENCE MAP — NOT DISEASE LOCALIZATION. "
        "A bright region only shows what influenced the selected model output. "
        "It may reflect anatomy, artifacts, borders, markers, positioning, "
        "or synthetic texture rather than medically valid evidence."
    )

    available_labels = [
        prediction.label
        for prediction in result.predictions
    ]

    target_label = st.selectbox(
        "Research label to explore",
        options=available_labels,
        format_func=_clean_label,
        key=f"gradcam_label_{report['analysis_id']}",
        help=(
            "Choose one of the displayed model labels. The visualization "
            "does not determine whether that finding is truly present."
        ),
    )

    st.caption(
        "Grad-CAM is generated from the model's final DenseNet feature layer. "
        "It is a model-behavior visualization, not a radiology annotation."
    )

    record_key = f"{report['analysis_id']}::{target_label}"
    records = st.session_state.setdefault(
        "gradcam_records",
        {},
    )

    generate_clicked = st.button(
        "Generate blue model-influence map",
        key=f"gradcam_run_{report['analysis_id']}",
        help=(
            "Runs a forward and backward pass for the selected research label."
        ),
    )

    if generate_clicked:
        try:
            with st.spinner(
                "Generating the Grad-CAM influence map on CPU..."
            ):
                explanation = generate_gradcam(
                    image=np.asarray(study.display_image),
                    target_label=target_label,
                )

        except (ModelUnavailableError, ExplainabilityError, ValueError) as exc:
            st.error(
                "RadiantGuard could not generate the model-influence map."
            )
            st.caption(str(exc))

        except Exception as exc:
            st.error(
                "An unexpected explainability error occurred. "
                "No medical conclusion should be drawn from this failure."
            )
            st.caption(str(exc))

        else:
            records[record_key] = {
                "result": explanation,
                "created_at_utc": (
                    datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                ),
            }
            st.session_state["gradcam_records"] = records

    record = records.get(record_key)

    if record is None:
        st.info(
            "No model-influence map has been generated for this label."
        )
        return _add_explainability_to_report(report, None)

    explanation = record["result"]

    if not isinstance(explanation, ExplainabilityResult):
        st.error("The stored explainability result is invalid.")
        return _add_explainability_to_report(report, None)

    control_one, control_two = st.columns(2)

    with control_one:
        opacity_percent = st.slider(
            "Overlay glow strength",
            min_value=10,
            max_value=90,
            value=58,
            step=2,
            key=f"gradcam_opacity_{record_key}",
        )

    with control_two:
        threshold_percent = st.slider(
            "Suppress weaker influence",
            min_value=0,
            max_value=80,
            value=18,
            step=2,
            key=f"gradcam_threshold_{record_key}",
            help=(
                "Raises the display threshold only. "
                "It does not change the model or its research score."
            ),
        )

    record["opacity_percent"] = opacity_percent
    record["threshold_percent"] = threshold_percent
    records[record_key] = record
    st.session_state["gradcam_records"] = records

    model_input = _model_input_preview(
        np.asarray(study.display_image)
    )

    threshold = threshold_percent / 100.0
    heatmap_rgb = _blue_glow_heatmap(
        explanation.heatmap,
        threshold,
    )
    overlay = _blue_glow_overlay(
        model_input,
        explanation.heatmap,
        opacity_percent / 100.0,
        threshold,
    )

    metric_one, metric_two, metric_three = st.columns(3)

    with metric_one:
        render_metric_card(
            "Selected label",
            _clean_label(explanation.label),
        )

    with metric_two:
        render_metric_card(
            "Research score",
            f"{explanation.research_score_percent}%",
        )

    with metric_three:
        render_metric_card(
            "Method",
            explanation.method,
        )

    original_col, heatmap_col, overlay_col = st.columns(3)

    with original_col:
        st.markdown("#### Model-space input")
        st.image(
            model_input,
            caption="Exact 224 × 224 image received by the model",
            use_container_width=True,
        )

    with heatmap_col:
        st.markdown("#### Blue influence map")
        st.image(
            heatmap_rgb,
            caption="Display-colored Grad-CAM values",
            use_container_width=True,
        )

    with overlay_col:
        st.markdown("#### Adjustable overlay")
        st.image(
            overlay,
            caption="Influence map blended in model space",
            use_container_width=True,
        )

    st.warning(
        "The model-space images above reflect center cropping and resizing. "
        "They are not full-resolution radiology images, and the highlighted "
        "region must not be interpreted as a confirmed abnormality."
    )

    export_png = _build_explainability_png(
        model_input=model_input,
        heatmap_rgb=heatmap_rgb,
        overlay=overlay,
        report=report,
        explanation=explanation,
        created_at_utc=str(record["created_at_utc"]),
        opacity_percent=opacity_percent,
        threshold_percent=threshold_percent,
    )

    st.download_button(
        "Download three-panel explainability graphic (PNG)",
        data=export_png,
        file_name=(
            f"{report['analysis_id']}_"
            f"{explanation.label.replace('_', '-')}_gradcam.png"
        ),
        mime="image/png",
        key=f"gradcam_png_{record_key}",
        help=(
            "Downloads a labeled three-panel research graphic with the "
            "analysis ID, model version, visualization settings, and "
            "non-diagnostic warning printed directly on the image."
        ),
    )

    st.caption(
        "The PNG contains the 224 × 224 model-space input and derived "
        "visualizations. It must not be placed in a medical record or "
        "used for diagnosis or treatment."
    )

    with st.expander(
        "Explainability limitations",
        expanded=True,
    ):
        for limitation in explanation.limitations:
            st.markdown(f"- {limitation}")

    st.caption(
        f"Generated: {record['created_at_utc']} · "
        f"Target layer: {explanation.target_layer}"
    )

    return _add_explainability_to_report(report, record)


def render_research_result(
    result: ResearchModelResult,
    report: dict[str, object],
    study: ImageStudy,
) -> None:
    """Render research results, explainability, and a reproducible report."""

    st.error(
        "UNVALIDATED RESEARCH OUTPUT — NOT A DIAGNOSIS. "
        "These scores are not calibrated probabilities and must not guide "
        "treatment or replace review by a qualified physician or radiologist."
    )

    report_col, time_col = st.columns(2)
    report_col.caption(f"Analysis ID: {report['analysis_id']}")
    time_col.caption(f"Created: {report['created_at_utc']}")

    st.markdown(f"#### {result.model_name}")
    st.caption(f"Version {result.model_version} · {result.mode}")

    st.markdown("##### Highest research scores")

    for prediction in result.predictions:
        label_col, score_col = st.columns([3, 1])
        clean_label = _clean_label(prediction.label)

        with label_col:
            st.markdown(f"**{clean_label}**")
            st.caption(prediction.explanation)

        with score_col:
            st.metric(
                "Research score",
                f"{prediction.confidence_percent}%",
                help=(
                    "A raw research-model score for interface comparison. "
                    "It is not a calibrated probability or diagnostic confidence."
                ),
            )

        st.progress(prediction.confidence_percent)

    enriched_report = render_explainability_workspace(
        result=result,
        study=study,
        report=report,
    )

    with st.expander("Required model limitations", expanded=True):
        for limitation in result.limitations:
            st.markdown(f"- {limitation}")

    if result.preprocessing:
        with st.expander("View preprocessing record"):
            for step in result.preprocessing:
                st.markdown(f"- {step}")

    report_json = json.dumps(
        enriched_report,
        indent=2,
        ensure_ascii=False,
    )

    st.download_button(
        "Download research report (JSON)",
        data=report_json,
        file_name=f"{report['analysis_id']}.json",
        mime="application/json",
        help=(
            "Downloads scores, model version, preprocessing, source declaration, "
            "suitability screening, explainability metadata, and required limitations. "
            "No image or heatmap pixels are included."
        ),
    )

    st.caption(
        "The downloaded report contains no image or heatmap pixels and should "
        "not be placed in a medical record or used for clinical decisions."
    )


def render_simulated_workspace() -> None:
    """Render the optional hard-coded interface demonstration."""

    show_demo = st.checkbox(
        "Show a simulated prediction-interface example",
        key="show_simulated_model_demo",
        help=(
            "This displays hard-coded sample values to test the interface. "
            "It does not analyze the uploaded image."
        ),
    )

    if not show_demo:
        st.caption(
            "The simulated preview is optional and remains off by default. "
            "It does not use the uploaded image."
        )
        return

    demo = get_simulated_demo_result()

    st.error(
        "SIMULATED UI DEMO — NO IMAGE ANALYSIS OCCURRED. "
        "The values below are hard-coded interface examples and are not medical findings."
    )

    st.markdown(f"#### {demo.model_name}")
    st.caption(f"Version {demo.model_version} · {demo.mode}")

    for prediction in demo.predictions:
        label_col, score_col = st.columns([3, 1])

        with label_col:
            st.markdown(f"**Example: {prediction.label}**")
            st.caption(prediction.explanation)

        with score_col:
            st.metric(
                "Simulated score",
                f"{prediction.confidence_percent}%",
            )

        st.progress(prediction.confidence_percent)

    with st.expander("View simulated-output limitations", expanded=True):
        for limitation in demo.limitations:
            st.markdown(f"- {limitation}")


def render_model_workspace(study: ImageStudy) -> None:
    model_status = get_model_status()

    st.markdown("### Research model workspace")

    status_col, version_col, mode_col = st.columns(3)

    with status_col:
        render_metric_card(
            "Inference",
            "Enabled" if model_status.available else "Disabled",
        )

    with version_col:
        render_metric_card("Model version", model_status.version)

    with mode_col:
        render_metric_card("Mode", model_status.mode)

    if not model_status.available:
        st.info(
            "The real model runner is protected by an off-by-default feature "
            "switch. RadiantGuard will not generate image-based scores in the "
            "current live configuration."
        )
        st.caption(model_status.description)
        render_simulated_workspace()
        return

    st.warning(
        "A public pretrained research baseline is available in this environment. "
        "Its output is unvalidated, non-diagnostic, and may be wrong."
    )

    st.markdown("#### Source and suitability screen")

    source_col, type_col = st.columns(2)

    with source_col:
        image_source = st.selectbox(
            "Image source",
            options=[
                "AI-generated or synthetic demo",
                "De-identified public research dataset",
                "De-identified educational sample",
                "Other or unknown source",
            ],
            key="research_image_source",
            help=(
                "Choose the best description of the image's provenance. "
                "Never upload identifiable patient data."
            ),
        )

    with type_col:
        declared_image_type = st.selectbox(
            "Declared image type",
            options=[
                "Frontal chest X-ray (PA or AP)",
                "Lateral chest X-ray",
                "CT, MRI, ultrasound, or other modality",
                "Unknown",
            ],
            key="research_declared_image_type",
            help=(
                "This is a user declaration for model-scope screening. "
                "RadiantGuard is not verifying anatomy or modality here."
            ),
        )

    suitability, caution_level, suitability_notes = _assess_model_suitability(
        image_source,
        declared_image_type,
    )

    suitability_col, caution_col = st.columns(2)
    with suitability_col:
        render_metric_card("Suitability", suitability)
    with caution_col:
        render_metric_card("Caution level", caution_level)

    for note in suitability_notes:
        st.info(note)

    if image_source == "AI-generated or synthetic demo":
        st.error(
            "SYNTHETIC DEMO IMAGE: Scores may reflect artificial textures and "
            "must not be interpreted as evidence of any medical finding."
        )

    outside_scope = declared_image_type != "Frontal chest X-ray (PA or AP)"

    if outside_scope:
        st.error(
            "The declared image type is outside this model's intended research scope. "
            "Inference is disabled for this upload."
        )

    acknowledged = st.checkbox(
        "I understand these are unvalidated research scores, not a diagnosis, "
        "and the image and output require independent physician or radiologist review.",
        key="research_model_acknowledgement",
    )

    current_key = _study_key(study)
    result_context_key = (
        current_key,
        image_source,
        declared_image_type,
    )

    if st.session_state.get("research_result_context") != result_context_key:
        st.session_state.pop("research_model_result", None)
        st.session_state.pop("research_model_report", None)
        st.session_state.pop("research_result_context", None)
        st.session_state.pop("gradcam_records", None)

    run_clicked = st.button(
        "Run unvalidated research baseline",
        type="primary",
        disabled=(not acknowledged or outside_scope),
        help=(
            "Runs the public research model on the displayed image. "
            "This does not provide medical advice or a diagnosis."
        ),
    )

    if run_clicked:
        image = np.asarray(study.display_image)

        try:
            with st.spinner(
                "Running the CPU research baseline. "
                "The first run may take longer while model weights load..."
            ):
                result = run_research_model(image)

        except ModelUnavailableError as exc:
            st.error("The protected research model is not available.")
            st.caption(str(exc))

        except Exception as exc:
            st.error(
                "RadiantGuard could not complete the research-model run. "
                "No medical conclusion should be drawn from this error."
            )
            st.caption(str(exc))

        else:
            created_at = datetime.now(timezone.utc)
            created_at_text = created_at.isoformat().replace("+00:00", "Z")
            analysis_id = _create_analysis_id(current_key, created_at)

            report = _build_research_report(
                result=result,
                study=study,
                image_source=image_source,
                declared_image_type=declared_image_type,
                suitability=suitability,
                caution_level=caution_level,
                suitability_notes=suitability_notes,
                analysis_id=analysis_id,
                created_at=created_at_text,
            )

            st.session_state["research_model_result"] = result
            st.session_state["research_model_report"] = report
            st.session_state["research_result_context"] = result_context_key

    stored_result = st.session_state.get("research_model_result")
    stored_report = st.session_state.get("research_model_report")
    stored_context = st.session_state.get("research_result_context")

    if (
        stored_result is not None
        and stored_report is not None
        and stored_context == result_context_key
    ):
        render_research_result(stored_result, stored_report, study)
    else:
        st.caption(
            "No research inference has been run for this uploaded image "
            "and declared source context."
        )


def render_study(study: ImageStudy) -> None:
    left, right = st.columns([1.45, 1], gap="large")

    with left:
        st.markdown("### Image viewer")
        st.image(
            study.display_image,
            caption=f"{study.file_name} · {study.width} × {study.height}",
            use_container_width=True,
        )
        st.caption(
            "Displayed pixels are for educational review only. "
            "Any research-model output is unvalidated and non-diagnostic."
        )

    with right:
        st.markdown("### Study summary")
        c1, c2 = st.columns(2)

        with c1:
            render_metric_card("Format", study.file_format)
            render_metric_card("Dimensions", f"{study.width} × {study.height}")

        with c2:
            render_metric_card(
                "Modality",
                study.metadata.get("Modality", "Not provided"),
            )
            render_metric_card(
                "View",
                study.metadata.get("View Position", "Not provided"),
            )

        st.markdown("#### Technical quality preview")
        q1, q2, q3 = st.columns(3)

        with q1:
            render_metric_card(
                "Brightness",
                f"{study.quality.brightness_score:.0f}/100",
            )

        with q2:
            render_metric_card(
                "Contrast",
                f"{study.quality.contrast_score:.0f}/100",
            )

        with q3:
            render_metric_card(
                "Detail",
                f"{study.quality.detail_score:.0f}/100",
            )

        if study.quality.messages:
            for message in study.quality.messages:
                st.info(message)
        else:
            st.success(
                "No obvious technical display warnings were detected "
                "by the basic preview checks."
            )

        st.caption(
            "These measurements describe pixel characteristics only. "
            "They do not establish clinical image adequacy."
        )

    st.markdown("### Pixel distribution")
    hist_values, _ = np.histogram(
        np.asarray(study.display_image),
        bins=64,
        range=(0, 255),
    )
    st.line_chart(hist_values, height=170)

    if study.metadata:
        with st.expander("View non-identifying image metadata"):
            meta_cols = st.columns(2)
            for index, (key, value) in enumerate(study.metadata.items()):
                meta_cols[index % 2].text(f"{key}: {value}")

            st.caption(
                "RadiantGuard intentionally excludes common patient-name "
                "and patient-ID fields from this display."
            )

    render_model_workspace(study)


def render_analyze_tab() -> None:
    intro, checklist = st.columns([1.4, 1], gap="large")

    with intro:
        st.markdown("## Medical image workspace")
        st.write(
            "Review how a medical image is represented, inspect technical "
            "pixel characteristics, and prepare it for future explainability research."
        )

    with checklist:
        st.markdown(
            """
            <div class="safety-strip">
                <strong>Before uploading</strong>
                <span>Use only de-identified educational or research images.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    consent = st.checkbox(
        "I understand this is a non-diagnostic prototype and I will not "
        "upload identifiable patient information.",
        key="upload_consent",
    )

    uploaded_file = st.file_uploader(
        "Upload a de-identified medical image",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
        disabled=not consent,
        help="Supported in this build: PNG, JPG, JPEG, DCM, and DICOM.",
    )

    if uploaded_file is None:
        render_empty_analysis()
        return

    try:
        study = load_uploaded_study(
            uploaded_file.name,
            uploaded_file.getvalue(),
        )
    except Exception as exc:
        st.error("RadiantGuard could not safely display this file.")
        st.caption(str(exc))
        return

    render_study(study)


def render_safety_lab_tab() -> None:
    st.markdown("## AI report safety lab")
    st.write(
        "Paste an AI-generated imaging statement to review its wording for "
        "common safety concerns. This first rules-based auditor evaluates "
        "language—not the medical accuracy of the report."
    )

    report_text = st.text_area(
        "AI-generated report or explanation",
        height=190,
        placeholder=(
            "Example: This image definitely confirms pneumonia and no "
            "radiologist review is necessary."
        ),
    )

    if not report_text.strip():
        st.markdown(
            """
            <div class="empty-state compact">
                <h3>Safety review awaiting text</h3>
                <p>
                    The auditor will flag overconfidence, missing human-review
                    language, and unsupported replacement claims.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    audit = audit_generated_report(report_text)
    score_col, status_col = st.columns([1, 2])

    score_col.metric("Language safety score", f"{audit.score}/100")
    status_col.markdown(f"### {audit.rating}")
    status_col.caption(
        "A language-quality indicator only; not a clinical accuracy score."
    )

    if audit.flags:
        st.markdown("### Review flags")
        for flag in audit.flags:
            st.warning(flag)
    else:
        st.success("No high-priority wording flags were found by the current rules.")

    st.markdown("### Positive signals")
    if audit.positive_signals:
        for signal in audit.positive_signals:
            st.success(signal)
    else:
        st.info(
            "Consider adding explicit uncertainty and qualified physician "
            "or radiologist review language."
        )

    st.markdown("### Safer prototype wording")
    st.code(audit.suggested_footer, language=None)


def render_about_tab() -> None:
    st.markdown("## Building transparent imaging AI")
    st.write(
        "RadiantGuard AI is an educational research sandbox focused on "
        "explainability, uncertainty, human oversight, and the evaluation "
        "of AI-generated medical-imaging outputs."
    )

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("### 01 · Foundation")
        st.write(
            "Secure upload flow, DICOM-aware viewing, technical QA, "
            "and safety documentation."
        )

    with c2:
        st.markdown("### 02 · Research model")
        st.write(
            "Protected model interface, reproducible preprocessing, "
            "versioned outputs, and model cards."
        )

    with c3:
        st.markdown("### 03 · Explainability")
        st.write(
            "Heatmaps, uncertainty, report comparison, model disagreement, "
            "and AI trust evaluation."
        )

    st.markdown("### Non-negotiable product principles")
    st.markdown(
        """
        - **Human oversight:** model output never replaces a qualified clinician.
        - **Transparent limitations:** every result identifies what the system cannot establish.
        - **Privacy by design:** development uses de-identified public or educational data.
        - **No silent certainty:** uncertainty and model disagreement remain visible.
        - **Reproducibility:** model versions, preprocessing, and thresholds are documented.
        """
    )

    st.markdown("### Intended users")
    st.write(
        "Students, healthcare AI evaluators, developers, researchers, and "
        "clinicians exploring responsible medical-imaging AI—not patients "
        "seeking a diagnosis."
    )


render_sidebar()
render_header()

analyze_tab, safety_tab, about_tab = st.tabs(
    [
        "Image Workspace",
        "AI Safety Lab",
        "About & Roadmap",
    ]
)

with analyze_tab:
    render_analyze_tab()

with safety_tab:
    render_safety_lab_tab()

with about_tab:
    render_about_tab()

render_footer(APP_VERSION)
