from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import streamlit as st

from src.imaging import ImageStudy, load_uploaded_study
from src.modeling import (
    ModelUnavailableError,
    ResearchModelResult,
    get_model_status,
    get_simulated_demo_result,
    run_research_model,
)
from src.safety import audit_generated_report
from src.ui import apply_theme, render_footer, render_header, render_metric_card


APP_VERSION = "0.4.0-research-reporting"


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
        st.progress(55, text="MVP roadmap")

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


def render_research_result(
    result: ResearchModelResult,
    report: dict[str, object],
) -> None:
    """Render a completed research-model result with explicit limitations."""

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

    with st.expander("Required limitations", expanded=True):
        for limitation in result.limitations:
            st.markdown(f"- {limitation}")

    if result.preprocessing:
        with st.expander("View preprocessing record"):
            for step in result.preprocessing:
                st.markdown(f"- {step}")

    report_json = json.dumps(
        report,
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
            "suitability screening, and required limitations. No image pixels are included."
        ),
    )

    st.caption(
        "The downloaded report contains no image pixels and should not be "
        "placed in a medical record or used for clinical decisions."
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
        render_research_result(stored_result, stored_report)
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
