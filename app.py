from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.imaging import ImageStudy, load_uploaded_study
from src.safety import audit_generated_report
from src.ui import apply_theme, render_footer, render_header, render_metric_card

APP_VERSION = "0.1.0-foundation"

st.set_page_config(
    page_title="RadiantGuard AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


def render_sidebar() -> None:
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
        st.markdown("**Foundation build**")
        st.caption(f"Version {APP_VERSION}")
        st.progress(18, text="MVP roadmap")

        st.markdown("### Current capabilities")
        st.markdown(
            """
            - Standard image viewing
            - DICOM pixel display
            - Technical image QA preview
            - AI report language audit
            - Privacy-first metadata display
            """
        )

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
            <p>Upload a de-identified PNG, JPG, or DICOM file to begin a non-diagnostic technical review.</p>
        </div>
        """,
        unsafe_allow_html=True,
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
            "Displayed pixels are for educational review only. No diagnostic model is connected in this build."
        )

    with right:
        st.markdown("### Study summary")
        c1, c2 = st.columns(2)
        with c1:
            render_metric_card("Format", study.file_format)
            render_metric_card("Dimensions", f"{study.width} × {study.height}")
        with c2:
            render_metric_card("Modality", study.metadata.get("Modality", "Not provided"))
            render_metric_card("View", study.metadata.get("View Position", "Not provided"))

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
            st.success("No obvious technical display warnings were detected by the basic preview checks.")

        st.caption(
            "These measurements describe pixel characteristics only. They do not establish clinical image adequacy."
        )

    st.markdown("### Pixel distribution")
    hist_values, _ = np.histogram(np.asarray(study.display_image), bins=64, range=(0, 255))
    st.line_chart(hist_values, height=170)

    if study.metadata:
        with st.expander("View non-identifying image metadata"):
            meta_cols = st.columns(2)
            for index, (key, value) in enumerate(study.metadata.items()):
                meta_cols[index % 2].text(f"{key}: {value}")
            st.caption(
                "RadiantGuard intentionally excludes common patient-name and patient-ID fields from this display."
            )

    st.markdown("### Research model workspace")
    st.markdown(
        """
        <div class="model-placeholder">
            <div>
                <span class="status-dot"></span>
                <strong>Clinical inference is not enabled</strong>
            </div>
            <p>
                The next model phase will add a validated public chest X-ray research model, calibrated confidence,
                and explainability overlays. Until then, this workspace will never generate a medical finding from an upload.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analyze_tab() -> None:
    intro, checklist = st.columns([1.4, 1], gap="large")
    with intro:
        st.markdown("## Medical image workspace")
        st.write(
            "Review how a medical image is represented, inspect technical pixel characteristics, "
            "and prepare it for future explainability research."
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
        "I understand this is a non-diagnostic prototype and I will not upload identifiable patient information.",
        key="upload_consent",
    )
    uploaded_file = st.file_uploader(
        "Upload a de-identified medical image",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
        disabled=not consent,
        help="Supported in this foundation build: PNG, JPG, JPEG, DCM, and DICOM.",
    )

    if uploaded_file is None:
        render_empty_analysis()
        return

    try:
        study = load_uploaded_study(uploaded_file.name, uploaded_file.getvalue())
    except Exception as exc:  # Friendly boundary around malformed or unsupported files.
        st.error("RadiantGuard could not safely display this file.")
        st.caption(str(exc))
        return

    render_study(study)


def render_safety_lab_tab() -> None:
    st.markdown("## AI report safety lab")
    st.write(
        "Paste an AI-generated imaging statement to review its wording for common safety concerns. "
        "This first rules-based auditor evaluates language—not the medical accuracy of the report."
    )

    report_text = st.text_area(
        "AI-generated report or explanation",
        height=190,
        placeholder=(
            "Example: This image definitely confirms pneumonia and no radiologist review is necessary."
        ),
    )

    if not report_text.strip():
        st.markdown(
            """
            <div class="empty-state compact">
                <h3>Safety review awaiting text</h3>
                <p>The auditor will flag overconfidence, missing human-review language, and unsupported replacement claims.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    audit = audit_generated_report(report_text)
    score_col, status_col = st.columns([1, 2])
    score_col.metric("Language safety score", f"{audit.score}/100")
    status_col.markdown(f"### {audit.rating}")
    status_col.caption("A language-quality indicator only; not a clinical accuracy score.")

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
        st.info("Consider adding explicit uncertainty and qualified physician or radiologist review language.")

    st.markdown("### Safer prototype wording")
    st.code(audit.suggested_footer, language=None)


def render_about_tab() -> None:
    st.markdown("## Building transparent imaging AI")
    st.write(
        "RadiantGuard AI is an educational research sandbox focused on explainability, uncertainty, "
        "human oversight, and the evaluation of AI-generated medical-imaging outputs."
    )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("### 01 · Foundation")
        st.write("Secure upload flow, DICOM-aware viewing, technical QA, and safety documentation.")
    with c2:
        st.markdown("### 02 · Research model")
        st.write("Public chest X-ray model, reproducible preprocessing, calibrated output, and model cards.")
    with c3:
        st.markdown("### 03 · Explainability")
        st.write("Heatmaps, uncertainty, report comparison, model disagreement, and AI trust evaluation.")

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
        "Students, healthcare AI evaluators, developers, researchers, and clinicians exploring responsible "
        "medical-imaging AI—not patients seeking a diagnosis."
    )


render_sidebar()
render_header()

analyze_tab, safety_tab, about_tab = st.tabs(
    ["Image Workspace", "AI Safety Lab", "About & Roadmap"]
)
with analyze_tab:
    render_analyze_tab()
with safety_tab:
    render_safety_lab_tab()
with about_tab:
    render_about_tab()

render_footer(APP_VERSION)
