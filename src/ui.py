from __future__ import annotations

from html import escape

import streamlit as st


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --rg-void: #020711;
    --rg-navy: #061325;
    --rg-navy-soft: #0a2038;
    --rg-panel: rgba(8, 27, 48, 0.82);
    --rg-panel-strong: rgba(6, 20, 38, 0.96);
    --rg-blue: #48a9ff;
    --rg-cyan: #77e9ff;
    --rg-ice: #d9f7ff;
    --rg-glow: rgba(72, 169, 255, 0.30);
    --rg-glow-strong: rgba(119, 233, 255, 0.42);
    --rg-line: rgba(126, 211, 255, 0.18);
    --rg-text: #edfaff;
    --rg-muted: #9db9c9;
    --rg-warning: #ffd38a;
}

html,
body,
[class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1,
h2,
h3,
h4 {
    font-family: 'Manrope', sans-serif;
    letter-spacing: -0.025em;
    color: var(--rg-text);
}

p,
label,
[data-testid="stMarkdownContainer"] {
    color: var(--rg-text);
}

.stApp {
    background:
        radial-gradient(circle at 84% 2%, rgba(57, 151, 255, 0.18), transparent 28rem),
        radial-gradient(circle at 10% 32%, rgba(34, 105, 205, 0.12), transparent 30rem),
        linear-gradient(180deg, #020711 0%, #06101f 42%, #030914 100%);
    color: var(--rg-text);
}

.stApp:before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.16;
    background-image:
        linear-gradient(rgba(126, 211, 255, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(126, 211, 255, 0.025) 1px, transparent 1px);
    background-size: 38px 38px;
    mask-image: linear-gradient(to bottom, black, transparent 92%);
}

[data-testid="stHeader"] {
    background: rgba(2, 7, 17, 0.70);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(126, 211, 255, 0.08);
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 20% 8%, rgba(72, 169, 255, 0.17), transparent 17rem),
        linear-gradient(180deg, #030b17 0%, #06172a 100%);
    color: white;
    border-right: 1px solid rgba(119, 233, 255, 0.13);
    box-shadow: 12px 0 48px rgba(0, 0, 0, 0.22);
}

[data-testid="stSidebar"] * {
    color: #e8f8ff;
}

[data-testid="stSidebar"] .stAlert * {
    color: #dcecf5;
}

.block-container {
    max-width: 1320px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 2.4rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(119, 233, 255, 0.18);
    border-radius: 26px;
    color: white;
    background:
        radial-gradient(circle at 88% 20%, rgba(71, 174, 255, 0.25), transparent 18rem),
        linear-gradient(130deg, #020812 0%, #071b33 58%, #073055 100%);
    box-shadow:
        0 28px 80px rgba(0, 0, 0, 0.38),
        0 0 55px rgba(59, 159, 255, 0.16),
        inset 0 0 40px rgba(94, 195, 255, 0.05);
}

.hero:before {
    content: "";
    position: absolute;
    inset: 0;
    opacity: 0.28;
    background:
        linear-gradient(
            90deg,
            transparent 0%,
            rgba(119, 233, 255, 0.06) 46%,
            rgba(119, 233, 255, 0.18) 50%,
            rgba(119, 233, 255, 0.06) 54%,
            transparent 100%
        );
    transform: translateX(-100%);
    animation: rg-scan 7s ease-in-out infinite;
}

.hero:after {
    content: "";
    position: absolute;
    width: 390px;
    height: 390px;
    right: -120px;
    top: -190px;
    border-radius: 50%;
    border: 1px solid rgba(119, 233, 255, 0.35);
    box-shadow:
        0 0 90px rgba(72, 169, 255, 0.25),
        inset 0 0 80px rgba(119, 233, 255, 0.09);
}

@keyframes rg-scan {
    0%, 18% { transform: translateX(-120%); opacity: 0; }
    40% { opacity: 0.30; }
    67%, 100% { transform: translateX(120%); opacity: 0; }
}

.hero-grid {
    display: flex;
    align-items: center;
    gap: 1.35rem;
    position: relative;
    z-index: 2;
}

.brand-mark {
    width: 76px;
    height: 76px;
    flex: 0 0 76px;
    display: grid;
    place-items: center;
    border: 1px solid rgba(217, 247, 255, 0.50);
    border-radius: 21px;
    font-family: 'Manrope', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #02101d;
    background:
        radial-gradient(circle at 30% 25%, #effdff 0%, #8beaff 32%, #3d9dff 100%);
    box-shadow:
        0 0 18px rgba(119, 233, 255, 0.60),
        0 0 42px rgba(72, 169, 255, 0.32),
        inset 0 0 18px rgba(255, 255, 255, 0.68);
}

.brand-mark.small {
    width: 43px;
    height: 43px;
    flex-basis: 43px;
    border-radius: 12px;
    font-size: 0.86rem;
}

.hero h1 {
    font-size: clamp(2rem, 4vw, 3.25rem);
    margin: 0 0 0.25rem;
    color: white;
    text-shadow: 0 0 25px rgba(119, 233, 255, 0.18);
}

.hero p {
    max-width: 850px;
    margin: 0.3rem 0 0;
    color: #c9e8f5;
    font-size: 1.03rem;
    line-height: 1.65;
}

.eyebrow {
    color: var(--rg-cyan);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    text-shadow: 0 0 12px rgba(119, 233, 255, 0.38);
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.35rem 0 1rem;
}

.sidebar-title {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}

.sidebar-subtitle {
    font-size: 0.76rem;
    color: #9fcfe2 !important;
}

.safety-strip {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 1rem 1.1rem;
    border: 1px solid rgba(72, 169, 255, 0.25);
    border-radius: 15px;
    background:
        linear-gradient(135deg, rgba(72, 169, 255, 0.10), rgba(11, 45, 77, 0.32));
    box-shadow: inset 0 0 28px rgba(72, 169, 255, 0.05);
}

.safety-strip span {
    color: var(--rg-muted);
    font-size: 0.9rem;
}

.empty-state {
    margin-top: 1rem;
    padding: 4.4rem 2rem;
    border: 1px dashed rgba(119, 233, 255, 0.28);
    border-radius: 22px;
    text-align: center;
    background:
        radial-gradient(circle at 50% 28%, rgba(72, 169, 255, 0.10), transparent 18rem),
        rgba(5, 20, 36, 0.72);
    box-shadow:
        inset 0 0 42px rgba(72, 169, 255, 0.04),
        0 20px 55px rgba(0, 0, 0, 0.22);
}

.empty-state.compact {
    padding: 2.2rem 1.5rem;
}

.empty-state h3 {
    margin: 0.25rem 0;
}

.empty-state p {
    margin: 0.35rem auto 0;
    max-width: 640px;
    color: var(--rg-muted);
}

.empty-icon {
    margin: auto;
    font-size: 2rem;
    color: var(--rg-cyan);
    text-shadow: 0 0 18px rgba(119, 233, 255, 0.62);
}

.metric-card {
    min-height: 96px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.75rem;
    border: 1px solid rgba(126, 211, 255, 0.17);
    border-radius: 17px;
    background:
        linear-gradient(145deg, rgba(12, 38, 65, 0.92), rgba(5, 20, 37, 0.94));
    box-shadow:
        0 10px 28px rgba(0, 0, 0, 0.22),
        0 0 24px rgba(72, 169, 255, 0.05),
        inset 0 0 25px rgba(119, 233, 255, 0.025);
}

.metric-label {
    color: #8bb5ca;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.metric-value {
    margin-top: 0.35rem;
    color: #effcff;
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    text-shadow: 0 0 16px rgba(119, 233, 255, 0.12);
}

.model-placeholder {
    padding: 1.15rem 1.25rem;
    border: 1px solid rgba(119, 233, 255, 0.18);
    border-radius: 17px;
    background:
        linear-gradient(135deg, rgba(7, 26, 48, 0.96), rgba(9, 47, 76, 0.68));
    box-shadow:
        0 14px 40px rgba(0, 0, 0, 0.20),
        inset 0 0 32px rgba(72, 169, 255, 0.04);
}

.model-placeholder p {
    margin: 0.65rem 0 0;
    color: var(--rg-muted);
}

.status-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    margin-right: 0.5rem;
    border-radius: 50%;
    background: var(--rg-warning);
    box-shadow: 0 0 0 5px rgba(255, 211, 138, 0.12);
}

[data-testid="stMetric"] {
    padding: 0.8rem 0.9rem;
    border: 1px solid rgba(126, 211, 255, 0.16);
    border-radius: 15px;
    background:
        linear-gradient(145deg, rgba(11, 34, 58, 0.92), rgba(5, 18, 34, 0.94));
    box-shadow:
        0 8px 22px rgba(0, 0, 0, 0.20),
        0 0 20px rgba(72, 169, 255, 0.04);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: var(--rg-text);
}

[data-testid="stImage"] {
    padding: 0.65rem;
    border: 1px solid rgba(119, 233, 255, 0.22);
    border-radius: 20px;
    background: rgba(0, 5, 12, 0.86);
    box-shadow:
        0 0 26px rgba(72, 169, 255, 0.18),
        0 22px 60px rgba(0, 0, 0, 0.38),
        inset 0 0 28px rgba(119, 233, 255, 0.04);
}

[data-testid="stImage"] img {
    border-radius: 13px;
    filter: contrast(1.03) brightness(1.02);
}

[data-testid="stFileUploader"] {
    padding: 0.55rem;
    border: 1px solid rgba(119, 233, 255, 0.16);
    border-radius: 18px;
    background: rgba(6, 23, 41, 0.72);
}

[data-testid="stFileUploaderDropzone"] {
    border-color: rgba(119, 233, 255, 0.28);
    background: rgba(8, 33, 57, 0.62);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.45rem;
    padding: 0.35rem;
    border: 1px solid rgba(119, 233, 255, 0.11);
    border-radius: 15px;
    background: rgba(4, 16, 30, 0.72);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 1rem;
    border-radius: 11px;
    color: #b9d6e3;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    color: white !important;
    background: linear-gradient(135deg, #0d4e83, #167fc1) !important;
    box-shadow: 0 0 20px rgba(72, 169, 255, 0.18);
}

[data-testid="stExpander"] {
    border: 1px solid rgba(119, 233, 255, 0.14);
    border-radius: 15px;
    background: rgba(6, 22, 40, 0.68);
}

[data-testid="stAlert"] {
    border-radius: 15px;
    background-color: rgba(8, 26, 44, 0.92);
    color: var(--rg-text);
}

[data-testid="stAlert"] * {
    color: #e7f7ff;
}

.stButton > button,
.stDownloadButton > button {
    min-height: 2.8rem;
    border: 1px solid rgba(119, 233, 255, 0.28);
    border-radius: 13px;
    color: #effcff;
    background: linear-gradient(135deg, #0f65a2, #168bc8);
    box-shadow:
        0 0 18px rgba(72, 169, 255, 0.18),
        inset 0 0 12px rgba(255, 255, 255, 0.05);
    transition:
        transform 150ms ease,
        box-shadow 150ms ease,
        border-color 150ms ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(217, 247, 255, 0.55);
    box-shadow:
        0 0 28px rgba(72, 169, 255, 0.32),
        0 8px 22px rgba(0, 0, 0, 0.28);
}

.stButton > button:disabled {
    color: #7793a2;
    background: rgba(23, 46, 66, 0.72);
    box-shadow: none;
}


/* Streamlit 1.60 button visibility override.
   Keep all button labels readable in normal, hover, focus, active,
   and disabled states—even when Streamlit's generated styles change. */
div.stButton > button,
div.stDownloadButton > button,
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-tertiary"],
[data-testid="stDownloadButton"] button,
[data-testid="stFileUploaderDropzone"] button {
    min-height: 2.8rem !important;
    border: 1px solid rgba(119, 233, 255, 0.34) !important;
    border-radius: 13px !important;
    color: #effcff !important;
    -webkit-text-fill-color: #effcff !important;
    background: linear-gradient(135deg, #0d568e, #1387c5) !important;
    box-shadow:
        0 0 18px rgba(72, 169, 255, 0.20),
        inset 0 0 12px rgba(255, 255, 255, 0.05) !important;
    opacity: 1 !important;
}

div.stButton > button *,
div.stDownloadButton > button *,
[data-testid="stBaseButton-primary"] *,
[data-testid="stBaseButton-secondary"] *,
[data-testid="stBaseButton-tertiary"] *,
[data-testid="stDownloadButton"] button *,
[data-testid="stFileUploaderDropzone"] button * {
    color: #effcff !important;
    -webkit-text-fill-color: #effcff !important;
    opacity: 1 !important;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-tertiary"]:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFileUploaderDropzone"] button:hover {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-color: rgba(217, 247, 255, 0.72) !important;
    background: linear-gradient(135deg, #1474b7, #1ba7db) !important;
    box-shadow:
        0 0 30px rgba(72, 169, 255, 0.38),
        0 8px 24px rgba(0, 0, 0, 0.30) !important;
    transform: translateY(-1px);
}

div.stButton > button:hover *,
div.stDownloadButton > button:hover *,
[data-testid="stBaseButton-primary"]:hover *,
[data-testid="stBaseButton-secondary"]:hover *,
[data-testid="stBaseButton-tertiary"]:hover *,
[data-testid="stDownloadButton"] button:hover *,
[data-testid="stFileUploaderDropzone"] button:hover * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

div.stButton > button:focus,
div.stButton > button:focus-visible,
div.stDownloadButton > button:focus,
div.stDownloadButton > button:focus-visible,
[data-testid="stBaseButton-primary"]:focus,
[data-testid="stBaseButton-secondary"]:focus,
[data-testid="stBaseButton-tertiary"]:focus,
[data-testid="stDownloadButton"] button:focus,
[data-testid="stFileUploaderDropzone"] button:focus {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    outline: 2px solid rgba(119, 233, 255, 0.78) !important;
    outline-offset: 2px !important;
    box-shadow:
        0 0 0 4px rgba(72, 169, 255, 0.18),
        0 0 26px rgba(72, 169, 255, 0.30) !important;
}

div.stButton > button:active,
div.stDownloadButton > button:active,
[data-testid="stBaseButton-primary"]:active,
[data-testid="stBaseButton-secondary"]:active,
[data-testid="stBaseButton-tertiary"]:active,
[data-testid="stDownloadButton"] button:active,
[data-testid="stFileUploaderDropzone"] button:active {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background: linear-gradient(135deg, #0b4a7b, #0f78ad) !important;
    transform: translateY(0);
}

div.stButton > button:disabled,
div.stDownloadButton > button:disabled,
[data-testid="stBaseButton-primary"]:disabled,
[data-testid="stBaseButton-secondary"]:disabled,
[data-testid="stBaseButton-tertiary"]:disabled,
[data-testid="stDownloadButton"] button:disabled,
[data-testid="stFileUploaderDropzone"] button:disabled {
    color: #b8cfda !important;
    -webkit-text-fill-color: #b8cfda !important;
    border-color: rgba(126, 211, 255, 0.12) !important;
    background: rgba(24, 50, 72, 0.88) !important;
    box-shadow: none !important;
    opacity: 0.72 !important;
    cursor: not-allowed !important;
}

div.stButton > button:disabled *,
div.stDownloadButton > button:disabled *,
[data-testid="stBaseButton-primary"]:disabled *,
[data-testid="stBaseButton-secondary"]:disabled *,
[data-testid="stBaseButton-tertiary"]:disabled *,
[data-testid="stDownloadButton"] button:disabled *,
[data-testid="stFileUploaderDropzone"] button:disabled * {
    color: #b8cfda !important;
    -webkit-text-fill-color: #b8cfda !important;
}

/* Keep link-style and toolbar buttons visible against the dark interface. */
button[aria-label],
[data-testid="stToolbar"] button,
[data-testid="stHeader"] button {
    color: #dff8ff !important;
    -webkit-text-fill-color: #dff8ff !important;
}

button[aria-label] svg,
[data-testid="stToolbar"] button svg,
[data-testid="stHeader"] button svg {
    fill: currentColor !important;
    color: #dff8ff !important;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] {
    color: var(--rg-text);
    border-color: rgba(119, 233, 255, 0.16);
    background: rgba(5, 19, 35, 0.88);
}

[data-baseweb="select"] * {
    color: var(--rg-text);
}

textarea,
input {
    color: var(--rg-text) !important;
}

[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #287fd2, #7ceaff);
    box-shadow: 0 0 12px rgba(119, 233, 255, 0.30);
}

hr {
    border-color: rgba(119, 233, 255, 0.10);
}

.footer {
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid rgba(119, 233, 255, 0.12);
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    color: var(--rg-muted);
    font-size: 0.82rem;
}

.footer span {
    color: var(--rg-muted);
}

@media (max-width: 700px) {
    .hero {
        padding: 1.5rem;
    }

    .hero-grid {
        align-items: flex-start;
    }

    .brand-mark {
        width: 56px;
        height: 56px;
        flex-basis: 56px;
        border-radius: 16px;
    }

    .footer {
        flex-direction: column;
    }
}

@media (prefers-reduced-motion: reduce) {
    .hero:before {
        animation: none;
        display: none;
    }

    .stButton > button,
    .stDownloadButton > button {
        transition: none;
    }
}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-grid">
                <div class="brand-mark">RG</div>
                <div>
                    <div class="eyebrow">Medical Imaging AI Research Sandbox</div>
                    <h1>RadiantGuard AI</h1>
                    <p>
                        Exploring medical-imaging AI through transparent research
                        workflows, visual explainability, uncertainty-aware evaluation,
                        and uncompromising human oversight.
                    </p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Educational research prototype only. Not clinically validated, not a "
        "medical device, and not for diagnosis or treatment. All information "
        "must be reviewed and double-checked with a qualified physician or radiologist."
    )


def render_metric_card(label: str, value: str) -> None:
    safe_label = escape(str(label))
    safe_value = escape(str(value))

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{safe_label}</div>
            <div class="metric-value">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer(version: str) -> None:
    safe_version = escape(str(version))

    st.markdown(
        f"""
        <div class="footer">
            <span>RadiantGuard AI · Responsible medical-imaging AI research</span>
            <span>Prototype {safe_version} · Human review required</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
