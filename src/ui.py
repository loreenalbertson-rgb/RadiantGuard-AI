from __future__ import annotations

from html import escape

import streamlit as st


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --rg-navy: #071625;
    --rg-navy-soft: #0f2638;
    --rg-teal: #21c7b7;
    --rg-cyan: #5ae2ec;
    --rg-mist: #eef8f8;
    --rg-line: rgba(31, 74, 93, .16);
    --rg-text: #102434;
    --rg-muted: #607483;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Manrope', sans-serif; letter-spacing: -.025em; }

.stApp {
    background:
        radial-gradient(circle at 83% 3%, rgba(90, 226, 236, .14), transparent 26rem),
        linear-gradient(180deg, #f8fcfd 0%, #ffffff 44%);
    color: var(--rg-text);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--rg-navy) 0%, #0b2031 100%);
    color: white;
    border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] * { color: #e8f4f5; }
[data-testid="stSidebar"] .stAlert * { color: var(--rg-text); }

.block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem; }

.hero {
    position: relative;
    overflow: hidden;
    padding: 2.25rem 2.4rem;
    margin-bottom: 1.4rem;
    border-radius: 24px;
    color: white;
    background: linear-gradient(125deg, #061725 0%, #0c2c3f 62%, #0b4d54 100%);
    box-shadow: 0 26px 70px rgba(7, 31, 47, .18);
}
.hero:after {
    content: '';
    position: absolute;
    width: 330px;
    height: 330px;
    right: -90px;
    top: -160px;
    border-radius: 50%;
    border: 1px solid rgba(90,226,236,.35);
    box-shadow: 0 0 80px rgba(33,199,183,.22), inset 0 0 60px rgba(33,199,183,.12);
}
.hero-grid { display: flex; align-items: center; gap: 1.25rem; position: relative; z-index: 2; }
.brand-mark {
    width: 72px;
    height: 72px;
    flex: 0 0 72px;
    display: grid;
    place-items: center;
    border-radius: 20px;
    font-family: 'Manrope', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -.04em;
    color: #071625;
    background: linear-gradient(145deg, var(--rg-cyan), var(--rg-teal));
    box-shadow: 0 12px 35px rgba(33,199,183,.28);
}
.brand-mark.small { width: 42px; height: 42px; flex-basis: 42px; border-radius: 12px; font-size: .86rem; }
.hero h1 { font-size: clamp(2rem, 4vw, 3.15rem); margin: 0 0 .25rem; color: white; }
.hero p { max-width: 820px; margin: .25rem 0 0; color: #cbe3e6; font-size: 1.02rem; }
.eyebrow { color: #77e5df; font-size: .76rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }

.sidebar-brand { display: flex; align-items: center; gap: .8rem; margin: .35rem 0 1rem; }
.sidebar-title { font-family: 'Manrope', sans-serif; font-weight: 800; }
.sidebar-subtitle { font-size: .76rem; color: #9fc2c8 !important; }

.safety-strip {
    display: flex;
    flex-direction: column;
    gap: .2rem;
    padding: 1rem 1.1rem;
    border: 1px solid rgba(33,199,183,.25);
    border-radius: 14px;
    background: rgba(33,199,183,.07);
}
.safety-strip span { color: var(--rg-muted); font-size: .9rem; }

.empty-state {
    margin-top: 1rem;
    padding: 4.4rem 2rem;
    border: 1px dashed rgba(35, 99, 118, .28);
    border-radius: 22px;
    text-align: center;
    background: rgba(255,255,255,.68);
}
.empty-state.compact { padding: 2.2rem 1.5rem; }
.empty-state h3 { margin: .25rem 0; }
.empty-state p { margin: .35rem auto 0; max-width: 640px; color: var(--rg-muted); }
.empty-icon { margin: auto; font-size: 2rem; color: var(--rg-teal); }

.metric-card {
    min-height: 94px;
    padding: .95rem 1rem;
    margin-bottom: .75rem;
    border: 1px solid var(--rg-line);
    border-radius: 16px;
    background: rgba(255,255,255,.9);
    box-shadow: 0 8px 24px rgba(29, 72, 88, .06);
}
.metric-label { color: var(--rg-muted); font-size: .77rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.metric-value { margin-top: .32rem; color: var(--rg-text); font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 1.05rem; }

.model-placeholder {
    padding: 1.15rem 1.25rem;
    border: 1px solid var(--rg-line);
    border-radius: 17px;
    background: linear-gradient(135deg, rgba(7,22,37,.035), rgba(33,199,183,.06));
}
.model-placeholder p { margin: .65rem 0 0; color: var(--rg-muted); }
.status-dot { display: inline-block; width: 9px; height: 9px; margin-right: .5rem; border-radius: 50%; background: #e8a73c; box-shadow: 0 0 0 5px rgba(232,167,60,.13); }

[data-testid="stMetric"] {
    padding: .75rem .8rem;
    border: 1px solid var(--rg-line);
    border-radius: 14px;
    background: rgba(255,255,255,.82);
}

.stTabs [data-baseweb="tab-list"] { gap: .45rem; }
.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 1rem;
    border-radius: 12px;
    background: rgba(13, 58, 76, .055);
}
.stTabs [aria-selected="true"] { background: #0d3549 !important; color: white !important; }

.footer {
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--rg-line);
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    color: var(--rg-muted);
    font-size: .82rem;
}

@media (max-width: 700px) {
    .hero { padding: 1.5rem; }
    .hero-grid { align-items: flex-start; }
    .brand-mark { width: 54px; height: 54px; flex-basis: 54px; border-radius: 15px; }
    .footer { flex-direction: column; }
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
                        Exploring explainable medical-imaging AI through transparent workflows,
                        uncertainty-aware evaluation, and uncompromising human oversight.
                    </p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Educational research prototype only. Not clinically validated, not a medical device, and not for diagnosis or treatment. "
        "All information must be reviewed and double-checked with a qualified physician or radiologist."
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
    st.markdown(
        f"""
        <div class="footer">
            <span>RadiantGuard AI · Responsible medical-imaging AI research</span>
            <span>Prototype {version} · Human review required</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
