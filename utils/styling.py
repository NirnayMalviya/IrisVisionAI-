"""
styling.py
----------
Central place for the IrisVision AI visual theme: a grayscale liquid-glass
morphism UI built with injected CSS. No colored accents are used anywhere -
only white / gray / black - to match the Bloom-style aesthetic requested.

Two tiers of glass panels are provided:
    .glass-light  -> background rgba(255,255,255,0.04), blur(6px)
    .glass-strong -> background rgba(255,255,255,0.07), blur(20px)
Used for hero sections, the leaderboard card, and the prediction card.
"""

import streamlit as st

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Source+Serif+4:ital@1&display=swap" rel="stylesheet">

<style>
:root {
    --bg-top: #0a0a0b;
    --bg-bottom: #1c1c1f;
    --glass-light-bg: rgba(255, 255, 255, 0.04);
    --glass-strong-bg: rgba(255, 255, 255, 0.07);
    --glass-border: rgba(255, 255, 255, 0.12);
    --text-primary: #f2f2f2;
    --text-muted: #a3a3a8;
}

/* ---------- Global ---------- */
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(circle at 20% 0%, #232326 0%, var(--bg-top) 45%, var(--bg-bottom) 100%);
    background-attachment: fixed;
}

em, i, .serif-accent {
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-weight: 400;
}

/* Hide default Streamlit chrome for a cleaner look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent !important;}

/* ---------- Glass panels ---------- */
.glass-light {
    background: var(--glass-light-bg);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid var(--glass-border);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    border-radius: 1rem;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}

.glass-strong {
    background: var(--glass-strong-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.10), 0 8px 32px rgba(0,0,0,0.35);
    border-radius: 1.25rem;
    padding: 2rem 2.25rem;
    margin-bottom: 1.5rem;
}

/* ---------- Hero ---------- */
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
    background: linear-gradient(180deg, #ffffff 0%, #b8b8bd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
    max-width: 640px;
}

/* ---------- Pills ---------- */
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin: 0.75rem 0 0.25rem 0;
}

.pill {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-primary);
    transition: transform 0.18s ease, background 0.18s ease;
}

.pill:hover {
    transform: scale(1.06);
    background: rgba(255,255,255,0.12);
}

/* ---------- Section headers ---------- */
.section-title {
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}

.section-caption {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-bottom: 1rem;
}

.best-model-tag {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.3);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 999px !important;
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.4rem !important;
    transition: transform 0.18s ease, background 0.18s ease !important;
}

.stButton > button:hover {
    transform: scale(1.04);
    background: rgba(255,255,255,0.16) !important;
    border-color: rgba(255,255,255,0.4) !important;
    color: var(--text-primary) !important;
}

.stButton > button:focus:not(:active) {
    color: var(--text-primary) !important;
}

/* Primary CTA variant */
button[kind="primary"] {
    background: rgba(255,255,255,0.9) !important;
    color: #0a0a0b !important;
    border: 1px solid rgba(255,255,255,0.9) !important;
    font-weight: 600 !important;
}
button[kind="primary"]:hover {
    background: #ffffff !important;
    color: #0a0a0b !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: rgba(10,10,11,0.6);
    backdrop-filter: blur(14px);
    border-right: 1px solid var(--glass-border);
}

/* ---------- Tables / dataframes ---------- */
[data-testid="stDataFrame"] {
    border-radius: 0.75rem;
    overflow: hidden;
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: var(--glass-light-bg);
    border: 1px solid var(--glass-border);
    border-radius: 0.9rem;
    padding: 0.9rem 1rem;
}

/* ---------- Sliders ---------- */
.stSlider [data-baseweb="slider"] > div > div {
    background: rgba(255,255,255,0.35) !important;
}

/* ---------- Divider ---------- */
hr {
    border-color: rgba(255,255,255,0.12) !important;
}
</style>
"""


def inject_css():
    """Injects the global grayscale glassmorphism CSS theme into the app."""
    st.markdown(CSS, unsafe_allow_html=True)


def glass_start(strong: bool = False):
    """Opens a glass-panel div. Must be paired with glass_end()."""
    cls = "glass-strong" if strong else "glass-light"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)


def glass_end():
    """Closes a glass-panel div opened by glass_start()."""
    st.markdown("</div>", unsafe_allow_html=True)


def render_pills(labels):
    """Renders a row of pill-style tags."""
    pills_html = "".join(f'<span class="pill">{label}</span>' for label in labels)
    st.markdown(f'<div class="pill-row">{pills_html}</div>', unsafe_allow_html=True)


def section_header(title: str, caption: str = ""):
    """Renders a consistent section title + optional caption."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)
