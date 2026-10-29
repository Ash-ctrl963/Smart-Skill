"""
Multimodal Construction Intelligence Platform
Skill Wallet Capstone Project 1

Streamlit + Google Gemini (multimodal). No database. Deploys on Streamlit Community Cloud.
Run locally:  streamlit run app.py
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------------------
# Page config & styling
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Construction Intelligence Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root { --brand:#f59e0b; --ink:#0f172a; --muted:#64748b; --card:#ffffff; --line:#e2e8f0; }
.block-container { padding-top: 1.6rem; max-width: 1250px; }
.hero {
    background: linear-gradient(120deg,#0f172a 0%,#1e293b 55%,#334155 100%);
    border-radius: 16px; padding: 22px 28px; color: #f8fafc; margin-bottom: 18px;
    border-left: 6px solid var(--brand);
}
.hero h1 { margin: 0; font-size: 1.75rem; color: #f8fafc; }
.hero p { margin: 4px 0 0 0; color: #cbd5e1; font-size: .95rem; }
.metric-card {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 16px 18px; box-shadow: 0 1px 3px rgba(15,23,42,.06); height: 100%;
}
.metric-card .label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; }
.metric-card .value { color: var(--ink); font-size: 1.7rem; font-weight: 700; margin-top: 2px; }
.metric-card .sub { color: var(--muted); font-size: .8rem; margin-top: 2px; }
.pill {
    display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .78rem;
    font-weight: 600; letter-spacing: .02em;
}
.pill-green  { background:#dcfce7; color:#166534; }
.pill-yellow { background:#fef9c3; color:#854d0e; }
.pill-orange { background:#ffedd5; color:#9a3412; }
.pill-red    { background:#fee2e2; color:#991b1b; }
.pill-grey   { background:#e2e8f0; color:#334155; }
.hazard {
    background: var(--card); border: 1px solid var(--line); border-left-width: 5px;
    border-radius: 10px; padding: 12px 16px; margin-bottom: 10px;
}
.hazard.Low { border-left-color:#22c55e; } .hazard.Medium { border-left-color:#eab308; }
.hazard.High { border-left-color:#f97316; } .hazard.Critical { border-left-color:#dc2626; }
.hazard .title { font-weight: 600; color: var(--ink); }
.hazard .remedy { color: #334155; font-size: .9rem; margin-top: 4px; }
.alert-box { border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; border: 1px solid var(--line); }
.alert-red { background:#fef2f2; border-color:#fecaca; }
.alert-amber { background:#fffbeb; border-color:#fde68a; }
.alert-green { background:#f0fdf4; border-color:#bbf7d0; }
.alert-blue { background:#eff6ff; border-color:#bfdbfe; }
.small-note { color: var(--muted); font-size: .82rem; }
footer { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_IMAGE_SIDE = 1600
DISCLAIMER = (
    "AI-generated screening output for educational/decision-support use only. "
    "It does not replace inspection by a certified safety officer or a licensed structural engineer."
)

PPE_ITEMS = ["helmet", "high_vis_vest", "steel_toe_boots", "safety_harness"]
PPE_LABELS = {
    "helmet": "Hard Helmet",
    "high_vis_vest": "High-Vis Vest",
    "steel_toe_boots": "Steel-Toe Boots",
    "safety_harness": "Safety Harness",
}
SEVERITIES = ["Low", "Medium", "High", "Critical"]
DEFECT_TYPES = ["Hairline", "Settlement", "Structural Shear", "Thermal Expansion", "Spalling"]

# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
SAFETY_PROMPT = """You are a senior construction safety officer (OSHA 29 CFR 1926 and BIS / IS 456:2000 aware).
Analyse the attached construction-site photograph for PPE compliance and site hazards.

Check every visible worker for: hard helmet, high-visibility vest, steel-toe safety boots, safety harness
(harness is mandatory only when work at height > 1.8 m / 6 ft is visible; otherwise mark "not_applicable").
Also look for hazards: unprotected edges, poor scaffolding, loose materials, exposed rebar, electrical risks,
housekeeping, formwork / shuttering issues, excavation safety, and any visible IS 456 concerns
(cover, curing, honeycombing, formwork supports).

Return ONLY valid JSON with exactly this schema:
{
  "safety_score": <integer 0-100>,
  "overall_status": "Compliant" | "Needs Improvement" | "Non-Compliant",
  "workers_detected": <integer>,
  "ppe_detection": {
     "helmet": "compliant" | "non_compliant" | "not_visible" | "not_applicable",
     "high_vis_vest": "compliant" | "non_compliant" | "not_visible" | "not_applicable",
     "steel_toe_boots": "compliant" | "non_compliant" | "not_visible" | "not_applicable",
     "safety_harness": "compliant" | "non_compliant" | "not_visible" | "not_applicable"
  },
  "hazards": [
     {"hazard": "<short title>", "severity": "Low" | "Medium" | "High" | "Critical",
      "standard_reference": "<OSHA / IS code or 'General best practice'>",
      "remedy": "<specific corrective action>"}
  ],
  "summary": "<2-3 sentence plain-language summary>"
}
If the image is not a construction site, set safety_score to 0, overall_status to "Non-Compliant",
and explain in summary."""

CRACK_PROMPT = """You are a licensed structural engineer experienced with IS 456:2000, IS 15988 and IS 13935.
Analyse the attached photograph of a concrete / masonry surface for cracks and defects.

Classify the dominant defect as exactly one of:
Hairline, Settlement, Structural Shear, Thermal Expansion, Spalling.
Estimate the maximum crack width in millimetres from visual cues (use a reference object if visible; state uncertainty).

Return ONLY valid JSON with exactly this schema:
{
  "defect_type": "Hairline" | "Settlement" | "Structural Shear" | "Thermal Expansion" | "Spalling",
  "confidence": <integer 0-100>,
  "estimated_crack_width_mm": <number>,
  "crack_pattern": "<vertical / horizontal / diagonal / map-pattern / stepped, location notes>",
  "severity": "Low" | "Medium" | "High" | "Critical",
  "urgency": "Monitor" | "Repair within 3 months" | "Repair within 2 weeks" | "Immediate action",
  "root_causes": ["<cause 1>", "<cause 2>"],
  "recommended_repairs": [
     {"method": "<e.g. Epoxy injection, Structural re-grouting, Polymer-modified mortar patching, Carbon-fibre wrapping>",
      "details": "<procedure and materials>"}
  ],
  "monitoring_plan": "<crack gauges / tell-tales / re-inspection interval>",
  "summary": "<2-3 sentence summary>"
}
If the image does not show a structural surface, set confidence to 0 and explain in summary."""

ASSISTANT_SYSTEM = """You are 'SiteMate', an expert construction engineering assistant for Indian building sites.
You know IS 456:2000 (RCC), IS 800 (steel), IS 1200 (measurement), IS 2502, NBC 2016, OSHA 1926 and BOCW rules.
Give practical, concise, numbered guidance with quantities, grades, and codes where relevant.
Flag safety-critical items clearly. If a question needs a site inspection or structural calculation by a licensed
engineer, say so. Keep answers under 250 words unless asked for detail."""

# --------------------------------------------------------------------------------------
# Demo-mode data (used when no API key is supplied)
# --------------------------------------------------------------------------------------
DEMO_SAFETY = {
    "safety_score": 58,
    "overall_status": "Needs Improvement",
    "workers_detected": 3,
    "ppe_detection": {
        "helmet": "compliant",
        "high_vis_vest": "non_compliant",
        "steel_toe_boots": "not_visible",
        "safety_harness": "non_compliant",
    },
    "hazards": [
        {"hazard": "Worker at height without fall-arrest harness", "severity": "Critical",
         "standard_reference": "OSHA 1926.501 / BOCW Rules 1998",
         "remedy": "Stop work at height until full-body harnesses are anchored to a rated lifeline; add edge guardrails."},
        {"hazard": "Exposed projecting rebar without caps", "severity": "High",
         "standard_reference": "OSHA 1926.701(b)",
         "remedy": "Fit mushroom caps or bend rebar ends; mark the zone with barricade tape."},
        {"hazard": "Loose material and debris on walkways", "severity": "Medium",
         "standard_reference": "General best practice / NBC 2016 Part 7",
         "remedy": "Assign housekeeping crew; maintain 1 m clear access paths."},
        {"hazard": "Formwork props show inadequate bracing", "severity": "Medium",
         "standard_reference": "IS 456:2000 Cl. 11 / IS 14687",
         "remedy": "Add diagonal bracing and verify prop spacing before the next pour."},
    ],
    "summary": "DEMO DATA: The site shows partial PPE compliance. Fall protection and high-visibility "
               "clothing are the main gaps and should be corrected before work continues.",
}

DEMO_CRACK = {
    "defect_type": "Structural Shear",
    "confidence": 82,
    "estimated_crack_width_mm": 2.4,
    "crack_pattern": "Diagonal 45° crack near beam-column junction, widening toward the support.",
    "severity": "High",
    "urgency": "Repair within 2 weeks",
    "root_causes": [
        "Insufficient shear reinforcement (stirrup spacing too wide near support)",
        "Overloading or load redistribution after construction",
        "Possible differential settlement of the column footing",
    ],
    "recommended_repairs": [
        {"method": "Epoxy injection",
         "details": "Clean crack, install injection ports at 150 mm c/c, seal with epoxy putty, inject low-viscosity epoxy resin at low pressure."},
        {"method": "Structural re-grouting / jacketing",
         "details": "Provide a micro-concrete or fibre-reinforced jacket with additional stirrups after a structural engineer's design."},
    ],
    "monitoring_plan": "Install tell-tale gauges across the crack; re-measure weekly for one month, then monthly for six months.",
    "summary": "DEMO DATA: A diagonal shear-type crack of roughly 2.4 mm is a structural concern. "
               "Engage a licensed structural engineer and avoid additional loading on the member.",
}

DEMO_CHAT = {
    "curing": "**Concrete curing (IS 456 Cl. 13.5):**\n1. Start curing within 24 h; keep moist for at least 7 days (OPC) or 10 days (PPC / mineral admixtures).\n2. Use ponding, wet hessian or curing compound.\n3. In hot weather, extend to 14 days and shade fresh pours.\n\n*Demo mode: add a Gemini API key for live answers.*",
    "slump": "**Slump guidance (IS 456 Table 7):** Beams/slabs 25-75 mm, columns 50-100 mm, pumped concrete 100-150 mm. Always check workability on every truck.\n\n*Demo mode: add a Gemini API key for live answers.*",
    "cement": "**Cement estimate rule of thumb:** ~0.4 bags per sq ft built-up for RCC residential. For M25 concrete, about 8 bags/m³ (nominal mix 1:1:2 is roughly 12 bags/m³).\n\n*Demo mode: add a Gemini API key for live answers.*",
    "steel": "**Steel estimate rule of thumb:** ~3.5-4.5 kg per sq ft built-up for residential RCC; slabs ~80-100 kg/m³ of concrete, beams 150-200 kg/m³, columns 200-250 kg/m³.\n\n*Demo mode: add a Gemini API key for live answers.*",
    "default": "I'm running in **demonstration mode** (no API key). I can answer a few canned topics: *curing, slump, cement, steel*. Add your Gemini API key in the sidebar for full multi-turn engineering assistance.",
}

# --------------------------------------------------------------------------------------
# Gemini helpers
# --------------------------------------------------------------------------------------
def get_client(api_key: str):
    """Create a Gemini client lazily so the app still runs without the SDK/key."""
    from google import genai  # imported lazily
    return genai.Client(api_key=api_key)


def prepare_image(uploaded) -> Image.Image:
    """Open an uploaded image, normalise orientation/mode and cap the size."""
    img = Image.open(uploaded)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return img


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from model text, tolerating markdown fences and stray prose."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def analyse_image(api_key: str, model: str, prompt: str, image: Image.Image) -> dict[str, Any]:
    from google.genai import types

    client = get_client(api_key)
    response = client.models.generate_content(
        model=model,
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return extract_json(response.text)


def chat_reply(api_key: str, model: str, history: list[dict[str, str]], context: str) -> str:
    from google.genai import types

    client = get_client(api_key)
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in history
    ]
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=ASSISTANT_SYSTEM + ("\n\nCurrent site context: " + context if context else ""),
            temperature=0.5,
        ),
    )
    return response.text or "I could not generate a response. Please try rephrasing."


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "permission" in low or "401" in low or "403" in low:
        return "The Gemini API rejected the key. Check that it is valid and has access to the selected model."
    if "quota" in low or "429" in low or "resource_exhausted" in low:
        return "Rate limit or quota reached. Wait a moment and try again, or use a different key."
    if "404" in low or "not found" in low:
        return "The selected model was not found. Pick another model name in the sidebar."
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "The model returned a response that could not be parsed as JSON. Please run the analysis again."
    if isinstance(exc, ModuleNotFoundError):
        return "The `google-genai` package is missing. Run `pip install -r requirements.txt`."
    return f"Unexpected error: {msg[:300]}"


# --------------------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------------------
def esc(value: Any) -> str:
    return html.escape(str(value))


def metric_card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="metric-card"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>'
    )


def pill(text: str, color: str) -> str:
    return f'<span class="pill pill-{color}">{esc(text)}</span>'


def severity_color(sev: str) -> str:
    return {"Low": "green", "Medium": "yellow", "High": "orange", "Critical": "red"}.get(sev, "grey")


def status_color(status: str) -> str:
    s = status.lower()
    if "non" in s:
        return "red"
    if "improve" in s:
        return "yellow"
    if "compliant" in s:
        return "green"
    return "grey"


def score_color(score: float) -> str:
    return "green" if score >= 80 else "yellow" if score >= 60 else "orange" if score >= 40 else "red"


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def download_json(label: str, data: dict[str, Any], filename: str, key: str) -> None:
    st.download_button(
        label,
        data=json.dumps(data, indent=2, ensure_ascii=False),
        file_name=filename,
        mime="application/json",
        key=key,
    )


def format_inr(amount: float) -> str:
    """Format INR in Crores / Lakhs."""
    if amount >= 1e7:
        return f"₹{amount / 1e7:,.2f} Cr"
    if amount >= 1e5:
        return f"₹{amount / 1e5:,.2f} L"
    return f"₹{amount:,.0f}"


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
def render_sidebar() -> tuple[str, str, bool]:
    with st.sidebar:
        st.markdown("## 🏗️ Construction Intelligence")
        st.caption("Skill Wallet Capstone · Project 1")

        st.markdown("### 🔐 Gemini API Key")
        key_input = st.text_input(
            "API key",
            type="password",
            placeholder="Paste your Google AI Studio key",
            help="Used only for this session and never stored. Get one at aistudio.google.com.",
            label_visibility="collapsed",
        )
        secret_key = ""
        try:
            secret_key = st.secrets.get("GEMINI_API_KEY", "")  # Streamlit Cloud secret
        except Exception:
            secret_key = ""
        api_key = (key_input or secret_key or "").strip()

        model = st.text_input("Model", value=DEFAULT_MODEL, help="Any multimodal Gemini model name.")

        demo_mode = not api_key
        if demo_mode:
            st.warning("**Demonstration mode**: no key supplied. Analyses return sample data.", icon="🧪")
        else:
            st.success("Live mode: Gemini connected.", icon="✅")

        st.markdown("### 🧭 Features")
        st.markdown(
            "- 🦺 **Safety & PPE Audit**\n"
            "- 🧱 **Crack Diagnostics**\n"
            "- 💰 **Planning & Valuation**\n"
            "- 🌦️ **Daily Site Assistant**"
        )

        with st.expander("🧩 Architecture overview"):
            st.markdown(
                """
```
 Streamlit UI (4 tabs)
      │
      ├─ Image tabs ──► PIL preprocess ──► Gemini multimodal
      │                      (JSON mode)      └─► structured JSON
      │                                            └─► cards / pills / download
      ├─ Valuation ───► Pure-Python cost engine + Pollinations render URL
      └─ Assistant ───► Gemini chat (session-state memory)
                         + rule-based weather advisories
```
Stateless: no database, session state only.
                """
            )
        st.caption(DISCLAIMER)
    return api_key, model, demo_mode


# --------------------------------------------------------------------------------------
# Tab 1: Safety
# --------------------------------------------------------------------------------------
def render_safety_tab(api_key: str, model: str, demo_mode: bool) -> None:
    st.subheader("🦺 Site Safety & PPE Audit")
    st.caption("Upload a site photo to audit PPE compliance and detect hazards (OSHA 1926 · IS 456:2000).")

    uploaded = st.file_uploader("Site photo", type=["jpg", "jpeg", "png"], key="safety_upload")
    if not uploaded:
        st.info("Upload a JPG/PNG photo to begin.")
        return

    try:
        image = prepare_image(uploaded)
    except Exception:
        st.error("That file could not be read as an image. Try another JPG or PNG.")
        return

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.image(image, caption="Uploaded site photo", use_container_width=True)
        run = st.button("🔍 Run safety audit", type="primary", use_container_width=True, key="safety_run")

    if run:
        with st.spinner("Analysing PPE and hazards..."):
            try:
                result = DEMO_SAFETY if demo_mode else analyse_image(api_key, model, SAFETY_PROMPT, image)
                st.session_state["safety_result"] = result
                st.session_state["safety_demo"] = demo_mode
            except Exception as exc:
                st.session_state.pop("safety_result", None)
                st.error(friendly_error(exc))

    result = st.session_state.get("safety_result")
    if not result:
        return

    with right:
        if st.session_state.get("safety_demo"):
            st.info("Showing **sample data** (demonstration mode).", icon="🧪")

        score = as_int(result.get("safety_score"))
        status = str(result.get("overall_status", "Unknown"))
        hazards = result.get("hazards") or []
        critical = sum(1 for h in hazards if h.get("severity") == "Critical")

        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Safety score", f"{score}/100", "Higher is safer"), unsafe_allow_html=True)
        c2.markdown(metric_card("Workers detected", str(as_int(result.get("workers_detected"))), "In frame"),
                    unsafe_allow_html=True)
        c3.markdown(metric_card("Critical hazards", str(critical), f"{len(hazards)} total"),
                    unsafe_allow_html=True)
        st.progress(min(max(score, 0), 100) / 100)
        st.markdown(
            f"Overall status: {pill(status, status_color(status))} &nbsp; "
            f"Score band: {pill('Good' if score >= 80 else 'Fair' if score >= 60 else 'Poor', score_color(score))}",
            unsafe_allow_html=True,
        )
        if result.get("summary"):
            st.write(result["summary"])

    st.markdown("#### PPE checklist")
    ppe = result.get("ppe_detection") or {}
    cols = st.columns(len(PPE_ITEMS))
    state_style = {
        "compliant": ("✅ Compliant", "green"),
        "non_compliant": ("❌ Missing", "red"),
        "not_visible": ("👁️ Not visible", "grey"),
        "not_applicable": ("➖ N/A", "grey"),
    }
    for col, item in zip(cols, PPE_ITEMS):
        label, color = state_style.get(str(ppe.get(item, "not_visible")), ("👁️ Not visible", "grey"))
        col.markdown(
            f'<div class="metric-card"><div class="label">{PPE_LABELS[item]}</div>'
            f'<div style="margin-top:8px">{pill(label, color)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Identified hazards")
    if not hazards:
        st.success("No hazards identified.")
    for h in sorted(hazards, key=lambda x: SEVERITIES.index(x.get("severity")) if x.get("severity") in SEVERITIES else -1,
                    reverse=True):
        sev = h.get("severity", "Low")
        st.markdown(
            f'<div class="hazard {esc(sev)}"><div class="title">{esc(h.get("hazard", "Hazard"))} '
            f'&nbsp;{pill(sev, severity_color(sev))}</div>'
            f'<div class="small-note">Reference: {esc(h.get("standard_reference", "n/a"))}</div>'
            f'<div class="remedy"><b>Remedy:</b> {esc(h.get("remedy", "n/a"))}</div></div>',
            unsafe_allow_html=True,
        )

    download_json("⬇️ Download JSON report", result, "safety_audit_report.json", key="safety_dl")
    st.caption(DISCLAIMER)


# --------------------------------------------------------------------------------------
# Tab 2: Cracks
# --------------------------------------------------------------------------------------
def render_crack_tab(api_key: str, model: str, demo_mode: bool) -> None:
    st.subheader("🧱 Crack Diagnostics & Structural Integrity")
    st.caption("Upload a concrete or wall surface photo to classify the defect and get repair guidance.")

    uploaded = st.file_uploader("Surface photo", type=["jpg", "jpeg", "png"], key="crack_upload")
    if not uploaded:
        st.info("Upload a JPG/PNG photo to begin. Include a coin or ruler for better width estimates.")
        return

    try:
        image = prepare_image(uploaded)
    except Exception:
        st.error("That file could not be read as an image. Try another JPG or PNG.")
        return

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.image(image, caption="Uploaded surface", use_container_width=True)
        run = st.button("🔬 Diagnose defect", type="primary", use_container_width=True, key="crack_run")

    if run:
        with st.spinner("Diagnosing structural defect..."):
            try:
                result = DEMO_CRACK if demo_mode else analyse_image(api_key, model, CRACK_PROMPT, image)
                st.session_state["crack_result"] = result
                st.session_state["crack_demo"] = demo_mode
            except Exception as exc:
                st.session_state.pop("crack_result", None)
                st.error(friendly_error(exc))

    result = st.session_state.get("crack_result")
    if not result:
        return

    with right:
        if st.session_state.get("crack_demo"):
            st.info("Showing **sample data** (demonstration mode).", icon="🧪")
        sev = str(result.get("severity", "Medium"))
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Defect type", str(result.get("defect_type", "n/a")),
                                f"Confidence {as_int(result.get('confidence'))}%"), unsafe_allow_html=True)
        c2.markdown(metric_card("Crack width", f"{as_float(result.get('estimated_crack_width_mm')):.1f} mm",
                                "Estimated"), unsafe_allow_html=True)
        c3.markdown(metric_card("Urgency", str(result.get("urgency", "n/a")), ""), unsafe_allow_html=True)
        st.markdown(f"Severity: {pill(sev, severity_color(sev))}", unsafe_allow_html=True)
        if result.get("crack_pattern"):
            st.markdown(f"**Pattern:** {result['crack_pattern']}")
        if result.get("summary"):
            st.write(result["summary"])

        width = as_float(result.get("estimated_crack_width_mm"))
        if width >= 1.5 or sev in ("High", "Critical"):
            st.error("Width/severity suggests possible structural distress. Consult a licensed structural engineer.",
                     icon="⚠️")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Probable root causes")
        for cause in result.get("root_causes") or []:
            st.markdown(f"- {cause}")
        if result.get("monitoring_plan"):
            st.markdown("#### Monitoring plan")
            st.write(result["monitoring_plan"])
    with b:
        st.markdown("#### Recommended engineering repairs")
        for rep in result.get("recommended_repairs") or []:
            with st.expander(str(rep.get("method", "Repair")), expanded=True):
                st.write(rep.get("details", ""))

    download_json("⬇️ Download JSON report", result, "crack_diagnostic_report.json", key="crack_dl")
    st.caption(DISCLAIMER)


# --------------------------------------------------------------------------------------
# Tab 3: Planning & valuation
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StructureProfile:
    cost_per_sqft: float      # construction cost, INR / sq ft (Tier 2 baseline)
    steel_kg_per_sqft: float
    cement_bags_per_sqft: float
    market_multiplier: float  # market value as multiple of construction cost


STRUCTURES: dict[str, StructureProfile] = {
    "Residential (RCC framed)":     StructureProfile(2000, 4.0, 0.42, 1.55),
    "Commercial Office":            StructureProfile(2800, 5.0, 0.48, 1.75),
    "Industrial Warehouse (PEB)":   StructureProfile(1600, 6.0, 0.25, 1.35),
    "Luxury Villa":                 StructureProfile(3500, 4.5, 0.45, 1.85),
    "Load-bearing Masonry":         StructureProfile(1500, 2.0, 0.40, 1.40),
}
TIERS = {
    "Tier 1 (Mumbai, Delhi, Bengaluru...)": {"cost": 1.25, "market": 1.9},
    "Tier 2 (Indore, Pune, Jaipur...)":     {"cost": 1.00, "market": 1.0},
    "Tier 3 (Smaller cities & towns)":      {"cost": 0.82, "market": 0.65},
}
MILESTONES = [
    ("Site prep & foundation", 15),
    ("Plinth & ground beams", 25),
    ("RCC superstructure", 55),
    ("Masonry & services rough-in", 70),
    ("Plaster & finishing", 90),
    ("Handover & snagging", 100),
]


def build_render_prompt(structure: str, tier: str, sqft: int, floors: int, style: str) -> str:
    return (
        f"Architectural 3D render of a {style.lower()} {structure.lower()} building, {floors} floor(s), "
        f"about {sqft} sq ft built-up, Indian {tier.split(' ')[0]} city context, "
        "photorealistic exterior, golden hour lighting, landscaped surroundings, "
        "wide-angle architectural visualisation, highly detailed"
    )


def pollinations_url(prompt: str, seed: int = 7) -> str:
    return (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(prompt)
        + f"?width=1024&height=640&nologo=true&seed={seed}"
    )


def render_planning_tab() -> None:
    st.subheader("💰 Planning & Real Estate Valuation")
    st.caption("Estimate budget, spend-to-date, asset value and material needs. Rates are indicative 2026 benchmarks.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        structure = st.selectbox("Structure type", list(STRUCTURES), key="plan_structure")
        sqft = st.number_input("Built-up area (sq ft)", min_value=200, max_value=500_000, value=2400, step=100,
                               key="plan_sqft")
        floors = st.slider("Number of floors", 1, 20, 2, key="plan_floors")
    with c2:
        tier = st.selectbox("Location tier", list(TIERS), index=1, key="plan_tier")
        completion = st.slider("Completion (%)", 0, 100, 45, key="plan_completion")
        contingency = st.slider("Contingency (%)", 0, 20, 8, key="plan_contingency")

    prof = STRUCTURES[structure]
    tier_f = TIERS[tier]
    base_cost = prof.cost_per_sqft * tier_f["cost"] * sqft
    total_budget = base_cost * (1 + contingency / 100)
    capex = total_budget * completion / 100
    market_full = base_cost * prof.market_multiplier * (tier_f["market"] / tier_f["cost"])
    # Asset value: market value of work in place (land excluded), weighted slightly below linear for early stages
    realised = (completion / 100) ** 1.1
    current_valuation = market_full * realised
    steel_tonnes = prof.steel_kg_per_sqft * sqft / 1000
    cement_bags = prof.cement_bags_per_sqft * sqft

    m = st.columns(4)
    m[0].markdown(metric_card("Total project budget", format_inr(total_budget),
                              f"Incl. {contingency}% contingency"), unsafe_allow_html=True)
    m[1].markdown(metric_card("Capex incurred", format_inr(capex), f"{completion}% of budget"),
                  unsafe_allow_html=True)
    m[2].markdown(metric_card("Current asset valuation", format_inr(current_valuation),
                              f"Completed value: {format_inr(market_full)}"), unsafe_allow_html=True)
    m[3].markdown(metric_card("Remaining spend", format_inr(total_budget - capex), "To complete"),
                  unsafe_allow_html=True)

    st.markdown("")
    m2 = st.columns(4)
    m2[0].markdown(metric_card("Steel requirement", f"{steel_tonnes:,.1f} t", f"{prof.steel_kg_per_sqft} kg/sq ft"),
                   unsafe_allow_html=True)
    m2[1].markdown(metric_card("Cement requirement", f"{cement_bags:,.0f} bags", "50 kg bags"),
                   unsafe_allow_html=True)
    m2[2].markdown(metric_card("Steel consumed", f"{steel_tonnes * min(completion / 70, 1):,.1f} t",
                               "Approx. by progress"), unsafe_allow_html=True)
    m2[3].markdown(metric_card("Cement consumed", f"{cement_bags * min(completion / 80, 1):,.0f} bags",
                               "Approx. by progress"), unsafe_allow_html=True)

    st.markdown("#### Milestone progress")
    st.progress(completion / 100, text=f"{completion}% complete")
    cols = st.columns(len(MILESTONES))
    for col, (name, threshold) in zip(cols, MILESTONES):
        done = completion >= threshold
        prev = MILESTONES[MILESTONES.index((name, threshold)) - 1][1] if threshold != MILESTONES[0][1] else 0
        in_progress = not done and completion > prev
        label, color = ("Done", "green") if done else ("In progress", "yellow") if in_progress else ("Pending", "grey")
        col.markdown(f'<div class="metric-card"><div class="label">{esc(name)}</div>'
                     f'<div style="margin-top:6px">{pill(label, color)}</div></div>', unsafe_allow_html=True)

    st.markdown("#### 🖼️ Conceptual 3D architectural rendering")
    style = st.selectbox("Architectural style", ["Modern", "Contemporary Indian", "Minimalist", "Neo-classical",
                                                  "Industrial"], key="plan_style")
    default_prompt = build_render_prompt(structure, tier, int(sqft), floors, style)
    prompt = st.text_area("Render prompt (editable)", value=default_prompt, height=90, key=f"plan_prompt_{style}_{structure}")
    seed = st.number_input("Seed", 0, 99999, 7, key="plan_seed")
    if st.button("🎨 Generate concept render", key="plan_render"):
        st.session_state["render_url"] = pollinations_url(prompt, int(seed))
    if st.session_state.get("render_url"):
        st.image(st.session_state["render_url"], caption="Concept render via Pollinations AI (illustrative only)",
                 use_container_width=True)
        st.caption(f"[Open image in new tab]({st.session_state['render_url']})")
    st.caption("Estimates are indicative and exclude land, approvals, GST and site-specific factors. "
               "Obtain a detailed BOQ from a quantity surveyor before committing funds.")


# --------------------------------------------------------------------------------------
# Tab 4: Daily site assistant
# --------------------------------------------------------------------------------------
WEATHER_ADVISORIES: dict[str, dict[str, Any]] = {
    "High Wind": {
        "icon": "💨", "level": "red", "headline": "High wind: elevated lifting and falling-object risk",
        "alerts": [
            ("Suspend crane and hoist operations", "Stop tower/mobile crane lifts when sustained winds exceed ~36 km/h (manufacturer limit applies)."),
            ("Secure loose materials", "Tie down sheeting, formwork panels, scaffolding boards and bagged cement."),
            ("Restrict work at height", "Halt work on open edges, scaffolds and roofs; check scaffold ties."),
            ("Barricade drop zones", "Extend exclusion zones below elevated work and façades."),
        ],
    },
    "Heavy Rain": {
        "icon": "🌧️", "level": "amber", "headline": "Heavy rain: electrical, excavation and slip hazards",
        "alerts": [
            ("Electrical hazard mitigation", "De-energise temporary supply in wet areas, use RCCB/ELCB protected circuits, keep cables off the ground."),
            ("Excavation and trench safety", "Inspect shoring and slopes; pump out water; keep personnel out of unsupported trenches."),
            ("Protect fresh concrete", "Cover freshly placed concrete with tarpaulin; avoid pouring during rainfall."),
            ("Slip and drainage control", "Clear drains, lay anti-skid walkways, stockpile cement on raised, covered platforms."),
        ],
    },
    "Extreme Heat": {
        "icon": "🔥", "level": "red", "headline": "Extreme heat: worker heat stress and concrete curing risk",
        "alerts": [
            ("Concrete curing protocol", "Pour early morning or evening, use chilled water/ice in the mix, start curing immediately and keep moist 10-14 days (IS 456 Cl. 13.5)."),
            ("Heat-stress management", "Mandatory water and ORS, shaded rest every hour, rotate heavy tasks, watch for heat-exhaustion signs."),
            ("Hot-metal precautions", "Cover exposed steel and scaffold tubes; use gloves when handling."),
            ("Adjust working hours", "Shift heavy work outside 12:00-16:00 where practical."),
        ],
    },
    "Normal": {
        "icon": "☀️", "level": "green", "headline": "Normal conditions: standard site protocols apply",
        "alerts": [
            ("Morning toolbox talk", "Brief the crew on the day's activities, hazards and emergency points."),
            ("Daily equipment checks", "Inspect scaffolds, hoists, harnesses, ladders and electrical leads."),
            ("PPE enforcement", "Helmet, vest and boots at all times; harness for work above 1.8 m."),
            ("Housekeeping", "Keep walkways clear and segregate waste at day end."),
        ],
    },
}


def demo_chat_reply(question: str) -> str:
    q = question.lower()
    for key in ("curing", "slump", "cement", "steel"):
        if key in q:
            return DEMO_CHAT[key]
    return DEMO_CHAT["default"]


def render_assistant_tab(api_key: str, model: str, demo_mode: bool) -> None:
    st.subheader("🌦️ Daily Site Assistant & Weather-Aware Advisory")

    weather = st.radio("Today's site weather", list(WEATHER_ADVISORIES), horizontal=True, key="weather_choice")
    adv = WEATHER_ADVISORIES[weather]
    st.markdown(
        f'<div class="alert-box alert-{adv["level"]}"><b>{adv["icon"]} {esc(weather)}</b>: {esc(adv["headline"])}</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (title, detail) in enumerate(adv["alerts"]):
        cols[i % 2].markdown(
            f'<div class="alert-box alert-blue"><b>{esc(title)}</b><br>'
            f'<span class="small-note">{esc(detail)}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 💬 Construction engineering assistant")
    if demo_mode:
        st.caption("Demonstration mode: limited canned answers (try: curing, slump, cement, steel).")

    if "chat" not in st.session_state:
        st.session_state["chat"] = [{
            "role": "assistant",
            "content": "Hi, I'm SiteMate. Ask me about IS codes, mix design, quantities, safety, scheduling or "
                       "today's weather precautions.",
        }]

    head_l, head_r = st.columns([5, 1])
    with head_r:
        if st.button("🗑️ Clear chat", key="chat_clear"):
            st.session_state.pop("chat", None)
            st.rerun()

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a construction question...")
    if prompt:
        st.session_state["chat"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if demo_mode:
                        reply = demo_chat_reply(prompt)
                    else:
                        history = [m for m in st.session_state["chat"][1:]]  # drop greeting
                        reply = chat_reply(api_key, model, history, f"Today's weather condition: {weather}.")
                except Exception as exc:
                    reply = f"⚠️ {friendly_error(exc)}"
            st.markdown(reply)
        st.session_state["chat"].append({"role": "assistant", "content": reply})


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    api_key, model, demo_mode = render_sidebar()

    st.markdown(
        '<div class="hero"><h1>🏗️ Multimodal Construction Intelligence Platform</h1>'
        "<p>AI-powered safety audits, structural diagnostics, project valuation and a weather-aware site assistant.</p></div>",
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs([
        "🦺 Safety & PPE Audit",
        "🧱 Crack Diagnostics",
        "💰 Planning & Valuation",
        "🌦️ Site Assistant",
    ])
    with t1:
        render_safety_tab(api_key, model, demo_mode)
    with t2:
        render_crack_tab(api_key, model, demo_mode)
    with t3:
        render_planning_tab()
    with t4:
        render_assistant_tab(api_key, model, demo_mode)


if __name__ == "__main__":
    main()
