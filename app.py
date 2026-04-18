import streamlit as st
import cv2
import json
import tempfile
import pandas as pd
import numpy as np

from core.pipeline import run_pipeline
from behaviour.tracker import ObjectTracker
from behaviour.tailgating import TailgatingDetector
from behaviour.lead_vehicle import select_lead_vehicle
from behaviour.lane_departure import LaneDepartureDetector
from behaviour.sign_violation import SignViolationDetector


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="DRIV3X",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- THEME STATE --------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

is_dark = st.session_state.theme == "dark"

# ── Theme variables ──
if is_dark:
    theme_vars = """
        --bg-deep:      #090c10;
        --bg-card:      #0f1318;
        --bg-card2:     #141920;
        --border:       #1e2730;
        --accent:       #00d4ff;
        --accent-glow:  rgba(0, 212, 255, 0.18);
        --text-primary: #e8edf2;
        --text-muted:   #5a6a7a;
        --success-bg:   #0f1a13;
        --success-bdr:  #1a3020;
        --tg-bg:        #2a1015;
        --ld-bg:        #2a1e0a;
        --sv-bg:        #0a1e2a;
        --empty-bg:     #0f1318;
        --empty-bdr:    #1e2730;
        --empty-fg:     #5a6a7a;
    """
else:
    theme_vars = """
        --bg-deep:      #f0f4f8;
        --bg-card:      #ffffff;
        --bg-card2:     #e8edf4;
        --border:       #cbd5e1;
        --accent:       #0077ff;
        --accent-glow:  rgba(0, 119, 255, 0.12);
        --text-primary: #0d1117;
        --text-muted:   #64748b;
        --success-bg:   #f0faf4;
        --success-bdr:  #a7f3c0;
        --tg-bg:        #fff1f2;
        --ld-bg:        #fffbeb;
        --sv-bg:        #eff6ff;
        --empty-bg:     #ffffff;
        --empty-bdr:    #cbd5e1;
        --empty-fg:     #94a3b8;
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');

:root,
html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlocksContainer"],
[data-testid="block-container"],
.main {{
    {theme_vars}
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{
    background-color: var(--bg-deep) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary) !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding: 2rem 3rem 4rem 3rem !important;
    max-width: 1200px;
}}

.driv3x-hero {{
    display: flex;
    align-items: center;
    padding: 2.5rem 0 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}}
.driv3x-logo {{
    font-family: 'Orbitron', sans-serif;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: 0.15em;
    color: var(--text-primary);
    line-height: 1;
    text-shadow: 0 0 24px rgba(0, 212, 255, 0.25);
}}
.driv3x-logo span {{ color: var(--accent); }}
.driv3x-tagline {{
    font-size: 0.8rem;
    font-weight: 300;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 0.25rem;
}}
.status-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: pulse 2s infinite;
    margin-right: 0.5rem;
    display: inline-block;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.35; }}
}}

.carousel-viewport {{ overflow: hidden; width: 100%; position: relative; margin-bottom: 1rem; }}
.carousel-track    {{ display: flex; width: 400%; animation: slide-loop 16s linear infinite; }}
.c-card            {{ width: 25%; flex-shrink: 0; box-sizing: border-box; padding: 0 0.5rem; }}
.c-inner {{
    background: linear-gradient(135deg, var(--accent) 0%, #0050cc 100%);
    border: none;
    border-radius: 8px;
    padding: 1.8rem 1.6rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 4px 24px rgba(0, 180, 255, 0.18);
}}
.c-inner strong {{
    color: #ffffff;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.6rem;
    opacity: 0.85;
}}
.c-inner p {{ color: rgba(255,255,255,0.92); font-size: 0.88rem; line-height: 1.65; margin: 0; }}

.carousel-dots {{ display: flex; justify-content: center; gap: 0.5rem; margin-top: 0.75rem; margin-bottom: 2rem; }}
.dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--border); animation: dot-cycle 16s linear infinite; }}
.dot:nth-child(2) {{ animation-delay: -12s; }}
.dot:nth-child(3) {{ animation-delay: -8s; }}
.dot:nth-child(4) {{ animation-delay: -4s; }}
@keyframes dot-cycle {{
    0%, 24.9% {{ background: var(--accent); box-shadow: 0 0 6px var(--accent); }}
    25%, 100% {{ background: var(--border); box-shadow: none; }}
}}
@keyframes slide-loop {{
    0%   {{ transform: translateX(0%);   }} 20%  {{ transform: translateX(0%);   }}
    25%  {{ transform: translateX(-25%); }} 45%  {{ transform: translateX(-25%); }}
    50%  {{ transform: translateX(-50%); }} 70%  {{ transform: translateX(-50%); }}
    75%  {{ transform: translateX(-75%); }} 95%  {{ transform: translateX(-75%); }}
    100% {{ transform: translateX(0%);   }}
}}

.upload-label {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}}
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileDropzoneInstructions"],
[data-testid="stFileUploaderDropzone"] > div {{
    background-color: var(--bg-card) !important;
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
}}
[data-testid="stFileUploader"]:hover {{ border-color: var(--accent) !important; }}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileDropzoneInstructions"] span {{ color: var(--text-muted) !important; }}

.stButton > button {{
    background: transparent !important;
    border: 1.5px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 0.55rem 2.5rem !important;
    transition: background 0.2s, box-shadow 0.2s !important;
}}
.stButton > button:hover {{
    background: var(--accent-glow) !important;
    box-shadow: 0 0 18px var(--accent-glow) !important;
}}

.section-heading {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 2.5rem 0 1.2rem 0;
}}

.score-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 2rem 2.5rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    margin-bottom: 1.5rem;
}}
.score-number {{ font-family: 'Rajdhani', sans-serif; font-size: 4.5rem; font-weight: 700; line-height: 1; }}
.score-label  {{ font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase; margin-top: 0.3rem; }}
.score-bar-wrap {{ flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }}
.score-bar-fill {{ height: 100%; border-radius: 3px; }}

.event-tile {{ background: var(--bg-card); border-radius: 8px; padding: 1.2rem 1rem; text-align: center; border-width: 1px; border-style: solid; }}
.event-tile-count {{ font-family: 'Rajdhani', sans-serif; font-size: 2.5rem; font-weight: 700; line-height: 1; }}
.event-tile-label {{ font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.4rem; opacity: 0.75; }}

.complete-banner {{
    display: flex; align-items: center; gap: 0.6rem;
    margin: 1rem 0;
    background: var(--success-bg);
    border: 1px solid var(--success-bdr);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    color: #00e676;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}}

.empty-state {{
    background: var(--empty-bg);
    border: 1px solid var(--empty-bdr);
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    color: var(--empty-fg);
    font-size: 0.85rem;
    letter-spacing: 0.05em;
}}

.processing-note {{ color: var(--text-muted); font-size: 0.82rem; letter-spacing: 0.05em; }}

[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"],
[data-testid="stExpander"],
[data-testid="stExpanderContent"] {{
    background-color: transparent !important;
    color: var(--text-primary) !important;
}}

[data-testid="stAlert"]                    {{ border-radius: 6px !important; font-size: 0.85rem; }}
[data-testid="stAlert"] > div             {{ background: var(--bg-card2) !important; }}
[data-testid="stProgressBar"] > div > div {{ background: var(--accent) !important; }}
[data-testid="stSpinner"] p               {{ color: var(--text-muted) !important; }}
[data-testid="stSpinner"] > div           {{ border-top-color: var(--accent) !important; }}
hr                                         {{ border-color: var(--border) !important; }}

.stDownloadButton > button {{
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    border-radius: 4px !important;
}}
.stDownloadButton > button:hover {{
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}}
video {{ border-radius: 8px !important; border: 1px solid var(--border) !important; }}
</style>
""", unsafe_allow_html=True)


# -------------------- THEME TOGGLE + HERO --------------------
col_logo, col_toggle = st.columns([0.92, 0.08])
with col_toggle:
    st.button("☀️" if is_dark else "🌙", on_click=toggle_theme, help="Switch theme")

st.markdown("""
<div class="driv3x-hero">
    <div>
        <div class="driv3x-logo">DRIV<span>3</span>X</div>
        <div class="driv3x-tagline"><span class="status-dot"></span>Driver Behaviour Monitoring &amp; Analysis System</div>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------- OBJECTIVES CAROUSEL --------------------
st.markdown("""
<div class="carousel-viewport">
    <div class="carousel-track">
        <div class="c-card"><div class="c-inner">
            <strong>01 — Detection</strong>
            <p>Detect vehicles, pedestrians, and cyclists from dashcam video feeds in real time.</p>
        </div></div>
        <div class="c-card"><div class="c-inner">
            <strong>02 — Behaviour Monitoring</strong>
            <p>Identify potential risks based on object distance, reaction time, and environmental factors.</p>
        </div></div>
        <div class="c-card"><div class="c-inner">
            <strong>03 — Risk Reporting</strong>
            <p>Generate a post-drive risk score report summarising driver performance and safety level.</p>
        </div></div>
        <div class="c-card"><div class="c-inner">
            <strong>04 — Insight</strong>
            <p>Assist new drivers in improving skills and supply reliable data for insurance analysis.</p>
        </div></div>
    </div>
</div>
<div class="carousel-dots">
    <div class="dot"></div><div class="dot"></div>
    <div class="dot"></div><div class="dot"></div>
</div>
""", unsafe_allow_html=True)


# -------------------- VIDEO UPLOAD --------------------
st.markdown('<div class="upload-label">Upload Driving Video</div>', unsafe_allow_html=True)

uploaded_video = st.file_uploader("", type=["mp4", "avi", "mov"], label_visibility="collapsed")

if uploaded_video is None:
    # Clear stored results when no video is uploaded
    st.session_state.pop("analysis_events", None)
    st.session_state.pop("analysis_json", None)
    st.markdown("""
    <div style="color:var(--text-muted); font-size:0.82rem; text-align:center; padding:0.5rem 0;">
        Accepted formats: MP4 · AVI · MOV
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Use a key based on the file name+size to detect when a new file is uploaded
file_key = f"{uploaded_video.name}_{uploaded_video.size}"
if st.session_state.get("last_file_key") != file_key:
    # New file uploaded — clear previous results
    st.session_state.pop("analysis_events", None)
    st.session_state.pop("analysis_json", None)
    st.session_state["last_file_key"] = file_key

with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
    tmp.write(uploaded_video.read())
    video_path = tmp.name

st.video(video_path)


# -------------------- DRIVING SCORE --------------------
def calculate_driving_score(events):
    base_score = 100
    severity_penalty = {"low": 1.0, "medium": 1.5, "high": 2.0}
    max_penalty = {"lane_departure": 15, "tailgating": 20, "sign_violation": 25}
    penalty_tracker = {"lane_departure": 0.0, "tailgating": 0.0, "sign_violation": 0.0}

    for e in events:
        event_type = e.get("event")
        severity = str(e.get("severity", "low")).lower()
        if event_type in penalty_tracker:
            penalty_tracker[event_type] += severity_penalty.get(severity, 0.5)

    total_penalty = sum(min(penalty_tracker[k], max_penalty[k]) for k in penalty_tracker)
    return round(max(base_score - total_penalty, 0), 1)


# -------------------- ANALYZE --------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("▶  RUN ANALYSIS"):
    with st.spinner("Initialising pipeline…"):
        pass

    st.markdown('<p class="processing-note">Processing frames — this may take a moment.</p>', unsafe_allow_html=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    tracker        = ObjectTracker()
    tailgating     = TailgatingDetector()
    lane_departure = LaneDepartureDetector(offset_threshold=50)
    sign_violation = SignViolationDetector()

    frame_idx    = 0
    all_events   = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress     = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        h, w = frame.shape[:2]

        output          = run_pipeline(frame)
        tracked_objects = tracker.update(output["objects"])
        lead_vehicle    = select_lead_vehicle(tracked_objects, output["lanes"])

        tg_event = tailgating.update(lead_vehicle, h, timestamp)
        if tg_event:
            all_events.append(tg_event)

        ld_event = lane_departure.update(output["lanes"]["offset_px"], timestamp)
        if ld_event:
            ld_event["offset_px"] = int(ld_event["offset_px"])
            all_events.append(ld_event)

        signs = output.get("signs", [])
        if signs:
            for s in signs:
                x1, y1, x2, y2 = s["bbox"]
                ratio = ((x2 - x1) * (y2 - y1)) / (h * w)
                sign_violation.update(s["class"], ratio, s["confidence"])
        else:
            sign_violation.update("none", 0, 0)

        sign_violation.update_states()
        sv_event = sign_violation.check(0, timestamp)
        if sv_event:
            all_events.append(sv_event)

        frame_idx += 1
        progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()

    def make_json_safe(obj):
        if isinstance(obj, dict):   return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [make_json_safe(v) for v in obj]
        elif isinstance(obj, np.integer):  return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        return obj

    all_events = make_json_safe(all_events)

    # ── FIX: Store results in session_state so they survive reruns ──
    st.session_state["analysis_events"] = all_events
    st.session_state["analysis_json"]   = json.dumps(all_events, indent=2)


# -------------------- RESULTS (rendered from session_state) --------------------
# This block runs on every rerun (theme toggle, download click, etc.)
if "analysis_events" in st.session_state:
    all_events  = st.session_state["analysis_events"]
    report_json = st.session_state["analysis_json"]

    st.markdown('<div class="complete-banner">✓ &nbsp; Analysis complete</div>', unsafe_allow_html=True)

    if all_events:
        df            = pd.DataFrame(all_events)
        driving_score = calculate_driving_score(all_events)

        if   driving_score >= 90: score_color = "#00e676"; tier = "EXCELLENT"
        elif driving_score >= 75: score_color = "#00d4ff"; tier = "GOOD"
        elif driving_score >= 60: score_color = "#ffab40"; tier = "MODERATE RISK"
        else:                     score_color = "#ff5252"; tier = "HIGH RISK"

        # Score card
        st.markdown('<div class="section-heading">Safety Score</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="score-card">
            <div>
                <div class="score-number" style="color:{score_color};">{driving_score}</div>
                <div class="score-label"  style="color:{score_color};">/ 100 &nbsp;·&nbsp; {tier}</div>
            </div>
            <div class="score-bar-wrap">
                <div class="score-bar-fill" style="width:{driving_score}%; background:{score_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Event breakdown tiles
        st.markdown('<div class="section-heading">Event Breakdown</div>', unsafe_allow_html=True)

        event_meta = {
            "tailgating":     {"color": "#ff5252", "bg_var": "--tg-bg"},
            "lane_departure": {"color": "#ffab40", "bg_var": "--ld-bg"},
            "sign_violation": {"color": "#00d4ff", "bg_var": "--sv-bg"},
        }

        counts = df["event"].value_counts()
        cols = st.columns(len(counts))
        for col, (evt, cnt) in zip(cols, counts.items()):
            meta  = event_meta.get(evt, {"color": "#e8edf2", "bg_var": "--bg-card"})
            color = meta["color"]
            bg    = meta["bg_var"]
            with col:
                st.markdown(f"""
                <div class="event-tile" style="background:var({bg}); border-color:{color}33;">
                    <div class="event-tile-count" style="color:{color};">{cnt}</div>
                    <div class="event-tile-label" style="color:{color};">{evt.replace("_"," ")}</div>
                </div>
                """, unsafe_allow_html=True)

        # Detailed event log
        st.markdown('<div class="section-heading">Detailed Event Log</div>', unsafe_allow_html=True)

        ev_style = {
            "tailgating":     {"accent": "#ff5252", "accent_dark": "#c0392b", "accent_alpha": "rgba(255,82,82,0.18)",  "bar": "rgba(255,82,82,0.55)"},
            "lane_departure": {"accent": "#ffab40", "accent_dark": "#b7600a", "accent_alpha": "rgba(255,171,64,0.18)", "bar": "rgba(255,171,64,0.55)"},
            "sign_violation": {"accent": "#00d4ff", "accent_dark": "#0077b6", "accent_alpha": "rgba(0,212,255,0.18)",  "bar": "rgba(0,212,255,0.55)"},
        }
        sev_style = {
            "low":    {"color": "#00e676", "dark": "#027a48"},
            "medium": {"color": "#ffab40", "dark": "#b7600a"},
            "high":   {"color": "#ff5252", "dark": "#c0392b"},
        }

        # ── Normalise timestamp: detectors may emit "time", "ts", or "timestamp" ──
        # Rename whichever variant exists to "timestamp" so the column is always present
        for ts_alias in ("time", "ts", "t"):
            if ts_alias in df.columns and "timestamp" not in df.columns:
                df.rename(columns={ts_alias: "timestamp"}, inplace=True)
                break
        # If still missing, add a blank column so it always appears in the table
        if "timestamp" not in df.columns:
            df["timestamp"] = float("nan")

        # ── Always show these columns; hide offset/confidence if absent ──
        display_cols = ["event", "severity", "timestamp"]
        if "offset_px"   in df.columns: display_cols.append("offset_px")
        if "confidence"  in df.columns: display_cols.append("confidence")

        col_labels = {
            "event":      "Event",
            "severity":   "Severity",
            "timestamp":  "Timestamp",
            "offset_px":  "Offset",
            "confidence": "Confidence",
        }

        # Debug expander — shows raw keys so you can verify detector output
        with st.expander("🔍 Raw event keys (debug)", expanded=False):
            st.write("Columns detected:", list(df.columns))
            st.json(all_events[:3] if all_events else [])

        header_cells = '<th class="evt-th" style="width:3px;padding:0;"></th>'
        header_cells += '<th class="evt-th">#</th>'
        header_cells += "".join(f'<th class="evt-th">{col_labels.get(c, c.replace("_"," ").title())}</th>' for c in display_cols)

        row_html = ""
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            evt  = str(row.get("event", ""))
            sev  = str(row.get("severity", "low")).lower()
            es   = ev_style.get(evt, {"accent": "#888", "accent_dark": "#555", "accent_alpha": "rgba(128,128,128,0.12)", "bar": "rgba(128,128,128,0.4)"})
            ss   = sev_style.get(sev, {"color": "#888", "dark": "#555"})

            cells  = f'<td style="padding:0;width:3px;background:{es["bar"]};"></td>'
            cells += f'<td class="evt-td evt-num" style="color:var(--text-muted);padding-right:0.5rem;">{i:02d}</td>'

            for c in display_cols:
                val = row.get(c, float("nan"))
                if c == "event":
                    label = str(val).replace("_", " ").title()
                    cells += (
                        f'<td class="evt-td">'
                        f'<span class="evt-badge" style="color:{es["accent_dark"]};background:{es["accent_alpha"]};border-color:{es["accent"]}44;">'
                        f'<span class="badge-pip" style="background:{es["accent"]};"></span>{label}'
                        f'</span></td>'
                    )
                elif c == "severity":
                    cells += (
                        f'<td class="evt-td">'
                        f'<span class="sev-chip" style="color:{ss["dark"]};">'
                        f'<span class="sev-pip" style="background:{ss["color"]};"></span>{str(val).title()}'
                        f'</span></td>'
                    )
                elif c == "timestamp":
                    try:
                        fval = float(val)
                        if pd.isna(fval):
                            raise ValueError
                        cells += f'<td class="evt-td evt-num" style="color:var(--text-primary);">{fval:.2f}s</td>'
                    except Exception:
                        cells += f'<td class="evt-td evt-num" style="color:var(--text-muted);">—</td>'
                elif c == "offset_px":
                    try:
                        v = int(val)
                        sign = "+" if v > 0 else ""
                        cells += f'<td class="evt-td evt-num">{sign}{v}px</td>'
                    except Exception:
                        cells += f'<td class="evt-td evt-num" style="color:var(--text-muted);">—</td>'
                elif c == "confidence":
                    try:
                        fval = float(val)
                        if pd.isna(fval):
                            raise ValueError
                        cells += f'<td class="evt-td evt-num">{fval:.2f}</td>'
                    except Exception:
                        # nan confidence → show dash, not "nan"
                        cells += f'<td class="evt-td evt-num" style="color:var(--text-muted);">—</td>'
                else:
                    cells += f'<td class="evt-td">{val}</td>'

            row_html += f'<tr class="evt-row">{cells}</tr>'

        st.markdown(f"""
        <style>
        .ev-wrap {{
            width:100%; overflow-x:auto;
            border:1px solid var(--border);
            border-radius:10px;
            background:var(--bg-card);
            margin-bottom:1.5rem;
        }}
        .ev-table {{
            width:100%; border-collapse:collapse;
            font-size:0.81rem; font-family:'Inter',sans-serif;
        }}
        .ev-table thead tr {{
            background:var(--bg-card2);
            border-bottom:1px solid var(--border);
        }}
        .evt-th {{
            padding:0.7rem 0.9rem;
            text-align:left;
            font-family:'Rajdhani',sans-serif;
            font-size:0.68rem; font-weight:700;
            letter-spacing:0.18em; text-transform:uppercase;
            color:var(--text-muted); white-space:nowrap;
        }}
        .evt-row {{
            border-bottom:1px solid var(--border);
            transition:background 0.12s;
        }}
        .evt-row:last-child {{ border-bottom:none; }}
        .evt-row:hover {{ background:var(--bg-card2); }}
        .evt-td {{
            padding:0.6rem 0.9rem;
            color:var(--text-primary);
            vertical-align:middle; white-space:nowrap;
        }}
        .evt-num {{
            font-family:'Courier New',monospace;
            font-size:0.76rem;
            color:var(--text-primary);
            letter-spacing:0.03em;
        }}
        .evt-badge {{
            display:inline-flex; align-items:center; gap:5px;
            padding:0.18rem 0.6rem;
            border-radius:4px; border:1px solid;
            font-family:'Rajdhani',sans-serif;
            font-size:0.7rem; font-weight:700;
            letter-spacing:0.1em; text-transform:uppercase;
        }}
        .badge-pip {{
            width:5px; height:5px;
            border-radius:50%; flex-shrink:0;
        }}
        .sev-chip {{
            display:inline-flex; align-items:center; gap:5px;
            font-size:0.76rem; font-weight:500;
            letter-spacing:0.05em;
        }}
        .sev-pip {{
            width:6px; height:6px;
            border-radius:50%; flex-shrink:0;
        }}
        </style>
        <div class="ev-wrap">
          <table class="ev-table">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{row_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

        # Download — reads from session_state, not the file
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "↓  Download Full Report (JSON)",
            report_json,
            "driv3x_report.json",
            "application/json"
        )

    else:
        st.markdown(
            '<div class="empty-state">No risky driving events detected in this session.</div>',
            unsafe_allow_html=True
        )