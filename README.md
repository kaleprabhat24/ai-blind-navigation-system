"""
VisionGuide AI — Intelligent Navigation Assistant
Gradio + gTTS | YOLOv8 | Person & Chair Detection
No system dependencies required for voice.
"""

import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import time
import os
import tempfile

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH        = "best.pt"
CONF_THRESHOLD    = 0.40
FOCAL_LEN         = 700
KNOWN_WIDTHS      = {0: 0.50, 56: 0.55}   # person, chair widths in metres
ZONE_SPLITS       = (1/3, 2/3)
DETECTION_CLASSES = [0, 56]                # 0=person, 56=chair
VOICE_COOLDOWN    = 6.0                    # seconds between YOLO + voice runs

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (once at startup)
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"[VisionGuide] Loaded custom model: {MODEL_PATH}")
else:
    print(f"[VisionGuide] '{MODEL_PATH}' not found — using yolov8n.pt fallback")
    model = YOLO("yolov8n.pt")

# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
def estimate_distance(box, class_id: int) -> float:
    x1, y1, x2, y2 = box
    pixel_width = max(float(x2 - x1), 1.0)
    real_width   = KNOWN_WIDTHS.get(int(class_id), 0.50)
    dist = (real_width * FOCAL_LEN) / pixel_width
    return float(np.clip(dist, 0.3, 15.0))

# ─────────────────────────────────────────────────────────────────────────────
# ZONE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def get_zone(cx: float, width: int) -> str:
    rel = cx / float(width)
    if rel < ZONE_SPLITS[0]: return "LEFT"
    if rel > ZONE_SPLITS[1]: return "RIGHT"
    return "CENTER"

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION — human, accurate, for blind users
# ─────────────────────────────────────────────────────────────────────────────
def build_navigation(detections: list) -> tuple:
    """Returns (display_text, voice_text)."""
    if not detections:
        return (
            "✅ Path is completely clear. You can walk forward safely.",
            "Path is clear. Walk forward."
        )

    dets = sorted(detections, key=lambda d: d["dist"])
    by_zone = {"LEFT": [], "CENTER": [], "RIGHT": []}
    for d in dets:
        by_zone[d["zone"]].append(d)

    def closest(zone):
        return by_zone[zone][0]["dist"] if by_zone[zone] else 99.0

    cd_c = closest("CENTER")
    cd_l = closest("LEFT")
    cd_r = closest("RIGHT")
    nearest = dets[0]
    label   = nearest["label"].capitalize()
    dist_s  = f"{nearest['dist']:.1f} metres"

    # STOP — within 0.9 m
    if nearest["dist"] < 0.9:
        zone_phrase = {
            "CENTER": "directly in front of you",
            "LEFT":   "on your left side",
            "RIGHT":  "on your right side",
        }.get(nearest["zone"], "very close")
        nav   = (f"⛔ STOP IMMEDIATELY! {label} is {zone_phrase}, "
                 f"only {dist_s} away. Do not take another step forward.")
        voice = f"Stop! {label} {zone_phrase}, {dist_s}. Do not move."
        return nav, voice

    # Center blocked
    if by_zone["CENTER"] and cd_c < 3.5:
        blk       = by_zone["CENTER"][0]
        blk_label = blk["label"].capitalize()
        blk_dist  = f"{blk['dist']:.1f} metres"
        left_safe  = not by_zone["LEFT"]  or cd_l > 2.5
        right_safe = not by_zone["RIGHT"] or cd_r > 2.5

        if left_safe and right_safe:
            side = "LEFT" if cd_l >= cd_r else "RIGHT"
            nav   = (f"⚠ {blk_label} is blocking ahead at {blk_dist}. "
                     f"Both sides are open. Move to your {side} — more space there. "
                     f"Walk around, then return to centre.")
            voice = (f"{blk_label} ahead at {blk_dist}. "
                     f"Move to your {side.lower()}. Both sides clear.")
        elif left_safe:
            nav   = (f"⚠ {blk_label} directly ahead at {blk_dist}. "
                     f"Right side is occupied. Move LEFT now and walk around.")
            voice = f"{blk_label} ahead at {blk_dist}. Right blocked. Move LEFT now."
        elif right_safe:
            nav   = (f"⚠ {blk_label} directly ahead at {blk_dist}. "
                     f"Left side is occupied. Move RIGHT now and walk around.")
            voice = f"{blk_label} ahead at {blk_dist}. Left blocked. Move RIGHT now."
        else:
            if nearest["dist"] > 2.0:
                side = "LEFT" if cd_l >= cd_r else "RIGHT"
                nav   = (f"⚠ Obstacles all around. Nearest is {blk_label} at {blk_dist}. "
                         f"Move slowly to your {side} — slightly more space there.")
                voice = (f"Obstacles all sides. {blk_label} at {blk_dist}. "
                         f"Move slowly {side.lower()}.")
            else:
                nav   = (f"🔴 Path blocked on all sides. {blk_label} at {blk_dist}. "
                         f"Stop and wait for path to clear.")
                voice = f"Stop. All blocked. {blk_label} at {blk_dist}. Wait."
        return nav, voice

    # Center open — side objects only
    side_notes = []
    if by_zone["LEFT"]:
        n = by_zone["LEFT"][0]
        side_notes.append(f"{n['label']} on your left at {n['dist']:.1f} metres")
    if by_zone["RIGHT"]:
        n = by_zone["RIGHT"][0]
        side_notes.append(f"{n['label']} on your right at {n['dist']:.1f} metres")

    if side_notes:
        note_str = " and ".join(side_notes)
        nav   = (f"✅ Centre path is open. Continue walking straight forward. "
                 f"Be aware: {note_str}. Stay in the middle.")
        voice = f"Centre clear. Walk forward. Be aware: {note_str}."
    else:
        nav   = "✅ Path is completely clear on all sides. Walk forward confidently."
        voice = "All clear. Walk forward."

    return nav, voice

# ─────────────────────────────────────────────────────────────────────────────
# DRAW ANNOTATIONS — stable, not flickering
# ─────────────────────────────────────────────────────────────────────────────
def draw_frame(frame_bgr: np.ndarray, detections: list) -> np.ndarray:
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    # Zone overlay
    z1, z2 = int(w * ZONE_SPLITS[0]), int(w * ZONE_SPLITS[1])
    overlay = vis.copy()
    cv2.rectangle(overlay, (z1, 0), (z2, h), (200, 200, 200), -1)
    vis = cv2.addWeighted(overlay, 0.04, vis, 0.96, 0)
    cv2.line(vis, (z1, 0), (z1, h), (60, 80, 110), 1)
    cv2.line(vis, (z2, 0), (z2, h), (60, 80, 110), 1)
    for txt, x in [("LEFT", 6), ("CENTER", z1 + 6), ("RIGHT", z2 + 6)]:
        cv2.putText(vis, txt, (x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 170), 1, cv2.LINE_AA)

    # Boxes
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        dist = d["dist"]

        if dist < 1.0:   col = (0, 0, 220)
        elif dist < 2.0: col = (0, 100, 255)
        elif dist < 3.5: col = (0, 200, 255)
        else:            col = (50, 200, 80)

        thickness = 3 if dist < 1.5 else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, thickness)

        tag = f"{d['label']}  {dist:.1f}m  [{d['zone']}]"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw + 10, y1), col, -1)
        cv2.putText(vis, tag, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 10, 10), 1, cv2.LINE_AA)

    return vis

# ─────────────────────────────────────────────────────────────────────────────
# gTTS VOICE
# ─────────────────────────────────────────────────────────────────────────────
def speak(text: str):
    try:
        tts = gTTS(text=text.strip(), lang="en", slow=False)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"[Voice Error] {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS — main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def process(frame, last_time, last_nav, last_voice_text, force_run=False):
    if frame is None:
        return None, "⚠ No video feed.", None, last_time, last_nav, last_voice_text

    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    # Convert RGB→BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
    h, w = frame_bgr.shape[:2]

    current_time = time.time()

    # THROTTLE: return last result visually without re-running YOLO
    if not force_run and (current_time - last_time) < VOICE_COOLDOWN:
        # Just return original frame (no boxes) to avoid flickering stale boxes
        return frame, last_nav, None, last_time, last_nav, last_voice_text

    # ── RUN YOLO ──
    results = model(frame_bgr, conf=CONF_THRESHOLD,
                    classes=DETECTION_CLASSES, verbose=False)

    detections = []
    if results and len(results[0].boxes) > 0:
        boxes   = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs   = results[0].boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            cx   = (x1 + x2) / 2.0
            dist = estimate_distance(box, int(cls))
            zone = get_zone(cx, w)
            name = model.names[int(cls)] if int(cls) < len(model.names) else f"obj{int(cls)}"
            detections.append({
                "label": name,
                "dist":  dist,
                "zone":  zone,
                "box":   (x1, y1, x2, y2),
                "conf":  float(conf),
            })

    nav_display, voice_text = build_navigation(detections)

    # Draw on frame
    vis_bgr = draw_frame(frame_bgr, detections)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    # Build info text
    if detections:
        lines = [nav_display, "", f"Detected {len(detections)} object(s):"]
        for d in sorted(detections, key=lambda x: x["dist"]):
            lines.append(
                f"  • {d['label'].capitalize()} — {d['dist']:.1f}m "
                f"— {d['zone']} zone  ({d['conf']:.0%} confidence)"
            )
        info_text = "\n".join(lines)
    else:
        info_text = nav_display

    # Voice: only generate if text changed
    audio_path = None
    if voice_text.strip() != last_voice_text.strip():
        audio_path = speak(voice_text)

    return vis_rgb, info_text, audio_path, current_time, nav_display, voice_text


def process_webcam(frame, last_time, last_nav, last_voice_text):
    return process(frame, last_time, last_nav, last_voice_text, force_run=False)


def process_upload(frame, last_nav, last_voice_text):
    return process(frame, 0.0, last_nav, last_voice_text, force_run=True)


# ─────────────────────────────────────────────────────────────────────────────
# CSS — Premium dark glassmorphism
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Space Grotesk', sans-serif !important; box-sizing: border-box; }

body, .gradio-container {
    background: #020408 !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 15% 0%,  rgba(56,189,248,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 100%, rgba(139,92,246,0.06) 0%, transparent 60%) !important;
    min-height: 100vh;
}

.gradio-container { max-width: 1380px !important; margin: 0 auto !important; padding: 20px !important; }

.header-wrap {
    text-align: center; padding: 28px 32px; margin-bottom: 22px;
    background: linear-gradient(135deg, rgba(56,189,248,0.06), rgba(139,92,246,0.04));
    border: 1px solid rgba(56,189,248,0.12); border-radius: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    position: relative; overflow: hidden;
}
.header-wrap::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.5), rgba(139,92,246,0.3), transparent);
}
.header-wrap h1 { font-size: 2.2rem; font-weight: 700; color: #f0f9ff; letter-spacing:-1px; margin:0; line-height:1; }
.header-wrap p  { color: #334155; font-size: 0.8rem; margin-top: 8px; letter-spacing: 0.5px; }

.sec-label {
    font-size: 0.58rem; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: #1e293b; margin-bottom: 10px;
}

/* Image panels */
.image-container img { border-radius: 14px !important; }

/* Textbox */
textarea {
    background: rgba(5,8,16,0.85) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    color: #94a3b8 !important;
    font-size: 0.84rem !important;
    line-height: 1.65 !important;
    resize: none !important;
}

label span, .label-wrap span {
    color: #1e293b !important; font-size: 0.6rem !important;
    font-weight: 700 !important; letter-spacing: 2px !important; text-transform: uppercase !important;
}

/* Tabs */
.tab-nav button {
    background: transparent !important; border: none !important;
    color: #334155 !important; font-weight: 600 !important;
    font-size: 0.8rem !important; border-radius: 8px !important;
    padding: 7px 16px !important; transition: all 0.2s !important;
}
.tab-nav button.selected {
    background: rgba(56,189,248,0.10) !important; color: #38bdf8 !important;
    border: 1px solid rgba(56,189,248,0.18) !important;
}

/* Button */
.btn-run {
    background: linear-gradient(135deg, rgba(56,189,248,0.14), rgba(14,165,233,0.09)) !important;
    border: 1px solid rgba(56,189,248,0.22) !important; color: #38bdf8 !important;
    font-weight: 600 !important; border-radius: 12px !important;
    padding: 10px 24px !important; transition: all 0.2s !important; width: 100% !important;
}
.btn-run:hover {
    background: linear-gradient(135deg, rgba(56,189,248,0.22), rgba(14,165,233,0.16)) !important;
    box-shadow: 0 0 20px rgba(56,189,248,0.12) !important;
}

/* Info chips */
.chips { display: flex; gap: 10px; margin-top: 18px; flex-wrap: wrap; }
.chip {
    flex: 1; min-width: 120px; text-align: center; padding: 12px 8px;
    background: rgba(10,15,28,0.7); border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px; font-size: 0.7rem; color: #334155;
}
.chip strong { color: #38bdf8; display: block; font-size: 1.0rem; margin-bottom: 2px; font-family: 'JetBrains Mono', monospace; }

.footer { text-align:center; margin-top:20px; font-size:0.6rem; color:#0f172a; letter-spacing:1px; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.2); border-radius: 4px; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO APP
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="VisionGuide AI") as app:

    # Hidden state
    last_time_state       = gr.State(0.0)
    last_nav_state        = gr.State("")
    last_voice_text_state = gr.State("")

    # Header
    gr.HTML("""
    <div class="header-wrap">
        <h1>🦯 VisionGuide <span style="color:#38bdf8;">AI</span></h1>
        <p>Real-Time Navigation · Person &amp; Chair Detection · Voice Guidance · Designed for Visually Impaired Users</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT COLUMN ──
        with gr.Column(scale=5):
            gr.HTML('<div class="sec-label">📷 Input Source</div>')
            with gr.Tabs():
                with gr.Tab("🎥 Live Webcam"):
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="Webcam Feed",
                        height=360,
                    )
                    gr.HTML('<div style="font-size:0.72rem;color:#1e293b;margin-top:8px;">'
                            '⚡ Detection runs every 6 seconds to stay stable — voice auto-plays.</div>')

                with gr.Tab("🖼 Upload Image"):
                    upload_input = gr.Image(
                        sources=["upload"],
                        label="Upload Image",
                        height=320,
                        type="numpy",
                    )
                    run_btn = gr.Button("🔍  Run Detection Now", elem_classes=["btn-run"])

        # ── RIGHT COLUMN ──
        with gr.Column(scale=5):
            gr.HTML('<div class="sec-label">🔍 AI Vision Output</div>')
            output_image = gr.Image(
                label="Detection View",
                height=290,
                interactive=False,
            )
            output_info = gr.Textbox(
                label="Navigation Status & Detections",
                lines=6,
                interactive=False,
                placeholder="Navigation instructions will appear here after detection…",
            )
            output_audio = gr.Audio(
                label="🔊 Voice Guidance",
                autoplay=True,
                interactive=False,
            )

    # Info chips
    gr.HTML("""
    <div class="chips">
        <div class="chip"><strong>Person</strong>Class 0 · 0.5m width</div>
        <div class="chip"><strong>Chair</strong>Class 56 · 0.55m width</div>
        <div class="chip"><strong>3 Zones</strong>Left · Center · Right</div>
        <div class="chip"><strong>gTTS Voice</strong>Auto-plays on change</div>
        <div class="chip"><strong>Throttled</strong>6s stable cooldown</div>
        <div class="chip"><strong>No espeak</strong>Works anywhere</div>
    </div>
    <div class="footer">YOLOv8 · Gradio · OpenCV · gTTS · Internet needed for voice</div>
    """)

    # ── EVENTS ──
    webcam_input.stream(
        fn=process_webcam,
        inputs=[webcam_input, last_time_state, last_nav_state, last_voice_text_state],
        outputs=[output_image, output_info, output_audio,
                 last_time_state, last_nav_state, last_voice_text_state],
        stream_every=0.5,
    )

    run_btn.click(
        fn=process_upload,
        inputs=[upload_input, last_nav_state, last_voice_text_state],
        outputs=[output_image, output_info, output_audio,
                 last_time_state, last_nav_state, last_voice_text_state],
    )

    upload_input.upload(
        fn=process_upload,
        inputs=[upload_input, last_nav_state, last_voice_text_state],
        outputs=[output_image, output_info, output_audio,
                 last_time_state, last_nav_state, last_voice_text_state],
    )


if __name__ == "__main__":
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CSS,
    )