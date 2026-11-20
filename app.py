"""
app.py - Streamlit UI for the face authorization app.

Run with:
    streamlit run app.py

Sidebar : guided multi-pose enrollment (or photo upload), list/delete people
Main    : live camera with green/red boxes, and a log of recent detections
"""

import os
import threading
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

import database as db
from recognizer import (
    Recognizer,
    analyze_face,
    enroll_from_image,
    enroll_poses,
    add_photo_for_person,
    draw_results,
)

PROCESS_EVERY = 3     # run recognition on every Nth video frame
LOG_COOLDOWN = 10     # seconds before the same person is logged again

# ---- guided enrollment settings (tweak these if checks feel too strict/loose) ----
POSES = [
    {"kind": "front", "title": "Look straight at the camera",
     "hint": "Keep your face centered with a neutral expression."},
    {"kind": "turn_a", "title": "Turn your head slightly to ONE side",
     "hint": "About 20-30 degrees. Keep your eyes on the screen."},
    {"kind": "turn_b", "title": "Now turn slightly to the OTHER side",
     "hint": "Same amount, opposite direction."},
    {"kind": "front", "title": "Look straight again and smile",
     "hint": "Face the camera directly."},
]
YAW_FRONT_MAX = 0.15     # how far from the first photo's angle still counts as "straight"
YAW_TURN_MIN = 0.12      # how far you must turn to count as a "side" pose
SAME_PERSON_MAX = 0.60   # looser than the match slider: turned faces look a bit different
MIN_FACE_WIDTH = 90      # smaller than this = too far from the camera
MIN_SHARPNESS = 25       # lower than this = too blurry

st.set_page_config(page_title="Face Authorization", page_icon="🔐", layout="wide")


# ---------- shared engine (created once, used by the video thread and the UI) ----------

class Engine:
    def __init__(self):
        self.rec = Recognizer()
        self.lock = threading.Lock()
        self.latest = None
        self.results = []
        self.count = 0
        self.last_logged = {}

    def process(self, frame):
        """Called from the video thread for every frame."""
        with self.lock:
            self.latest = frame.copy()
        self.count += 1
        if self.count % PROCESS_EVERY == 0:
            self.results = self.rec.identify(frame)
            self._log(self.results)
        return draw_results(frame, self.results)

    def _log(self, results):
        now = time.time()
        for r in results:
            key = r["name"] if r["authorized"] else "UNAUTHORIZED"
            if now - self.last_logged.get(key, 0) > LOG_COOLDOWN:
                db.log_detection(key, r["authorized"])
                self.last_logged[key] = now

    def get_latest(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()


@st.cache_resource
def get_engine():
    db.init_db()
    return Engine()


engine = get_engine()

if "guided" not in st.session_state:
    st.session_state.guided = None   # None, or {"name", "step", "shots"}
if "flash" not in st.session_state:
    st.session_state.flash = None    # (kind, message) shown once at top of sidebar


def flash(kind, message):
    st.session_state.flash = (kind, message)


# ---------- guided enrollment logic ----------

def start_guided():
    name = st.session_state.get("enroll_name", "").strip()
    if not name:
        flash("error", "Enter a name first.")
        return
    if engine.get_latest() is None:
        flash("error", "Camera isn't running. Click START on the right first.")
        return
    st.session_state.guided = {"name": name, "step": 0, "shots": []}


def cancel_guided():
    st.session_state.guided = None
    flash("info", "Enrollment cancelled. Nothing was saved.")


def check_pose(g, info):
    """Return a message describing what's wrong with this capture, or None if OK."""
    if info["face_width"] < MIN_FACE_WIDTH:
        return "Move closer to the camera."
    if info["sharpness"] < MIN_SHARPNESS:
        return "The image is blurry. Hold still and add more light on your face."

    shots = g["shots"]
    if shots:
        dist = min(float(np.linalg.norm(s["encoding"] - info["encoding"])) for s in shots)
        if dist > SAME_PERSON_MAX:
            return (f"This looks like a different person than the earlier photos "
                    f"(difference {dist:.2f}, limit {SAME_PERSON_MAX}). "
                    "Only one person per enrollment.")

    yaw0 = shots[0]["yaw"] if shots else 0.0
    delta = info["yaw"] - yaw0
    kind = POSES[g["step"]]["kind"]

    if kind == "front":
        if abs(delta) > YAW_FRONT_MAX:
            return (f"Face the camera straight on "
                    f"(off by {abs(delta):.2f}, limit {YAW_FRONT_MAX}).")
    elif kind == "turn_a":
        if abs(delta) < YAW_TURN_MIN:
            return (f"Turn your head a bit more to the side "
                    f"(turn {abs(delta):.2f}, need {YAW_TURN_MIN}).")
    elif kind == "turn_b":
        first_turn = shots[1]["yaw"] - yaw0
        if delta * first_turn > 0:
            return "Turn to the OTHER side this time (you turned the same way as before)."
        if abs(delta) < YAW_TURN_MIN:
            return (f"Turn your head a bit more to the side "
                    f"(turn {abs(delta):.2f}, need {YAW_TURN_MIN}).")
    return None


def finish_guided():
    g = st.session_state.guided
    name = g["name"]
    encodings = [s["encoding"] for s in g["shots"]]

    existing = next((p for p in db.list_people()
                     if p["name"].lower() == name.lower()), None)
    match_name, _ = engine.rec.best_match(encodings[0])

    if existing:
        if match_name is None or match_name.lower() != name.lower():
            flash("error", f"This face doesn't match the existing '{existing['name']}'. "
                           "Nothing was saved.")
            st.session_state.guided = None
            return
    elif match_name is not None:
        flash("error", f"This face is already authorized as '{match_name}'. Nothing was saved.")
        st.session_state.guided = None
        return

    try:
        enroll_poses(name, encodings, g["shots"][0]["frame"])
    except ValueError as e:
        flash("error", str(e))
        st.session_state.guided = None
        return

    engine.rec.reload()
    st.session_state.guided = None
    flash("success", f"'{name}' is now authorized ({len(encodings)} poses saved).")


def capture_pose():
    g = st.session_state.guided
    frame = engine.get_latest()
    if frame is None:
        flash("error", "Camera isn't running. Click START first.")
        return
    try:
        info = analyze_face(frame)
    except ValueError as e:
        flash("error", str(e))
        return

    problem = check_pose(g, info)
    if problem:
        flash("error", problem)
        return

    thumb = cv2.cvtColor(cv2.resize(frame, (120, 90)), cv2.COLOR_BGR2RGB)
    g["shots"].append(
        {"frame": frame, "encoding": info["encoding"], "yaw": info["yaw"], "thumb": thumb}
    )
    g["step"] += 1
    if g["step"] >= len(POSES):
        finish_guided()


# ---------- other sidebar helpers ----------

def delete_person(person_id, photo_path):
    db.remove_person(person_id)
    if photo_path and os.path.exists(photo_path):
        os.remove(photo_path)
    engine.rec.reload()


def shrink(frame, max_side=1000):
    h, w = frame.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        frame = cv2.resize(frame, (int(w * s), int(h * s)))
    return frame


def try_enroll_upload(name, frame):
    name = name.strip()
    if not name:
        st.sidebar.error("Enter a name first.")
        return
    try:
        existing = next((p for p in db.list_people()
                         if p["name"].lower() == name.lower()), None)
        if existing:
            add_photo_for_person(existing["id"], frame)
            msg = f"Added another photo for '{existing['name']}'."
        else:
            enroll_from_image(name, frame)
            msg = f"'{name}' is now authorized."
        engine.rec.reload()
        st.sidebar.success(msg)
    except ValueError as e:
        st.sidebar.error(str(e))


# ---------- sidebar ----------

st.sidebar.title("🔐 Authorize a person")

if st.session_state.flash:
    kind, message = st.session_state.flash
    getattr(st.sidebar, kind)(message)
    st.session_state.flash = None

g = st.session_state.guided
if g:
    step = g["step"]
    st.sidebar.markdown(f"**Enrolling: {g['name']}**")
    st.sidebar.progress(step / len(POSES), text=f"Pose {step + 1} of {len(POSES)}")
    st.sidebar.info(f"**{POSES[step]['title']}**\n\n{POSES[step]['hint']}")
    st.sidebar.caption("Look at the live camera on the right, then click Capture.")

    if g["shots"]:
        cols = st.sidebar.columns(len(POSES))
        for i, shot in enumerate(g["shots"]):
            cols[i].image(shot["thumb"], caption=f"{i + 1}")

    b1, b2 = st.sidebar.columns(2)
    b1.button("📸 Capture", type="primary", on_click=capture_pose, use_container_width=True)
    b2.button("Cancel", on_click=cancel_guided, use_container_width=True)
else:
    st.sidebar.text_input("Name", key="enroll_name")
    method = st.sidebar.radio(
        "How to add", ["Guided capture (recommended)", "Upload photo"], horizontal=False
    )
    if method.startswith("Guided"):
        st.sidebar.caption(
            f"You'll be asked for {len(POSES)} poses so the app learns your face "
            "from different angles."
        )
        st.sidebar.button("Start guided capture", type="primary", on_click=start_guided)
    else:
        upload = st.sidebar.file_uploader("Photo", type=["jpg", "jpeg", "png"])
        if st.sidebar.button("Authorize from photo", type="primary"):
            if upload is None:
                st.sidebar.error("Choose a photo first.")
            else:
                data = np.frombuffer(upload.getvalue(), np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is None:
                    st.sidebar.error("Could not read that image.")
                else:
                    try_enroll_upload(st.session_state.get("enroll_name", ""), shrink(frame))

st.sidebar.divider()
st.sidebar.subheader("Authorized people")
people = db.list_people()
if not people:
    st.sidebar.caption("Nobody yet.")
for p in people:
    c1, c2, c3 = st.sidebar.columns([1, 3, 1])
    if p["photo_path"] and os.path.exists(p["photo_path"]):
        c1.image(p["photo_path"], width=40)
    c2.write(f"{p['name']} ({p['photo_count']})")
    c3.button("🗑", key=f"del_{p['id']}", on_click=delete_person,
              args=(p["id"], p["photo_path"]), help=f"Remove {p['name']}")

st.sidebar.divider()
engine.rec.tolerance = st.sidebar.slider(
    "Match strictness (lower = stricter)", 0.30, 0.70, float(engine.rec.tolerance), 0.01
)


# ---------- main area: live detection + log ----------

st.title("Face Authorization")
st.caption("Green box = authorized. Red box = unauthorized.")


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = engine.process(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="face-auth",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
        "audio": False,
    },
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)


@st.fragment(run_every=3)
def show_log():
    st.subheader("Recent detections")
    rows = db.recent_detections(15)
    if not rows:
        st.caption("No detections yet.")
        return
    st.dataframe(
        [
            {
                "Time": r["timestamp"],
                "Name": r["name"],
                "Status": "✅ Authorized" if r["authorized"] else "🚫 Unauthorized",
            }
            for r in rows
        ],
        hide_index=True,
    )


show_log()