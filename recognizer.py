"""
recognizer.py - face recognition engine for the authorization app.

Used by the UI, but you can also test it from the terminal:

    python recognizer.py enroll "Alice"    # webcam opens, SPACE = capture, ESC = cancel
    python recognizer.py run               # live detection, 'q' to quit
"""

import os
import re
import sys
import threading

import cv2
import numpy as np
import face_recognition

import database as db

PHOTO_DIR = "authorized_faces"
TOLERANCE = 0.5   # lower = stricter (library default is 0.6)
SCALE = 0.5       # live frames are shrunk by this factor for faster detection

# dlib's models are NOT thread-safe. The live video runs in one thread and the
# sidebar buttons run in another, so every dlib call goes through this lock.
_dlib_lock = threading.Lock()


class Recognizer:
    def __init__(self, tolerance=TOLERANCE, owner_id=None):
        self.tolerance = tolerance
        self.owner_id = owner_id   # None = old single-user data (Streamlit prototype)
        self.encodings, self.names = [], []
        self.reload()

    def reload(self):
        """Re-read authorized faces from the database (call after add/remove)."""
        self.encodings, self.names = db.load_known_faces(owner_id=self.owner_id)

    def best_match(self, encoding):
        """Closest authorized person to this encoding.
        Returns (name, distance) if within tolerance, else (None, distance or None)."""
        known, names = self.encodings, self.names
        if not known:
            return None, None
        dists = face_recognition.face_distance(known, encoding)
        best = int(np.argmin(dists))
        if dists[best] <= self.tolerance and best < len(names):
            return names[best], float(dists[best])
        return None, float(dists[best])

    def identify(self, frame_bgr):
        """Find all faces in a BGR frame.
        Returns a list of dicts: box (top, right, bottom, left) in original
        frame coordinates, name, authorized (bool), distance (float or None)."""
        small = cv2.resize(frame_bgr, (0, 0), fx=SCALE, fy=SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        with _dlib_lock:
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

        results = []
        for loc, enc in zip(locations, encodings):
            name, distance = self.best_match(enc)
            authorized = name is not None
            box = tuple(int(v / SCALE) for v in loc)
            results.append(
                {
                    "box": box,
                    "name": name if authorized else "UNAUTHORIZED",
                    "authorized": authorized,
                    "distance": distance,
                }
            )
        return results


# ---------- enrollment helpers ----------

def _shrink(frame_bgr, max_side=800):
    h, w = frame_bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)))
    return frame_bgr


def _find_faces(rgb):
    """Call inside the lock. Retries with a more thorough search if nothing found."""
    locations = face_recognition.face_locations(rgb)
    if not locations:
        locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=2)
    return locations


def _require_one(locations):
    if len(locations) == 0:
        raise ValueError("No face found. Try better lighting and face the camera.")
    if len(locations) > 1:
        raise ValueError("More than one face found. Only one person at a time.")


def _single_encoding(frame_bgr):
    """Encoding of the one face in the image (raises ValueError otherwise)."""
    rgb = cv2.cvtColor(_shrink(frame_bgr), cv2.COLOR_BGR2RGB)
    with _dlib_lock:
        locations = _find_faces(rgb)
        _require_one(locations)
        return face_recognition.face_encodings(rgb, locations)[0]


def analyze_face(frame_bgr):
    """Analyze the one face in a frame, for guided enrollment.

    Returns a dict:
        encoding    128-number face encoding
        yaw         head turn estimate: ~0 facing the camera, positive/negative
                    when turned to one side or the other (|0.15| is roughly 15-20 degrees)
        face_width  width of the face in pixels (on an image at most 800px wide)
        sharpness   blur score of the face area (higher = sharper)
    Raises ValueError if there isn't exactly one face."""
    small = _shrink(frame_bgr)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    with _dlib_lock:
        locations = _find_faces(rgb)
        _require_one(locations)
        encoding = face_recognition.face_encodings(rgb, locations)[0]
        landmarks = face_recognition.face_landmarks(rgb, locations)

    top, right, bottom, left = locations[0]
    crop = gray[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]
    sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0

    yaw = 0.0
    if landmarks:
        lm = landmarks[0]
        eye_a = np.mean(lm["left_eye"], axis=0)
        eye_b = np.mean(lm["right_eye"], axis=0)
        nose = np.mean(lm["nose_tip"], axis=0)
        eye_dist = float(np.linalg.norm(eye_b - eye_a))
        if eye_dist > 0:
            yaw = float((nose[0] - (eye_a[0] + eye_b[0]) / 2) / eye_dist)

    return {
        "encoding": encoding,
        "yaw": yaw,
        "face_width": int(right - left),
        "sharpness": sharpness,
    }


def _photo_path_for(name):
    os.makedirs(PHOTO_DIR, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()) or "person"
    return os.path.join(PHOTO_DIR, f"{safe_name}.jpg")


def enroll_from_image(name, frame_bgr):
    """Validate the image has exactly one face, then save the person.
    Raises ValueError with a readable message if something is wrong."""
    encoding = _single_encoding(frame_bgr)
    photo_path = _photo_path_for(name)
    person_id = db.add_person(name, encoding, photo_path)  # raises if name exists
    cv2.imwrite(photo_path, frame_bgr)
    return person_id


def add_photo_for_person(person_id, frame_bgr):
    """Add another photo's encoding to an existing person."""
    db.add_encoding(person_id, _single_encoding(frame_bgr))


def enroll_poses(name, encodings, profile_frame_bgr):
    """Save a person from several pose encodings (guided enrollment).
    If the name already exists, the encodings are added to that person."""
    name = name.strip()
    existing = next(
        (p for p in db.list_people() if p["name"].lower() == name.lower()), None
    )
    if existing:
        person_id = existing["id"]
        to_add = list(encodings)
    else:
        photo_path = _photo_path_for(name)
        person_id = db.add_person(name, encodings[0], photo_path)
        cv2.imwrite(photo_path, profile_frame_bgr)
        to_add = list(encodings[1:])
    for enc in to_add:
        db.add_encoding(person_id, enc)
    return person_id


def draw_results(frame, results):
    """Draw green (authorized) or red (unauthorized) boxes with labels."""
    for r in results:
        top, right, bottom, left = r["box"]
        color = (0, 200, 0) if r["authorized"] else (0, 0, 255)
        label = r["name"] if r["authorized"] else "UNAUTHORIZED"
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


# ---------- terminal test mode ----------

def _cli_enroll(name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    print("Look at the camera. SPACE = capture, ESC = cancel.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Enroll - SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Cancelled.")
            break
        if key == 32:
            try:
                enroll_from_image(name, frame)
                print(f"Enrolled '{name}'.")
                break
            except ValueError as e:
                print("Could not enroll:", e)
    cap.release()
    cv2.destroyAllWindows()


def _cli_run():
    rec = Recognizer()
    print(f"Loaded {len(rec.names)} face(s). Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        draw_results(frame, rec.identify(frame))
        cv2.imshow("Face Authorization", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    db.init_db()
    if len(sys.argv) >= 3 and sys.argv[1] == "enroll":
        _cli_enroll(" ".join(sys.argv[2:]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "run":
        _cli_run()
    else:
        print('Usage:\n  python recognizer.py enroll "Name"\n  python recognizer.py run')