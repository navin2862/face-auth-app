"""
routes_people.py - endpoints for managing a user's authorized people.

Every endpoint needs a logged-in user (Bearer token) and only ever touches
that user's own data.

Design choice: photos are NOT stored. Only the face encoding (128 numbers) is
kept, which is better for privacy and works on hosts with temporary disks.
"""

from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel

import auth
import database as db
from recognizer import Recognizer, analyze_face

router = APIRouter(prefix="/people", tags=["people"])

MAX_PHOTOS = 8                   # photos accepted per request
MAX_BYTES = 5 * 1024 * 1024      # 5 MB per photo
MIN_FACE_WIDTH = 90              # face smaller than this = too far away
MIN_SHARPNESS = 25               # lower than this = too blurry
SAME_PERSON_MAX = 0.60           # photos in one request must be the same person


class PersonOut(BaseModel):
    id: int
    name: str
    photo_count: int
    created_at: datetime


# ---------- helpers ----------

def _read_frames(photos):
    if not photos:
        raise HTTPException(422, "Upload at least one photo.")
    if len(photos) > MAX_PHOTOS:
        raise HTTPException(422, f"At most {MAX_PHOTOS} photos at a time.")
    frames = []
    for i, upload in enumerate(photos, 1):
        data = upload.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"Photo {i} is larger than 5 MB.")
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(422, f"Photo {i} is not a readable image.")
        frames.append(frame)
    return frames


def _encode_all(frames):
    """Validate every photo and return their encodings."""
    encodings = []
    for i, frame in enumerate(frames, 1):
        try:
            info = analyze_face(frame)
        except ValueError as e:
            raise HTTPException(422, f"Photo {i}: {e}")
        if info["face_width"] < MIN_FACE_WIDTH:
            raise HTTPException(422, f"Photo {i}: face is too small. Move closer.")
        if info["sharpness"] < MIN_SHARPNESS:
            raise HTTPException(422, f"Photo {i}: photo is too blurry.")
        if encodings:
            dist = min(float(np.linalg.norm(e - info["encoding"])) for e in encodings)
            if dist > SAME_PERSON_MAX:
                raise HTTPException(
                    422, f"Photo {i} looks like a different person than the earlier photos."
                )
        encodings.append(info["encoding"])
    return encodings


def _get_person(person_id, owner_id):
    person = next(
        (p for p in db.list_people(owner_id=owner_id) if p["id"] == person_id), None
    )
    if not person:
        raise HTTPException(404, "Person not found.")
    return person


# ---------- endpoints ----------

@router.get("", response_model=list[PersonOut])
def list_people(user=Depends(auth.get_current_user)):
    return db.list_people(owner_id=user["id"])


@router.post("", response_model=PersonOut, status_code=201)
def create_person(
    name: str = Form(..., max_length=100),
    photos: list[UploadFile] = File(...),
    user=Depends(auth.get_current_user),
):
    """Authorize a new person from one or more photos (ideally a few angles)."""
    name = name.strip()
    if not name:
        raise HTTPException(422, "Name cannot be empty.")

    encodings = _encode_all(_read_frames(photos))

    # don't let the same face be authorized twice under different names
    match, _ = Recognizer(owner_id=user["id"]).best_match(encodings[0])
    if match:
        raise HTTPException(409, f"This face is already authorized as '{match}'.")

    try:
        person_id = db.add_person(name, encodings[0], None, owner_id=user["id"])
    except ValueError as e:
        raise HTTPException(409, str(e))
    try:
        for enc in encodings[1:]:
            db.add_encoding(person_id, enc, owner_id=user["id"])
    except Exception:
        db.remove_person(person_id, owner_id=user["id"])   # don't leave half a person
        raise

    return _get_person(person_id, user["id"])


@router.post("/{person_id}/photos", response_model=PersonOut)
def add_photos(
    person_id: int,
    photos: list[UploadFile] = File(...),
    user=Depends(auth.get_current_user),
):
    """Add more photos to an existing person (must be the same face)."""
    person = _get_person(person_id, user["id"])
    encodings = _encode_all(_read_frames(photos))

    match, _ = Recognizer(owner_id=user["id"]).best_match(encodings[0])
    if match is None or match.lower() != person["name"].lower():
        raise HTTPException(422, f"These photos don't match '{person['name']}'.")

    for enc in encodings:
        db.add_encoding(person_id, enc, owner_id=user["id"])
    return _get_person(person_id, user["id"])


@router.delete("/{person_id}", status_code=204)
def delete_person(person_id: int, user=Depends(auth.get_current_user)):
    if not db.remove_person(person_id, owner_id=user["id"]):
        raise HTTPException(404, "Person not found.")
    return Response(status_code=204)