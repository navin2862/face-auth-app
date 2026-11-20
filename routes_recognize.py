"""
routes_recognize.py - check a photo against the logged-in user's authorized people.

POST /recognize    upload ONE photo (or webcam frame); returns every face found,
                   whether each one is authorized, and logs each detection.
GET  /detections   recent detection log for the logged-in user.

Photos are never stored. They are decoded in memory and thrown away.
"""

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

import auth
import database as db
from recognizer import Recognizer

router = APIRouter(tags=["recognition"])

MAX_BYTES = 5 * 1024 * 1024   # 5 MB per photo
MAX_SIDE = 1000               # bigger photos are shrunk for speed


class FaceResult(BaseModel):
    box: list[int]            # [top, right, bottom, left] in the uploaded photo's pixels
    name: str                 # person's name, or "UNAUTHORIZED"
    authorized: bool
    distance: float | None    # lower = closer match (None if nobody is enrolled yet)


class RecognizeOut(BaseModel):
    face_count: int
    authorized_count: int
    faces: list[FaceResult]


class DetectionOut(BaseModel):
    timestamp: str
    name: str
    authorized: bool


def _read_frame(upload: UploadFile):
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Photo is larger than 5 MB.")
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "That file is not a readable image.")
    return frame


@router.post("/recognize", response_model=RecognizeOut)
def recognize(
    photo: UploadFile = File(...),
    log: bool = Query(True, description="Save each detection to the log"),
    user=Depends(auth.get_current_user),
):
    """Find faces in one photo and say who is authorized."""
    frame = _read_frame(photo)

    # shrink big photos for speed, then scale the boxes back to the original size
    h, w = frame.shape[:2]
    factor = 1.0
    if max(h, w) > MAX_SIDE:
        factor = MAX_SIDE / max(h, w)
        frame = cv2.resize(frame, (int(w * factor), int(h * factor)))

    results = Recognizer(owner_id=user["id"]).identify(frame)

    faces = []
    for r in results:
        box = [int(v / factor) for v in r["box"]]
        faces.append(
            FaceResult(
                box=box,
                name=r["name"],
                authorized=r["authorized"],
                distance=r["distance"],
            )
        )
        if log:
            db.log_detection(r["name"], r["authorized"], owner_id=user["id"])

    return RecognizeOut(
        face_count=len(faces),
        authorized_count=sum(1 for f in faces if f.authorized),
        faces=faces,
    )


@router.get("/detections", response_model=list[DetectionOut])
def detections(
    limit: int = Query(20, ge=1, le=200),
    user=Depends(auth.get_current_user),
):
    """Most recent detections, newest first."""
    return db.recent_detections(limit=limit, owner_id=user["id"])