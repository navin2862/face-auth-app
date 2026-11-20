"""
auth.py - password hashing and JWT login for the Face Authorization API.

Needs JWT_SECRET in your .env file (a long random string).
"""

import os
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash

import database as db

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET is not set. Add a line like JWT_SECRET=<long random string> to .env"
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = 60

# Argon2 is the recommended modern password hash. We never store real passwords.
password_hash = PasswordHash.recommended()
# Used to keep login timing similar whether or not the email exists.
_DUMMY_HASH = password_hash.hash("not-a-real-password")

# Makes the "Authorize" button appear in /docs and reads the Bearer token.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def authenticate(email: str, password: str):
    """Return the user row if email and password are correct, else None."""
    user = db.get_user_by_email(email)
    if not user:
        password_hash.verify(password, _DUMMY_HASH)
        return None
    if not password_hash.verify(password, user["password_hash"]):
        return None
    return user


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    """FastAPI dependency: returns the logged-in user or raises 401."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise credentials_error

    user = db.get_user_by_id(user_id)
    if not user:
        raise credentials_error
    return user