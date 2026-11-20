"""
main.py - Face Authorization API (FastAPI).

Run with:
    uvicorn main:app --reload
Web app:
    http://127.0.0.1:8000/
API docs:
    http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field

import auth
import database as db
import routes_people
import routes_recognize


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()   # create/upgrade tables on startup
    yield


app = FastAPI(title="Face Authorization API", lifespan=lifespan)
app.include_router(routes_people.router)      # must come AFTER app is created
app.include_router(routes_recognize.router)


# ---------- request / response models ----------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    email: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------- endpoints ----------

@app.get("/", include_in_schema=False)
def home():
    """Serves the web app (index.html in the project folder)."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", response_model=UserOut, status_code=201)
def register(body: RegisterRequest):
    try:
        user_id = db.create_user(body.email, auth.hash_password(body.password))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"id": user_id, "email": body.email.strip().lower()}


@app.post("/auth/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends()):
    # OAuth2 forms call the email field "username"
    user = auth.authenticate(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": auth.create_access_token(user["id"]), "token_type": "bearer"}


@app.get("/me", response_model=UserOut)
def me(user=Depends(auth.get_current_user)):
    return {"id": user["id"], "email": user["email"]}