"""Authentication: validation, bcrypt hashing, registration, login, sessions.

Uses bcrypt directly (salted, adaptive hashing) and Streamlit's session_state.
"""

import re
from typing import Optional, Tuple

import bcrypt
import streamlit as st

import database

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_PASSWORD_LEN = 64  # bcrypt only uses the first 72 bytes; keep a safe limit


# ------------------------------------------------------------ validation
def validate_registration(
    username: str, email: str, password: str, confirm: str
) -> Optional[str]:
    """Return an error message, or None if all inputs are valid."""
    if not USERNAME_RE.match(username):
        return "Username must be 3-30 characters: letters, numbers or underscores only."
    if len(email) > 120 or not EMAIL_RE.match(email):
        return "Please enter a valid email address."
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if len(password) > MAX_PASSWORD_LEN:
        return f"Password must be at most {MAX_PASSWORD_LEN} characters long."
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        return "Password must contain at least one letter and one number."
    if password != confirm:
        return "Passwords do not match."
    return None


# --------------------------------------------------------------- hashing
def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


def verify_password(password: str, password_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), bytes(password_hash))
    except (ValueError, TypeError):
        return False


# ------------------------------------------------------ register / login
def register_user(
    username: str, email: str, password: str, confirm: str
) -> Tuple[bool, str]:
    username, email = username.strip(), email.strip().lower()
    error = validate_registration(username, email, password, confirm)
    if error:
        return False, error
    if not database.create_user(username, email, hash_password(password)):
        return False, "That username or email is already registered."
    return True, "Account created successfully. You can now log in."


# Hash used to keep login timing similar when the username does not exist.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password-1", bcrypt.gensalt())


def login_user(username: str, password: str) -> Tuple[bool, str]:
    username = username.strip()
    if not username or not password or len(password) > MAX_PASSWORD_LEN:
        return False, "Invalid username or password."
    user = database.get_user_by_username(username)
    if user is None:
        verify_password(password, _DUMMY_HASH)
        return False, "Invalid username or password."
    if not verify_password(password, user["password_hash"]):
        return False, "Invalid username or password."
    st.session_state["user_id"] = int(user["id"])
    st.session_state["username"] = user["username"]
    return True, f"Welcome back, {user['username']}!"


# --------------------------------------------------------------- session
def is_logged_in() -> bool:
    return "user_id" in st.session_state


def current_user_id() -> Optional[int]:
    return st.session_state.get("user_id")


def current_username() -> Optional[str]:
    return st.session_state.get("username")


def logout() -> None:
    """Clear everything stored in the session."""
    st.session_state.clear()
