"""Automated tests for the database and authentication logic (no OpenAI calls)."""

import pytest

import auth
import database
import health_consultation as hc


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    database.init_db()


def make_user(name="alice", email="alice@example.com", pw="Passw0rdX"):
    assert database.create_user(name, email, auth.hash_password(pw))
    return database.get_user_by_username(name)["id"]


def test_password_is_hashed_and_verifies():
    h = auth.hash_password("Secret123")
    assert b"Secret123" not in h
    assert auth.verify_password("Secret123", h)
    assert not auth.verify_password("wrong", h)


def test_registration_validation():
    assert auth.validate_registration("ab", "a@b.co", "Passw0rd1", "Passw0rd1")
    assert auth.validate_registration("alice", "bad", "Passw0rd1", "Passw0rd1")
    assert auth.validate_registration("alice", "a@b.co", "short1", "short1")
    assert auth.validate_registration("alice", "a@b.co", "onlyletters", "onlyletters")
    assert auth.validate_registration("alice", "a@b.co", "Passw0rd1", "Different1")
    assert auth.validate_registration("alice", "a@b.co", "Passw0rd1", "Passw0rd1") is None


def test_duplicate_registration_rejected():
    ok, _ = auth.register_user("bob", "bob@example.com", "Passw0rd1", "Passw0rd1")
    assert ok
    ok, _ = auth.register_user("BOB", "other@example.com", "Passw0rd1", "Passw0rd1")
    assert not ok
    ok, _ = auth.register_user("bob2", "bob@example.com", "Passw0rd1", "Passw0rd1")
    assert not ok


def test_sql_injection_username_is_just_text():
    ok, _ = auth.register_user("x' OR '1'='1", "i@example.com", "Passw0rd1", "Passw0rd1")
    assert not ok  # rejected by validation
    assert database.get_user_by_username("x' OR '1'='1") is None


def test_users_cannot_access_each_others_history():
    a = make_user("alice", "alice@example.com")
    b = make_user("bella", "bella@example.com")
    cid = database.add_consultation(a, "headache for two days", "response", "Adult (18-64)", "1-3 days")

    assert database.get_consultation(a, cid) is not None
    assert database.get_consultation(b, cid) is None
    assert database.list_consultations(b) == []
    assert database.delete_consultation(b, cid) is False
    assert database.get_consultation(a, cid) is not None
    assert database.delete_consultation(a, cid) is True
    assert database.get_consultation(a, cid) is None


def test_delete_all_and_dataframe():
    a = make_user()
    for i in range(3):
        database.add_consultation(a, f"symptom number {i}", "resp", "Not specified", "Not specified")
    assert len(database.consultations_dataframe(a)) == 3
    assert database.delete_all_consultations(a) == 3
    assert database.consultations_dataframe(a).empty


def test_symptom_validation_and_missing_key(monkeypatch):
    assert hc.validate_symptoms("short")
    assert hc.validate_symptoms("x" * (hc.MAX_SYMPTOM_CHARS + 1))
    assert hc.validate_symptoms("a mild headache since yesterday") == ""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(hc, "load_dotenv", lambda *a, **k: None)
    ok, msg = hc.get_health_guidance("a mild headache since yesterday", "Not specified", "Not specified")
    assert not ok and "API key" in msg
