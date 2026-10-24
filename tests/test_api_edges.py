"""Failure, configuration, and context-budget behavior at the HTTP boundary."""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI
from pydantic import SecretStr
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.main import create_app


def _app(tmp_path: Path, *, context_model=None, reply_model=None, **settings):
    return create_app(
        settings=Settings(
            _env_file=None,
            database_path=tmp_path / "chat.sqlite3",
            otel_exporter_otlp_endpoint=None,
            **settings,
        ),
        context_model=context_model
        or TestModel(custom_output_args={"language_scope": "unchanged"}),
        reply_model=reply_model or TestModel(custom_output_text="OK"),
    )


@pytest.mark.integration
def test_request_ids_validation_and_missing_configuration(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        good = client.post("/chat", json={"user_id": "a", "session_id": "one", "message": "Hello"})
        another = client.post(
            "/chat", json={"user_id": "a", "session_id": "one", "message": "Again"}
        )
        blank = client.post("/chat", json={"user_id": " ", "session_id": "one", "message": "Hello"})
        unknown = client.post(
            "/chat",
            json={"user_id": "a", "session_id": "one", "message": "Hello", "admin": True},
        )
    assert good.status_code == another.status_code == 200
    assert good.json()["request_id"] == good.headers["X-Request-ID"]
    assert good.json()["request_id"] != another.json()["request_id"]
    assert good.json()["message_id"] < good.json()["response_id"]
    assert blank.status_code == unknown.status_code == 422

    unavailable = create_app(
        settings=Settings(
            _env_file=None,
            database_path=tmp_path / "unavailable.sqlite3",
            llm_base_url=None,
            llm_api_key=None,
            otel_exporter_otlp_endpoint=None,
        )
    )
    with TestClient(unavailable) as client:
        response = client.post(
            "/chat", json={"user_id": "a", "session_id": "one", "message": "Hello"}
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "LLM service is not configured"


@pytest.mark.integration
def test_model_failure_is_safe_and_does_not_fabricate_assistant_message(tmp_path: Path, caplog):
    def fail(_messages, _info):
        raise UnexpectedModelBehavior("private provider response")

    app = _app(tmp_path, context_model=FunctionModel(fail))
    with TestClient(app) as client:
        response = client.post(
            "/chat", json={"user_id": "a", "session_id": "one", "message": "Hello"}
        )
    assert response.status_code == 502
    assert response.json()["detail"] == "LLM provider unavailable"
    assert "private provider response" not in caplog.text
    with sqlite3.connect(app.state.chat_engine.store.path) as db:
        roles = db.execute("SELECT role FROM conversation_messages ORDER BY id").fetchall()
    assert roles == [("user",)]


@pytest.mark.integration
def test_complete_history_uses_budget_without_turn_cap(tmp_path: Path) -> None:
    prompts: list[dict] = []

    def reply(messages, _info):
        prompts.append(json.loads(messages[-1].parts[-1].content))
        return ModelResponse(parts=[TextPart("OK")])

    app = _app(tmp_path, reply_model=FunctionModel(reply))
    with TestClient(app) as client:
        for number in range(23):
            response = client.post(
                "/chat",
                json={"user_id": "a", "session_id": "one", "message": f"Turn {number}"},
            )
            assert response.status_code == 200
    assert len(prompts[-1]["recent_history"]) == 44
    assert prompts[-1]["recent_history"][0]["content"] == "Turn 0"
    assert prompts[-1]["current_message"] == "Turn 22"


def test_secret_settings_aliases_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_PROVIDER_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LITELLM_PROVIDER_API_KEY", "private-test-credential")
    monkeypatch.setenv("LITELLM_MODEL", "test-model")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_base_url == "https://example.invalid/v1"
    assert settings.llm_model == "test-model"
    assert isinstance(settings.llm_api_key, SecretStr)
    assert "private-test-credential" not in repr(settings)


@pytest.mark.integration
def test_owned_provider_client_closes_with_lifespan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[AsyncOpenAI] = []
    real_client = AsyncOpenAI

    def track_client(**kwargs):
        client = real_client(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr("app.main.AsyncOpenAI", track_client)
    app = create_app(
        settings=Settings(
            _env_file=None,
            database_path=tmp_path / "owned.sqlite3",
            otel_exporter_otlp_endpoint=None,
            llm_base_url="https://example.invalid/v1",
            llm_api_key=SecretStr("private-test-credential"),
            llm_model="test-model",
        )
    )
    with TestClient(app):
        assert len(created) == 1
        assert not created[0].is_closed()
    assert created[0].is_closed()
