"""Focused checks for trace parentage and content suppression."""

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.main import create_app
from app.observability import configure_tracing, shutdown_tracing, trace_step


@pytest.mark.unit
def test_agent_spans_share_request_trace_without_private_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    monkeypatch.setattr("app.observability.OTLPSpanExporter", lambda **_kwargs: exporter)
    provider = configure_tracing("http://phoenix:6006/v1/traces")
    assert provider is not None
    try:
        with trace_step("chat.request"):
            with trace_step("chat.service"):
                with trace_step("chat.memory.retrieve"):
                    pass
                result = Agent(
                    TestModel(custom_output_text="synthetic response"), name="main_agent"
                ).run_sync("private-test-message-7362")
        assert result.output == "synthetic response"
        provider.force_flush()
        spans = exporter.get_finished_spans()
        names = {span.name for span in spans}
        assert {"chat.request", "chat.service", "chat.memory.retrieve"} <= names
        assert any("main_agent" in name or "agent" in name for name in names)
        request_span = next(span for span in spans if span.name == "chat.request")
        assert all(span.context.trace_id == request_span.context.trace_id for span in spans)
        assert all(span.parent is not None for span in spans if span.name != "chat.request")
        exported = repr(
            [(span.name, span.attributes, span.events, span.resource.attributes) for span in spans]
        )
        assert "private-test-message-7362" not in exported
        assert "synthetic response" not in exported
    finally:
        shutdown_tracing(provider)


@pytest.mark.unit
def test_invalid_otel_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="without credentials"):
        configure_tracing("http://secret@phoenix:6006/v1/traces")


@pytest.mark.unit
def test_request_accepts_remote_traceparent(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    monkeypatch.setattr("app.observability.OTLPSpanExporter", lambda **_kwargs: exporter)
    provider = configure_tracing("http://phoenix:6006/v1/traces")
    assert provider is not None
    trace_id = "1234567890abcdef1234567890abcdef"
    parent_id = "1234567890abcdef"
    try:
        with trace_step("chat.request", traceparent=f"00-{trace_id}-{parent_id}-01"):
            with trace_step("chat.language"):
                pass
        provider.force_flush()
        spans = exporter.get_finished_spans()
        request = next(span for span in spans if span.name == "chat.request")
        language = next(span for span in spans if span.name == "chat.language")
        assert format(request.context.trace_id, "032x") == trace_id
        assert format(request.parent.span_id, "016x") == parent_id
        assert language.parent.span_id == request.context.span_id
        assert not request.attributes
    finally:
        shutdown_tracing(provider)


@pytest.mark.integration
def test_three_http_turns_export_one_private_safe_trace(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = InMemorySpanExporter()
    monkeypatch.setattr("app.observability.OTLPSpanExporter", lambda **_kwargs: exporter)
    app = create_app(
        settings=Settings(
            _env_file=None,
            database_path=tmp_path / "trace.sqlite3",
            otel_exporter_otlp_endpoint="http://phoenix:6006/v1/traces",
            llm_api_key=SecretStr("private-api-key-8901"),
        ),
        context_model=TestModel(custom_output_args={"language_scope": "unchanged"}),
        reply_model=TestModel(custom_output_text="Synthetic answer."),
    )
    trace_id = "1234567890abcdef1234567890abcdef"
    parent_id = "1234567890abcdef"
    with TestClient(app) as client:
        for number in range(3):
            response = client.post(
                "/chat",
                headers={"traceparent": f"00-{trace_id}-{parent_id}-01"},
                json={
                    "user_id": "private-user",
                    "session_id": "private-session",
                    "message": f"private-message-8291 turn {number}",
                },
            )
            assert response.status_code == 200

    spans = exporter.get_finished_spans()
    requests = [span for span in spans if span.name == "chat.request"]
    assert len(requests) == 3
    assert {span.context.trace_id for span in spans} == {int(trace_id, 16)}
    assert all(request.parent.span_id == int(parent_id, 16) for request in requests)
    names = {span.name for span in spans}
    assert {"chat.request", "chat.service", "chat.language", "chat.main_agent"} <= names
    exported = repr(
        [(span.name, span.attributes, span.events, span.resource.attributes) for span in spans]
    )
    for private in (
        "private-user",
        "private-session",
        "private-message-8291",
        "private-api-key-8901",
    ):
        assert private not in exported


@pytest.mark.integration
def test_provider_failure_does_not_export_private_error_text(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = InMemorySpanExporter()
    monkeypatch.setattr("app.observability.OTLPSpanExporter", lambda **_kwargs: exporter)

    def fail(_messages, _info):
        raise UnexpectedModelBehavior("private-provider-body-6381")

    app = create_app(
        settings=Settings(
            _env_file=None,
            database_path=tmp_path / "failed-trace.sqlite3",
            otel_exporter_otlp_endpoint="http://phoenix:6006/v1/traces",
        ),
        context_model=FunctionModel(fail),
        reply_model=TestModel(custom_output_text="unused"),
    )
    with TestClient(app) as client:
        response = client.post(
            "/chat", json={"user_id": "a", "session_id": "one", "message": "Hello"}
        )
    assert response.status_code == 502
    exported = repr(
        [
            (span.name, span.attributes, span.events, span.resource.attributes)
            for span in exporter.get_finished_spans()
        ]
    )
    assert "private-provider-body-6381" not in exported
