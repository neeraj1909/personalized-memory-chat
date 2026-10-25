"""Public journeys through FastAPI, PydanticAI test models, and SQLite."""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.main import create_app


@pytest.fixture
def chat_app(tmp_path: Path):
    return create_app(
        settings=Settings(
            _env_file=None,
            database_path=tmp_path / "chat.sqlite3",
            otel_exporter_otlp_endpoint=None,
        ),
        context_model=TestModel(custom_output_args={"language_scope": "unchanged"}),
        reply_model=TestModel(custom_output_text="A useful reply."),
    )


def send(client: TestClient, app, message: str, decision: dict, *, user="a", session="one"):
    with app.state.context_agent.override(model=TestModel(custom_output_args=decision)):
        return client.post(
            "/chat", json={"user_id": user, "session_id": session, "message": message}
        )


def recording_app(tmp_path: Path):
    prompts: list[dict] = []

    def reply(messages, _info):
        prompts.append(json.loads(messages[-1].parts[-1].content))
        return ModelResponse(parts=[TextPart("A useful reply.")])

    app = create_app(
        settings=Settings(
            _env_file=None,
            database_path=tmp_path / "recording.sqlite3",
            otel_exporter_otlp_endpoint=None,
        ),
        context_model=TestModel(custom_output_args={"language_scope": "unchanged"}),
        reply_model=FunctionModel(reply),
    )
    return app, prompts


@pytest.mark.integration
@pytest.mark.parametrize(
    ("message", "opposing_history"),
    [
        (
            "How should I prepare for a data analyst role?",
            "Yaar, shuruat ke liye ek plan bana lete hain.",
        ),
        ("डेटा विश्लेषक की भूमिका के लिए तैयारी कैसे करूँ?", "What is a data analyst?"),
        (
            "Yaar, main data analyst role ke liye prepare kar raha hoon, start kahan se karoon",
            "मुझे करियर सलाह चाहिए।",
        ),
    ],
)
@pytest.mark.parametrize("with_history", [False, True], ids=["fresh", "mixed_history"])
def test_unset_language_uses_full_current_turn_without_saving_preference(
    tmp_path: Path, message: str, opposing_history: str, with_history: bool
) -> None:
    app, prompts = recording_app(tmp_path)
    with TestClient(app) as client:
        if with_history:
            assert send(client, app, opposing_history, {}).status_code == 200
        response = send(client, app, message, {})
    assert response.status_code == 200
    assert prompts[-1]["current_message"] == message
    assert prompts[-1]["reply_language"] is None
    if with_history:
        assert prompts[-1]["recent_history"][0] == {
            "role": "user",
            "content": opposing_history,
        }
    else:
        assert prompts[-1]["recent_history"] == []
    with sqlite3.connect(app.state.chat_engine.store.path) as db:
        preferences = db.execute(
            "SELECT value FROM user_memories WHERE user_id='a' AND memory_key='language_preference'"
        ).fetchall()
    assert preferences == []


@pytest.mark.integration
@pytest.mark.parametrize(
    ("statement", "followup", "finding", "expected_key"),
    [
        (
            "मेरा लक्ष्य अगले साल शिक्षक बनना है।",
            "What goal did I mention?",
            {
                "kind": "goal",
                "key": "career_goal",
                "value": "become a teacher next year",
                "evidence": "मेरा लक्ष्य अगले साल शिक्षक बनना है",
            },
            "career_goal",
        ),
        (
            "Mujhe sitar bajana pasand hai.",
            "What instrument do I like?",
            {
                "kind": "interest",
                "key": "interest_sitar",
                "value": "playing sitar",
                "evidence": "Mujhe sitar bajana pasand hai",
            },
            "interest_sitar",
        ),
    ],
)
def test_hindi_and_hinglish_self_facts_recall_across_sessions(
    chat_app, statement: str, followup: str, finding: dict, expected_key: str
) -> None:
    with TestClient(chat_app) as client:
        first = send(client, chat_app, statement, {"new_memories": [finding]})
        second = send(client, chat_app, followup, {"relevant_memory_ids": [1]}, session="two")
    assert first.status_code == second.status_code == 200
    assert expected_key in second.json()["context_used"]
    assert second.json()["message_id"] < second.json()["response_id"]


@pytest.mark.integration
def test_quoted_third_party_goal_is_not_user_memory(chat_app) -> None:
    quote = "Here is my sister's quote: 'I plan to become a chef.' Please translate it."
    mistaken_finding = {
        "kind": "goal",
        "key": "career_goal",
        "value": "become a chef",
        "evidence": "I plan to become a chef",
    }
    with TestClient(chat_app) as client:
        response = send(client, chat_app, quote, {"new_memories": [mistaken_finding]})
    assert response.status_code == 200
    with sqlite3.connect(chat_app.state.chat_engine.store.path) as db:
        rows = db.execute("SELECT memory_key FROM user_memories WHERE user_id='a'").fetchall()
    assert not any(key == "career_goal" for (key,) in rows)


@pytest.mark.integration
def test_full_current_message_history_and_user_session_scope(tmp_path: Path) -> None:
    app, prompts = recording_app(tmp_path)
    long_message = "A specific plan: " + "career " * 5000
    with TestClient(app) as client:
        first = send(client, app, long_message, {})
        other = send(client, app, "Other user's secret.", {}, user="b")
        followup = send(client, app, "Please continue.", {})
    assert first.status_code == other.status_code == followup.status_code == 200
    assert prompts[-1]["current_message"] == "Please continue."
    assert prompts[-1]["recent_history"] == [
        {"role": "user", "content": long_message},
        {"role": "assistant", "content": "A useful reply."},
    ]
    with sqlite3.connect(app.state.chat_engine.store.path) as db:
        stored = db.execute(
            "SELECT content FROM conversation_messages WHERE id=?", (first.json()["message_id"],)
        ).fetchone()[0]
    assert stored == long_message


@pytest.mark.integration
def test_relevant_memory_only_and_no_cross_user_ids(tmp_path: Path) -> None:
    app, prompts = recording_app(tmp_path)
    catalogues: list[dict] = []

    def choose_career(messages, info):
        catalogues.append(json.loads(messages[-1].parts[-1].content))
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"relevant_memory_ids": [1]})]
        )

    with TestClient(app) as client:
        first = send(
            client,
            app,
            "I plan to change jobs next year, I enjoy balcony gardening, and I prefer tea.",
            {
                "new_memories": [
                    {
                        "kind": "goal",
                        "key": "career_goal",
                        "value": "change jobs next year",
                        "evidence": "I plan to change jobs next year",
                    },
                    {
                        "kind": "interest",
                        "key": "interest_balcony_gardening",
                        "value": "balcony gardening",
                        "evidence": "I enjoy balcony gardening",
                    },
                    {
                        "kind": "preference",
                        "key": "drink_preference",
                        "value": "tea",
                        "evidence": "I prefer tea",
                    },
                ]
            },
        )
        with app.state.context_agent.override(model=FunctionModel(choose_career)):
            career = client.post(
                "/chat",
                json={
                    "user_id": "a",
                    "session_id": "two",
                    "message": "What should I do for my career?",
                },
            )
        stranger = send(
            client, app, "What do you know about my plans?", {"relevant_memory_ids": [1]}, user="b"
        )
    assert first.status_code == career.status_code == stranger.status_code == 200
    assert career.json()["context_used"] == ["career_goal"]
    assert {item["key"] for item in catalogues[0]["stored_memories"]} == {
        "career_goal",
        "interest_balcony_gardening",
        "drink_preference",
    }
    assert all("value" not in item for item in catalogues[0]["stored_memories"])
    assert "balcony gardening" not in json.dumps(catalogues[0])
    assert [item["key"] for item in prompts[-2]["relevant_memories"]] == ["career_goal"]
    assert stranger.json()["context_used"] == []
    assert prompts[-1]["relevant_memories"] == []


@pytest.mark.integration
def test_durable_language_and_one_turn_override(tmp_path: Path) -> None:
    app, prompts = recording_app(tmp_path)
    with TestClient(app) as client:
        hindi = send(
            client,
            app,
            "Aaj se Hindi mein jawab dena.",
            {
                "language_scope": "durable",
                "reply_language": "Hindi",
                "language_evidence": "Aaj se Hindi mein jawab dena",
            },
        )
        temporary = send(
            client,
            app,
            "For this answer only, use English.",
            {
                "language_scope": "current_turn",
                "reply_language": "English",
                "language_evidence": "For this answer only, use English",
            },
            session="two",
        )
        resumed = send(client, app, "What next?", {}, session="three")
        ambiguous = send(client, app, "Use that language.", {"language_scope": "clarify"})
    assert hindi.status_code == temporary.status_code == resumed.status_code == 200
    assert [prompt["reply_language"] for prompt in prompts] == ["Hindi", "English", "Hindi"]
    assert temporary.json()["context_used"] == []
    assert resumed.json()["context_used"] == ["language_preference"]
    assert ambiguous.json()["response"] == "Which language would you like me to use?"
    assert ambiguous.json()["context_used"] == ["language_preference"]
    with sqlite3.connect(app.state.chat_engine.store.path) as db:
        values = db.execute(
            "SELECT value FROM user_memories WHERE user_id='a' AND memory_key='language_preference'"
        ).fetchall()
    assert values == [("Hindi",)]


@pytest.mark.integration
def test_quoted_language_instruction_cannot_replace_preference(chat_app) -> None:
    with TestClient(chat_app) as client:
        send(
            client,
            chat_app,
            "Please reply in Hindi.",
            {
                "language_scope": "durable",
                "reply_language": "Hindi",
                "language_evidence": "Please reply in Hindi",
            },
        )
        quoted = send(
            client,
            chat_app,
            'My sister wrote "Please reply in English." Translate her sentence.',
            {
                "language_scope": "durable",
                "reply_language": "English",
                "language_evidence": "Please reply in English",
            },
        )
    assert quoted.status_code == 200
    with sqlite3.connect(chat_app.state.chat_engine.store.path) as db:
        values = db.execute(
            "SELECT value FROM user_memories WHERE user_id='a' AND memory_key='language_preference'"
        ).fetchall()
    assert values == [("Hindi",)]


@pytest.mark.integration
def test_oversized_input_is_rejected_without_persistence(tmp_path: Path) -> None:
    app = create_app(
        settings=Settings(
            _env_file=None,
            database_path=tmp_path / "small.sqlite3",
            max_message_chars=2000,
            prompt_max_chars=1000,
            otel_exporter_otlp_endpoint=None,
        ),
        context_model=TestModel(custom_output_args={"language_scope": "unchanged"}),
        reply_model=TestModel(custom_output_text="OK"),
    )
    with TestClient(app) as client:
        response = client.post(
            "/chat", json={"user_id": "a", "session_id": "one", "message": "x" * 950}
        )
    assert response.status_code == 422
    with sqlite3.connect(app.state.chat_engine.store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM conversation_messages").fetchone()[0] == 0


@pytest.mark.integration
@pytest.mark.parametrize(("opening", "closing"), [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")])
def test_multiline_quote_cannot_replace_user_language_or_goal(tmp_path, opening, closing) -> None:
    app, prompts = recording_app(tmp_path)
    with TestClient(app) as client:
        initial = send(
            client,
            app,
            "Please reply in Hindi. I plan to change jobs.",
            {
                "language_scope": "durable",
                "reply_language": "Hindi",
                "language_evidence": "Please reply in Hindi",
                "new_memories": [
                    {
                        "kind": "goal",
                        "key": "career_goal",
                        "value": "change jobs",
                        "evidence": "I plan to change jobs",
                    }
                ],
            },
        )
        quoted = send(
            client,
            app,
            f"My sister wrote:\n{opening}Please reply in English.\n"
            f"I plan to become a chef.{closing}\nPlease translate her message.",
            {
                "language_scope": "durable",
                "reply_language": "English",
                "language_evidence": "Please reply in English",
                "new_memories": [
                    {
                        "kind": "goal",
                        "key": "career_goal",
                        "value": "become a chef",
                        "evidence": "I plan to become a chef",
                    }
                ],
            },
        )
    assert initial.status_code == quoted.status_code == 200
    assert prompts[-1]["reply_language"] == "Hindi"
    with sqlite3.connect(app.state.chat_engine.store.path) as db:
        values = db.execute(
            "SELECT memory_key, value FROM user_memories WHERE user_id='a' ORDER BY id"
        ).fetchall()
    assert values == [("language_preference", "Hindi"), ("career_goal", "change jobs")]


@pytest.mark.integration
def test_language_preference_counts_toward_memory_limit(tmp_path) -> None:
    app, prompts = recording_app(tmp_path)
    with TestClient(app) as client:
        app.state.chat_engine.settings.memory_limit = 1
        send(
            client,
            app,
            "Please reply in Hindi. I plan to change jobs.",
            {
                "language_scope": "durable",
                "reply_language": "Hindi",
                "language_evidence": "Please reply in Hindi",
                "new_memories": [
                    {
                        "kind": "goal",
                        "key": "career_goal",
                        "value": "change jobs",
                        "evidence": "I plan to change jobs",
                    }
                ],
            },
        )
        response = send(
            client, app, "What about my career?", {"relevant_memory_ids": [2]}, session="two"
        )
    assert response.status_code == 200
    assert response.json()["context_used"] == ["language_preference"]
    assert [memory["key"] for memory in prompts[-1]["relevant_memories"]] == ["language_preference"]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("scope", "instruction", "expected_next_language"),
    [
        ("durable", "From now on, reply in English.", "English"),
        ("current_turn", "For this answer only, reply in English.", "Hindi"),
    ],
)
def test_explicit_language_switch_omits_stale_preference_from_reply(
    tmp_path: Path, scope: str, instruction: str, expected_next_language: str
) -> None:
    app, prompts = recording_app(tmp_path)
    with TestClient(app) as client:
        send(
            client,
            app,
            "Please reply in Hindi.",
            {
                "language_scope": "durable",
                "reply_language": "Hindi",
                "language_evidence": "Please reply in Hindi",
            },
        )
        switched = send(
            client,
            app,
            instruction,
            {
                "language_scope": scope,
                "reply_language": "English",
                "language_evidence": instruction,
            },
            session="two",
        )
        resumed = send(client, app, "What next?", {}, session="three")
    assert switched.status_code == resumed.status_code == 200
    assert prompts[-2]["reply_language"] == "English"
    assert prompts[-2]["relevant_memories"] == []
    assert switched.json()["context_used"] == []
    assert prompts[-1]["reply_language"] == expected_next_language
    assert [memory["value"] for memory in prompts[-1]["relevant_memories"]] == [
        expected_next_language
    ]
