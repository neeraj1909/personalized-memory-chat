# MyNaksh personalized chat

A small FastAPI chat service for the MyNaksh AI/ML Engineer assignment. It saves
session messages in SQLite, keeps selected user facts across sessions, and sends
only relevant stored facts to the reply model. There is no frontend or astrology
calculation engine.

## Run it

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required. From this directory:

```bash
uv sync
cp -n .env.example .env
```

Set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in your private `.env`, then run:

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`LLM_BASE_URL` is an OpenAI-compatible API base URL, usually ending in `/v1`.
`LANGUAGE_MODEL` optionally chooses a cheaper model for the context decision;
it defaults to `LLM_MODEL`. `.env` is ignored by Git and is never needed for
offline tests.

A local SQLite file is created at `DATABASE_PATH` (default `memory.db`). Use a
file path, including for tests; SQLite `:memory:` is deliberately unsupported
because this app opens short connections across a turn. Existing files with the
assignment's `conversation_messages` and `user_memories` tables remain readable.

## Chat API

```bash
curl -sS http://127.0.0.1:8000/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"user-123","session_id":"first","message":"I plan to switch jobs next year. Please reply in Hindi."}'
```

An example response shape (text and IDs vary):

```json
{
  "request_id": "7eb48f9aab53484bbeb2654998c258ce",
  "message_id": 1,
  "response_id": 2,
  "user_id": "user-123",
  "session_id": "first",
  "response": "अगले साल नौकरी बदलने के लिए अपनी तैयारी की योजना बनाइए।",
  "context_used": []
}
```

A later request in a new session can recall the goal:

```bash
curl -sS http://127.0.0.1:8000/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"user-123","session_id":"second","message":"What do you remember about my career plan?"}'
```

`context_used` lists keys of **previously stored** memories selected for that
answer, such as `career_goal` and `language_preference`; it never includes their
values. The current message is not counted as recalled context. `X-Request-ID`
repeats the JSON request ID. Blank or oversized messages return 422, missing
provider settings return 503, and provider failures return a generic 502.

## How a turn works

1. FastAPI validates the request. SQLite loads the current user's latest
   language preference, a bounded catalogue of memory IDs/kinds/keys, and a
   few prior user messages from this session.
2. A PydanticAI context agent reads the full current message and that small
   catalogue. Its typed decision proposes useful new facts, relevant memory
   IDs, and any explicit response-language choice. The catalogue contains no
   stored fact values; the language preference is the one exception needed to
   resolve a switch. Evidence inside paired quotation marks is rejected
   before a proposed fact or language switch is persisted.
3. The app accepts only IDs from the current user's catalogue, loads their
   values, and gives them to the PydanticAI reply agent. The reply also sees
   complete recent messages from this user's current session and the full
   current message once. A short clarification replaces the reply when the
   requested language is genuinely ambiguous.
4. SQLite stores the accepted user message and assistant response. Useful
   goals, preferences, interests, and facts are stored under stable keys per
   user. The latest explicit durable language choice is effective across
   sessions; a one-turn override does not change it.

This design uses two model calls for a normal turn. That costs more time and
provider usage than simple keyword rules, but lets one structured decision
handle Hindi, Hinglish, selective memory, and response-language instructions
without a large language-specific regex parser. The context model sees up to
`MEMORY_CANDIDATES` labels (default 40), so a fact outside that shortlist may
not be recalled. The reply model sees only selected values. The configured
`PROMPT_MAX_CHARS` (default 256000) is an **operational input guard**, not a
claim about any model's token window. The app keeps complete older messages
that fit and rejects a current message that cannot fit; it does not silently
clip accepted text.

`app/main.py` owns HTTP, model construction, and lifespan cleanup;
`app/chat.py` owns the typed agent decision and turn sequence;
`app/storage.py` owns SQLite queries. `app/config.py` holds private settings.
The provider is replaceable by injecting PydanticAI models into `create_app`;
all routine tests do this with `TestModel` or `FunctionModel` and make no paid
requests. Model calls run outside SQLite transactions.

## Verify

```bash
uv run ruff format --check app tests
uv run ruff check app tests
uv run pytest -q
uv lock --check
```

The public tests cover HTTP, cross-session Hindi/Hinglish recall, quoted
third-party text, complete follow-up history, irrelevant-memory exclusion,
user isolation, language switching, ambiguous requests, safe errors, and input
limits. Offline tests verify workflow and prompt boundaries; they do not prove
that a live model will classify every phrase or write every answer correctly.

Docker Compose and Phoenix are optional for local trace inspection:

```bash
docker compose up --build -d
```

To use a separate context model, set `LANGUAGE_MODEL` when starting Compose:

```bash
LANGUAGE_MODEL=your-context-model docker compose up --build -d
```

When unset, the context agent uses the configured `LLM_MODEL`.

The app listens on `127.0.0.1:8000` and Phoenix on `127.0.0.1:6006`. Set
`OTEL_EXPORTER_OTLP_ENDPOINT` to an OTLP `/v1/traces` URL when running outside
Compose. Traces omit message text, memory values, user/session IDs, and
credentials. Compose uses named data volumes; it does not use the repository's
`memory.db`.
