# Chatbot A-Z

Stage 1 is a FastAPI chat service with a Google ADK orchestrator and a model gateway.
It stores conversation sessions in a database. MCP, RAG, specialist agents, and A2A
are planned for later stages and are not wired into this release.

## Run locally

Requires Python 3.11+ and `uv`.

```bash
uv sync
cp .env.example .env
```

Edit `.env`: set a random `CHAT_API_KEY` of at least 24 characters and a real
`GOOGLE_API_KEY`. Then start the API. Restart it whenever you change `.env`;
the server reads its configuration at startup.

```bash
uv run uvicorn chatbot.main:app --host 127.0.0.1 --port 8000
```

Send a message:

```bash
export CHAT_API_KEY="$(uv run python -c 'from chatbot.config import Settings; print(Settings().chat_api_key)')"
curl -s http://127.0.0.1:8000/v1/chat \
  -H "X-API-Key: $CHAT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"message":"Hello"}'
```

`X-API-Key` must contain `CHAT_API_KEY`. `GOOGLE_API_KEY` is used only by the
backend when it calls Gemini. To check authentication without calling a model, send
`GET /v1/models` with the same header; a 200 means the local API key matches.
Opening `/v1/models` directly in a browser sends no `X-API-Key`, so it returns 401.
For browser testing, open `/docs`, click **Authorize**, and enter `CHAT_API_KEY`.

The response includes `conversation_id`; include it in the next request to continue
the conversation. Add `"model":"flash-lite"` to select another configured model for
that request, including a follow-up in the same conversation. `GET /v1/models` lists
the allowlisted aliases. Unknown model aliases return 400 and unknown conversation IDs
return 404. `/healthz` is available without auth.
API documentation is at `/docs`.

The default aliases are `default` (`gemini-flash-latest`), `flash-lite`
(`gemini-3.5-flash-lite`), and `stable-flash` (`gemini-3.8-flash`). Set
`DEFAULT_MODEL` to change the default alias. Set `MODEL_ROUTES` to a JSON object to
replace the catalog. Each entry has `provider` (`gemini` for now) and `model`.
Only configured aliases are accepted; callers cannot supply arbitrary provider model
names or credentials. Availability and free-tier limits depend on your Gemini account.
The gateway is the place to add another provider adapter later.

`SESSION_DB_URL` defaults to a local SQLite database. Use a managed SQL database and
an appropriate async SQLAlchemy URL for multi-instance deployment. Do not expose this
service without TLS and a gateway that applies request limits. The shared API key is
intended for a trusted first-party client; user identity and access control are future
work before serving multiple independent users.

## Verify

```bash
uv run pytest
uv run ruff check .
```
