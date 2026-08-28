# Hermes Sidecar Setup

This project expects a separately managed Hermes Agent installation. Follow the
current official documentation rather than adding Hermes to this application's
`pyproject.toml`.

## Setup

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
hermes model
```

Select Google AI Studio/Gemini and verify ordinary chat first. Then put secrets
in the Hermes environment (normally `~/.hermes/.env`):

```env
GOOGLE_API_KEY=...
API_SERVER_ENABLED=true
API_SERVER_KEY=generate-a-strong-secret
```

Start the process:

```bash
hermes gateway run
```

The application defaults to `http://127.0.0.1:8642/v1`. Set the matching values
in the application's local `.env`:

```env
HERMES_API_URL=http://127.0.0.1:8642/v1
HERMES_API_KEY=generate-a-strong-secret
HERMES_MODEL_NAME=hermes-agent
```

## Security Rules

- Keep the API on loopback or a private service network.
- Never expose the bearer key in logs, screenshots, or Git.
- Restrict Hermes toolsets and command approvals for the deployment environment.
- Do not run Hermes' Telegram adapter with this application's bot token. Two
  polling consumers for one token conflict.
- Run `hermes doctor` after provider upgrades or model changes.

## Verification

```bash
python scripts/smoke.py hermes
```

The smoke check calls health first and performs one simple turn only when
`HERMES_API_KEY` is configured.

Official references:

- <https://hermes-agent.nousresearch.com/docs/guides/google-gemini>
- <https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server>
- <https://hermes-agent.nousresearch.com/docs/developer-guide/programmatic-integration>
