# Backend

Two modular Python packages:

- **`ai/`** (`klg_ai`) — the AI engine. Pure Python, **no web dependencies**:
  loads the static map, builds the prerequisite graph, computes per-learner
  activation, status, and gaps. Testable in isolation.
- **`server/`** (`klg_server`) — the server app. A thin FastAPI layer that
  imports `klg_ai` and exposes it over HTTP.

The split keeps the AI core reusable and lets the server be swapped/ported
(e.g. fronted by Subturtle's Node server at ship time) without touching the engine.

## Setup

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -e backend/ai            # engine (+ networkx)
pip install -e "backend/server[dev]" # server (+ fastapi, uvicorn, pytest, httpx)
```

## Test

```bash
pytest backend/ai backend/server -q
```

## Run the API

```bash
uvicorn klg_server.main:app --reload --port 8000
```

- `GET /api/health`
- `GET /api/map` — the static syntax map
- `GET /api/learner/{id}/status` — per-learner node statuses (try `demo`)
- Interactive docs at `http://localhost:8000/docs`

## Layout

```
backend/
  ai/
    klg_ai/  loader · graph · activation · status · gaps
    tests/
  server/
    klg_server/  main · deps · routers/(map, learner) · schemas/
    tests/
```
