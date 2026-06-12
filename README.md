# linguistic-knowledge-tree

An AI-driven personal knowledge tracing system that models, visualizes, and dynamically updates a learner's English grammar mastery using a static CEFR syntax map, per-learner activation, and graph reasoning.

See [`docs/roadmap.md`](docs/roadmap.md) for the phased plan and [`paper-work/`](paper-work/) for the thesis.

## Structure

```
map/          static English syntax map (data) + validator        — the fixed, global graph
backend/
  ai/         klg_ai   — the AI engine (pure Python, no web deps): loader, graph, activation, status, gaps
  server/     klg_server — FastAPI app exposing the engine over HTTP
frontend/     Vue 3 + Vite + TypeScript + Pinia + Cytoscape.js     — the visualization
docs/         specs, roadmap, design notes
paper-work/   thesis
```

The map is the source of truth (same for every learner); the `ai` engine computes a per-learner overlay (known / interior-gap / frontier / further); the `server` serves it; the `frontend` renders it.

## Quickstart

**Backend** (Python 3.10+):

```bash
python3 -m venv backend/.venv && source backend/.venv/bin/activate
pip install -e backend/ai -e "backend/server[dev]"
pytest backend/ai backend/server -q                               # run tests
uvicorn klg_server.main:app --reload --port 8000 --env-file .env  # run API  (docs at /docs)
```

**Frontend** (Node 18+), in a second terminal:

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173  (expects the API on :8000)
```

**Validate the map** after editing it:

```bash
python3 map/validate.py
```
