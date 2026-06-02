"""FastAPI application entry point."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import learner, map as map_router, status
from .schemas import HealthOut

app = FastAPI(title="Linguistic Knowledge Graph API", version="0.1.0")

# Vite dev server origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(map_router.router)
app.include_router(learner.router)
app.include_router(status.router)


@app.get("/api/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    return HealthOut(status="ok")
