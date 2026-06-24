import json
import logging
from pathlib import Path

from fastapi import APIRouter
from backend.models.schemas import ModelListResponse, Settings
from backend.services.ai_engine import get_engine_status
from backend.services.rag_retriever import get_rag_runtime_status

router = APIRouter()
logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path("backend/settings.json")


def _load() -> Settings:
    if _SETTINGS_PATH.exists():
        try:
            return Settings(**json.loads(_SETTINGS_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return Settings()


def _save(s: Settings) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(s.model_dump_json(indent=2), encoding="utf-8")


@router.get("", response_model=Settings)
async def get_settings():
    return _load()


@router.post("", response_model=Settings)
async def save_settings(settings: Settings):
    _save(settings)
    return settings


@router.get("/models", response_model=ModelListResponse)
async def get_models():
    return ModelListResponse()


@router.get("/status")
async def get_status():
    s = _load()
    status = get_engine_status(s)
    status["rag"] = get_rag_runtime_status()
    return status
