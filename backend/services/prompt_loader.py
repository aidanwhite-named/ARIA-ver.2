"""
프롬프트 파일 로더

backend/prompts/ 폴더의 .txt 파일에서 프롬프트를 읽습니다.
파일을 편집하면 reload_all() 호출 후 바로 반영됩니다.

사용법:
  load_prompt("system_report_base.txt")            # 정적 프롬프트
  render_prompt("prompt_phase1_main.txt", x=val)   # ${x} 자리표시자 치환
"""
from __future__ import annotations
import logging
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent.parent / "prompts"
_cache: dict[str, str] = {}


def load_prompt(name: str, fallback: str = "") -> str:
    """파일에서 프롬프트를 읽어 반환 (캐시됨). 파일 없으면 fallback 반환."""
    if name not in _cache:
        path = _PROMPT_DIR / name
        if path.exists():
            _cache[name] = path.read_text(encoding="utf-8")
            logger.debug(f"Prompt loaded: {name}")
        else:
            logger.warning(f"Prompt file not found: {path}")
            return fallback
    return _cache[name]


def render_prompt(name: str, fallback: str = "", **kwargs) -> str:
    """파일에서 프롬프트를 읽어 ${variable} 자리표시자를 치환하여 반환."""
    template_str = load_prompt(name, fallback)
    if not template_str:
        return fallback
    return Template(template_str).safe_substitute(**kwargs)


def reload_all() -> None:
    """캐시 초기화 — 다음 접근 시 파일에서 새로 읽습니다."""
    _cache.clear()
    logger.info("Prompt cache cleared.")
