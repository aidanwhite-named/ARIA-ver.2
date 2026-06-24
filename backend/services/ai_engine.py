from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import closing
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from typing import AsyncGenerator

from backend.models.schemas import Settings

logger = logging.getLogger(__name__)

_active_procs: set[subprocess.Popen] = set()
_active_lock = threading.Lock()

_DEFAULT_MODEL = {
    "claude": "claude-haiku-4-5-20251001",
    "openai": "gpt-5.4-mini",
    "agy": "gemini-3.5-flash",
    "gemini": "gemini-3.5-flash",
}

_BENIGN_STDERR_PATTERNS = (
    "256-color support not detected",
    "True color (24-bit) support not detected",
    "NODE_TLS_REJECT_UNAUTHORIZED",
    "Ripgrep is not available. Falling back to GrepTool.",
)

_AGY_ARG_PROMPT_LIMIT = 12_000
_AGY_PROMPT_DIR = Path("uploads") / "_agy_prompts"
_AGY_TRANSCRIPT_MAX_AGE_SECONDS = 180
_AGY_TRANSCRIPT_FLUSH_WAIT_SECONDS = 2.0
_AGY_TRUNCATED_RE = re.compile(r"<truncated\s+\d+\s+bytes>")

_cli_cache: dict[str, bool] = {}
_account_cache: dict[str, str] = {}
_status_probe_cache: dict[str, tuple[float, dict]] = {}
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SAFE_MODEL_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_UUID_RE = re.compile(
    r"^(?:bot-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

def _normalize_engine(engine: str) -> str:
    engine = (engine or "").lower()
    return "agy" if engine == "gemini" else engine


def _resolve_model(settings: Settings, agent: str) -> str:
    model = {
        "parser": settings.model_parser,
        "category": settings.model_parser,
        "compare": settings.model_compare,
        "report": settings.model_report,
    }.get(agent, "")
    engine = _normalize_engine(settings.engine)
    if model and model.strip():
        selected = model.strip()
        if engine == "claude" and selected.lower().startswith("gemini"):
            return _DEFAULT_MODEL["claude"]
        if engine == "openai" and (
            selected.lower().startswith("claude") or selected.lower().startswith("gemini")
        ):
            return _DEFAULT_MODEL["openai"]
        if engine == "agy" and selected.lower().startswith("claude"):
            return _DEFAULT_MODEL["agy"]
        return selected
    return _DEFAULT_MODEL.get(engine, _DEFAULT_MODEL["claude"])


def _decode(b: bytes) -> str:
    for enc in ("utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


def _clean_cli_stderr(stderr: str) -> str:
    lines = []
    for line in (stderr or "").splitlines():
        if any(pattern in line for pattern in _BENIGN_STDERR_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _get_fixed_home() -> Path:
    home = Path.home()
    if sys.platform == "win32":
        home_str = str(home)
        # 드라이브 문자(예: C:)나 네트워크 경로(\\)로 시작하지 않으면 SystemDrive를 명시적으로 붙여줌
        if not (home_str.startswith("\\\\") or (len(home_str) >= 2 and home_str[1] == ":")):
            drive = os.environ.get("SystemDrive", "C:")
            if not drive.endswith(":"):
                drive += ":"
            home = Path(drive + "\\") / home_str.lstrip("\\/")
    return home


def _normalize_win_path(path_str: str, fixed_home: Path) -> str:
    if not path_str:
        return ""
    if path_str.startswith("\\\\") or (len(path_str) >= 2 and path_str[1] == ":"):
        return path_str

    normalized = path_str.replace("/", "\\")
    drive = fixed_home.drive or "C:"
    home_without_drive = str(fixed_home)[len(drive):]

    if home_without_drive and normalized.lower().startswith(home_without_drive.lower()):
        return drive + normalized
    return str(fixed_home / normalized.lstrip("\\"))


def _cli_env() -> dict:
    env = os.environ.copy()
    fixed_home = _get_fixed_home()

    # Windows 환경일 경우 홈 및 앱 데이터 관련 환경 변수 경로 정규화 (C: 누락 방지)
    if sys.platform == "win32":
        # HOME 및 USERPROFILE 강제 정규화
        env["HOME"] = str(fixed_home)
        env["USERPROFILE"] = str(fixed_home)

        # HOMEDRIVE, HOMEPATH 강제 보정
        drive = fixed_home.drive or "C:"
        env["HOMEDRIVE"] = drive
        env["HOMEPATH"] = str(fixed_home.relative_to(drive + "\\")) if fixed_home.drive else str(fixed_home)
        if env["HOMEPATH"] and not env["HOMEPATH"].startswith("\\"):
            env["HOMEPATH"] = "\\" + env["HOMEPATH"]

        # APPDATA, LOCALAPPDATA 보정 (드라이브 누락 대응)
        for key in ("APPDATA", "LOCALAPPDATA"):
            val = env.get(key, "")
            if val:
                env[key] = _normalize_win_path(val, fixed_home)

    extra_paths = [
        fixed_home / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links",
        fixed_home / "AppData" / "Local" / "OpenAI" / "Codex" / "bin",
        fixed_home / "AppData" / "Local" / "agy" / "bin",
        Path(env.get("LOCALAPPDATA", "")) / "Programs" / "Antigravity" / "bin",
    ]
    existing = env.get("PATH", "")
    for p in extra_paths:
        if p.exists() and str(p) not in existing:
            existing = f"{p}{os.pathsep}{existing}"
    env["PATH"] = existing
    env.setdefault("TERM", "xterm-256color")
    env.setdefault("NO_COLOR", "1")
    env.setdefault("NODE_NO_WARNINGS", "1")
    env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    return env


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    try:
        if proc.poll() is not None:
            return
        if sys.platform == "win32":
            subprocess.run(
                f"taskkill /T /F /PID {proc.pid}",
                shell=True,
                capture_output=True,
                timeout=10,
            )
        else:
            proc.kill()
        logger.info("CLI process tree was terminated (pid=%s)", proc.pid)
    except Exception:
        logger.warning("Failed to terminate CLI process tree (pid=%s)", proc.pid, exc_info=True)


def kill_active_cli_procs() -> int:
    with _active_lock:
        procs = list(_active_procs)
    for proc in procs:
        _kill_proc_tree(proc)
    return len(procs)


def _is_gemini_unsupported_client_error(text: str) -> bool:
    lowered = (text or "").lower()
    return (
        "ineligibletiererror" in lowered
        or "unsupported_client" in lowered
        or "no longer supported for gemini code assist" in lowered
        or "migrate to the antigravity suite" in lowered
    )


def _gemini_unsupported_message() -> str:
    return (
        "현재 로그인된 Gemini/AGY CLI 계정 또는 클라이언트가 지원되지 않습니다. "
        "설정에서 엔진을 Claude로 변경하거나, Google 안내에 따라 지원되는 AGY 환경으로 다시 로그인해 주세요."
    )


def _agy_prompt_arg(full_prompt: str) -> tuple[str, Path | None]:
    if len(full_prompt) <= _AGY_ARG_PROMPT_LIMIT:
        return full_prompt, None

    _AGY_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".txt",
        prefix="prompt_",
        dir=_AGY_PROMPT_DIR,
        delete=False,
    ) as f:
        f.write(full_prompt)
        prompt_path = Path(f.name).resolve()

    return (
        "Read the complete UTF-8 prompt from the file below, then carry out "
        "that prompt exactly. Return only the final response requested by the file.\n\n"
        f"Prompt file: {prompt_path}",
        prompt_path,
    )


def _cleanup_prompt_file(path: Path | None) -> None:
    if not path:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.debug("Failed to delete temporary AGY prompt file: %s", path, exc_info=True)


def _agy_cwd() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        return str(Path.cwd())
    except Exception:
        return None


def _agy_app_data_dir() -> Path:
    return _get_fixed_home() / ".gemini" / "antigravity-cli"


def _transcript_matches_prompt(path: Path, prompt_marker: str) -> bool:
    if not prompt_marker:
        return True
    try:
        marker = prompt_marker.strip()
        if not marker:
            return True
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False

    # transcript.jsonl stores Windows paths with JSON-escaped backslashes.
    # Compare against decoded content instead of the raw JSON text so a marker
    # such as ``D:\\project\\prompt.txt`` matches its transcript entry.
    marker = marker[:500]
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = item.get("content")
        if isinstance(content, str) and marker in content:
            return True
    return False


def _latest_agy_transcript(started_at: float, prompt_marker: str = "") -> Path | None:
    brain_dir = _agy_app_data_dir() / "brain"
    if not brain_dir.exists():
        return None

    newest: tuple[float, Path] | None = None
    min_mtime = started_at - 5
    try:
        for path in brain_dir.glob("*/.system_generated/logs/transcript.jsonl"):
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size <= 0 or stat.st_mtime < min_mtime:
                continue
            if not _transcript_matches_prompt(path, prompt_marker):
                continue
            if newest is None or stat.st_mtime > newest[0]:
                newest = (stat.st_mtime, path)
    except Exception:
        logger.debug("Failed to inspect AGY transcript directory", exc_info=True)
        return None
    return newest[1] if newest else None


def _read_protobuf_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while pos < len(data) and shift < 70:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
    raise ValueError("invalid protobuf varint")


def _iter_protobuf_text_fields(data: bytes, depth: int = 0):
    """Yield UTF-8 length-delimited fields from an AGY protobuf payload."""
    if depth > 5:
        return
    pos = 0
    while pos < len(data):
        start = pos
        try:
            tag, pos = _read_protobuf_varint(data, pos)
            wire_type = tag & 0x07
            if wire_type == 0:
                _, pos = _read_protobuf_varint(data, pos)
            elif wire_type == 1:
                pos += 8
            elif wire_type == 2:
                size, pos = _read_protobuf_varint(data, pos)
                end = pos + size
                if end > len(data):
                    return
                value = data[pos:end]
                pos = end
                try:
                    yield value.decode("utf-8")
                except UnicodeDecodeError:
                    pass
                yield from _iter_protobuf_text_fields(value, depth + 1)
            elif wire_type == 5:
                pos += 4
            else:
                return
        except (IndexError, ValueError):
            return
        if pos <= start or pos > len(data):
            return


def _restore_agy_truncated_response(transcript_path: Path, response: str) -> str:
    """Restore an AGY response that transcript.jsonl shortened in the middle."""
    match = _AGY_TRUNCATED_RE.search(response)
    if not match:
        return response

    # AGY places the marker on its own line even when it removed bytes from
    # the middle of a JSON string, so the marker-adjacent newlines are not
    # part of the original response.
    prefix = response[:match.start()].rstrip()
    suffix = response[match.end():].lstrip()
    if len(prefix) < 40 or len(suffix) < 40:
        return response

    try:
        conversation_id = transcript_path.parents[2].name
        db_path = _agy_app_data_dir() / "conversations" / f"{conversation_id}.db"
        if not db_path.exists():
            return response
        with closing(sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=1)) as conn:
            rows = conn.execute(
                "SELECT step_payload FROM steps WHERE step_payload IS NOT NULL ORDER BY idx DESC"
            ).fetchall()
    except (OSError, sqlite3.Error):
        logger.debug("Failed to open AGY conversation database for %s", conversation_id, exc_info=True)
        return response

    best = ""
    for (payload,) in rows:
        if not isinstance(payload, bytes):
            continue
        for candidate in _iter_protobuf_text_fields(payload):
            if _AGY_TRUNCATED_RE.search(candidate):
                continue
            if candidate.startswith(prefix) and candidate.endswith(suffix):
                if not best or len(candidate) < len(best):
                    best = candidate
    if best:
        logger.warning(
            "Restored truncated AGY response from conversation database %s (%d -> %d chars)",
            db_path,
            len(response),
            len(best),
        )
        return best
    return response


def _iter_recent_agy_conversation_dbs(started_at: float):
    conversations_dir = _agy_app_data_dir() / "conversations"
    if not conversations_dir.exists():
        return

    min_mtime = started_at - 5
    newest: list[tuple[float, Path]] = []
    try:
        for path in conversations_dir.glob("*.db"):
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size <= 0 or stat.st_mtime < min_mtime:
                continue
            newest.append((stat.st_mtime, path))
    except Exception:
        logger.debug("Failed to inspect AGY conversation directory", exc_info=True)
        return

    for _, path in sorted(newest, key=lambda item: item[0], reverse=True):
        yield path


def _clean_agy_payload_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _is_agy_internal_text(text: str, prompt_marker: str = "") -> bool:
    text = _clean_agy_payload_text(text)
    if not text:
        return True

    if _UUID_RE.fullmatch(text):
        return True
    if text in {"sessionID", "Confirmation Of Request"}:
        return True
    if text.startswith(("file://", "http://", "https://")):
        return True
    if re.fullmatch(r"-?\d{8,}", text):
        return True
    if len(text) >= 2 and any(ord(ch) < 32 and ch not in "\n\t" for ch in text):
        return True
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", text):
        return True
    if re.search(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"):
        return True

    marker = (prompt_marker or "").strip()[:500]
    if marker and (text == marker or marker in text):
        return True
    return False


def _is_complete_json_response(text: str) -> bool:
    """Return True when the whole AGY candidate is one JSON object or array."""
    candidate = _clean_agy_payload_text(text)
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(parsed, (list, dict))


def _is_complete_json_array(text: str) -> bool:
    candidate = _clean_agy_payload_text(text)
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return isinstance(json.loads(candidate), list)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False

def _select_agy_response_candidate(candidates: list[str]) -> str:
    """Prefer a complete structured/final response over longer reasoning text."""
    cleaned = [_clean_agy_payload_text(item) for item in candidates]
    cleaned = [item for item in cleaned if item]
    if not cleaned:
        return ""

    arrays = [item for item in cleaned if _is_complete_json_array(item)]
    if arrays:
        return max(arrays, key=len)

    counts = Counter(cleaned)
    structured = [item for item in cleaned if _is_complete_json_response(item)]
    repeated_structured = [item for item in structured if counts[item] > 1]
    if repeated_structured:
        return max(repeated_structured, key=len)
    if structured:
        return max(structured, key=len)

    repeated = [item for item in cleaned if counts[item] > 1 and len(item) >= 40]
    if repeated:
        return max(repeated, key=len)

    return max(cleaned, key=len)

def _conversation_db_matches_prompt(conn: sqlite3.Connection, prompt_marker: str) -> bool:
    marker = (prompt_marker or "").strip()[:500]
    if not marker:
        return True
    try:
        rows = conn.execute(
            "SELECT step_payload FROM steps WHERE step_payload IS NOT NULL ORDER BY idx ASC"
        ).fetchall()
    except sqlite3.Error:
        return False

    for (payload,) in rows:
        if not isinstance(payload, bytes):
            continue
        for text in _iter_protobuf_text_fields(payload):
            if marker in text:
                return True
    return False


def _read_agy_conversation_response(started_at: float, prompt_marker: str = "") -> str:
    for db_path in _iter_recent_agy_conversation_dbs(started_at) or []:
        try:
            with closing(sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=1)) as conn:
                if not _conversation_db_matches_prompt(conn, prompt_marker):
                    continue
                rows = conn.execute(
                    """
                    SELECT idx, step_payload
                    FROM steps
                    WHERE step_type = 15 AND step_payload IS NOT NULL
                    ORDER BY idx DESC
                    """
                ).fetchall()
        except (OSError, sqlite3.Error):
            logger.debug("Failed to read AGY conversation database: %s", db_path, exc_info=True)
            continue

        for _idx, payload in rows:
            if not isinstance(payload, bytes):
                continue
            candidates: list[str] = []
            for text in _iter_protobuf_text_fields(payload):
                cleaned = _clean_agy_payload_text(text)
                if not _is_agy_internal_text(cleaned, prompt_marker):
                    candidates.append(cleaned)
            if candidates:
                response = _select_agy_response_candidate(candidates)
                logger.warning("AGY CLI returned empty stdout; recovered response from %s", db_path)
                return response

    return ""

def _read_agy_transcript_response(started_at: float, prompt_marker: str = "") -> str:
    deadline = time.monotonic() + _AGY_TRANSCRIPT_FLUSH_WAIT_SECONDS
    while True:
        path = _latest_agy_transcript(started_at, prompt_marker)
        if path:
            try:
                if time.time() - path.stat().st_mtime <= _AGY_TRANSCRIPT_MAX_AGE_SECONDS:
                    contents = path.read_text(encoding="utf-8", errors="replace")
                else:
                    contents = ""
            except Exception:
                logger.debug("Failed to read AGY transcript: %s", path, exc_info=True)
                contents = ""

            responses: list[str] = []
            for line in contents.splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("source") != "MODEL":
                    continue
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    responses.append(content.strip())
            if responses:
                response = _restore_agy_truncated_response(path, responses[-1])
                logger.warning("AGY CLI returned empty stdout; recovered response from %s", path)
                return response

        db_response = _read_agy_conversation_response(started_at, prompt_marker)
        if db_response:
            return db_response
        if time.monotonic() >= deadline:
            return ""
        time.sleep(0.1)


def _build_cmd(
    engine: str,
    model: str,
    full_prompt: str,
    web_search: bool = False,
):
    env = _cli_env()
    if engine == "claude":
        tools = ' --allowedTools "WebSearch,WebFetch"' if web_search else ""
        return f"claude -p --model {model}{tools} -", True, full_prompt.encode("utf-8"), subprocess.PIPE, None, env

    if engine == "agy":
        agy_bin = shutil.which("agy", path=env.get("PATH", ""))
        if not agy_bin:
            raise RuntimeError("AGY CLI를 찾을 수 없습니다. AGY를 설치했는지와 PATH 설정을 확인해 주세요.")
        prompt_arg, prompt_file = _agy_prompt_arg(full_prompt)
        cmd = [agy_bin, "--model", model]
        if prompt_file:
            cmd.extend(["--add-dir", str(prompt_file.parent)])
        if web_search:
            cmd.append("--dangerously-skip-permissions")
        cmd.extend(["--print-timeout", "10m0s"])
        cmd.extend(["--print", prompt_arg])
        return cmd, False, None, subprocess.DEVNULL, prompt_file, env

    if engine == "openai":
        codex_bin = shutil.which("codex", path=env.get("PATH", ""))
        if not codex_bin:
            raise RuntimeError("OpenAI Codex CLI를 찾을 수 없습니다. Codex 설치 및 PATH 설정을 확인해 주세요.")
        cmd = [
            codex_bin,
            "exec",
            "--model",
            model,
            "--skip-git-repo-check",
            "-",
        ]
        return cmd, False, full_prompt.encode("utf-8"), subprocess.PIPE, None, env

    raise ValueError(f"지원하지 않는 엔진입니다: {engine} (claude, openai 또는 agy만 사용 가능)")


def _format_empty_response_error(engine: str, stderr: str, returncode: int) -> str:
    detail = _clean_cli_stderr(stderr)
    if detail:
        return f"{engine} CLI가 응답 본문을 반환하지 않았습니다. CLI 메시지: {detail[:1000]}"
    return (
        f"{engine} CLI가 빈 응답을 반환했습니다. 종료 코드는 {returncode}입니다. "
        "CLI 로그인 상태, 모델명, 사용량 제한 또는 권한 프롬프트가 없는지 확인해 주세요."
    )


async def call_ai(
    prompt: str,
    system: str,
    settings: Settings,
    agent: str = "default",
    web_search: bool = False,
) -> str:
    engine = _normalize_engine(settings.engine)
    if engine not in ("claude", "openai", "agy"):
        raise ValueError(f"지원하지 않는 엔진입니다: {engine} (claude, openai 또는 agy만 사용 가능)")
    return await _cli_run(engine, prompt, system, _resolve_model(settings, agent), web_search)


async def call_ai_streaming(
    prompt: str,
    system: str,
    settings: Settings,
    agent: str = "default",
) -> AsyncGenerator[str, None]:
    engine = _normalize_engine(settings.engine)
    if engine not in ("claude", "openai", "agy"):
        raise ValueError(f"지원하지 않는 엔진입니다: {engine}")
    async for chunk in _cli_run_streaming(engine, prompt, system, _resolve_model(settings, agent)):
        yield chunk


async def _cli_run(
    engine: str,
    prompt: str,
    system: str,
    model: str,
    web_search: bool = False,
) -> str:
    full_prompt = f"{system}\n\n{prompt}"
    loop = asyncio.get_running_loop()
    holder: dict = {}

    def _sync_run():
        prompt_files: list[Path] = []

        def _run_once():
            started_at = time.time()
            cmd, shell, stdin_payload, stdin_pipe, prompt_file, env = _build_cmd(
                engine, model, full_prompt, web_search
            )
            if prompt_file:
                prompt_files.append(prompt_file)
            prompt_marker = str(prompt_file) if prompt_file else full_prompt.strip()[:500]
            proc = subprocess.Popen(
                cmd,
                shell=shell,
                stdin=stdin_pipe,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=_agy_cwd() if engine == "agy" else None,
            )
            holder["proc"] = proc
            with _active_lock:
                _active_procs.add(proc)
            try:
                try:
                    stdout_b, stderr_b = proc.communicate(input=stdin_payload, timeout=600)
                except subprocess.TimeoutExpired:
                    _kill_proc_tree(proc)
                    raise RuntimeError(f"{engine} CLI 응답 시간이 초과되었습니다.") from None
            finally:
                with _active_lock:
                    _active_procs.discard(proc)
            stdout = _decode(stdout_b or b"").strip()
            stderr = _decode(stderr_b or b"").strip()
            if engine == "agy" and proc.returncode == 0 and not stdout:
                stdout = _read_agy_transcript_response(started_at, prompt_marker)
            return proc.returncode, stdout, stderr

        try:
            returncode, stdout, stderr = _run_once()
            stderr_for_user = _clean_cli_stderr(stderr)

            if returncode == 0 and stdout:
                return stdout

            if returncode == 0 and not stdout:
                raise RuntimeError(_format_empty_response_error(engine, stderr, returncode))

            logger.error("%s CLI failed with rc=%s:\n%s", engine, returncode, stderr)
            if engine == "agy" and _is_gemini_unsupported_client_error(stderr_for_user or stderr):
                raise RuntimeError(_gemini_unsupported_message())
            if stdout and not stderr_for_user:
                logger.warning("%s CLI returned rc=%s with usable stdout only.", engine, returncode)
                return stdout

            lowered = stderr_for_user.lower()
            cli_not_found = (
                "is not recognized" in lowered
                or "not found" in lowered
                or "no such file or directory" in lowered
                or (not stderr_for_user and returncode == 127)
            )
            if cli_not_found:
                raise RuntimeError(f"{engine} CLI를 찾을 수 없습니다. 설치 및 PATH 설정을 확인해 주세요.")

            model_not_found = (
                "model not found" in lowered
                or "unknown model" in lowered
                or "invalid model" in lowered
                or "modelnotfounderror" in lowered
                or "requested entity was not found" in lowered
            )
            if model_not_found:
                raise RuntimeError(f"선택한 모델을 사용할 수 없습니다: '{model}'. 설정에서 모델명을 확인해 주세요.")

            if "authentication cancelled" in lowered or "exitcode: 130" in lowered or "[Y/n]" in stdout:
                raise RuntimeError(
                    f"{engine} CLI 인증이 필요합니다. 터미널에서 `{engine} \"hello\"`를 실행해 "
                    "인증을 완료한 뒤 다시 시도해 주세요."
                )

            detail = stderr_for_user or stdout or f"종료 코드 {returncode}"
            raise RuntimeError(f"{engine} CLI 실행 실패: {detail[:2000]}")
        finally:
            for prompt_file in prompt_files:
                _cleanup_prompt_file(prompt_file)

    try:
        return await loop.run_in_executor(None, _sync_run)
    except asyncio.CancelledError:
        proc = holder.get("proc")
        if proc is not None:
            _kill_proc_tree(proc)
        raise


async def _cli_run_streaming(
    engine: str,
    prompt: str,
    system: str,
    model: str,
) -> AsyncGenerator[str, None]:
    full_prompt = f"{system}\n\n{prompt}"
    cmd, shell, stdin_payload, stdin_pipe, prompt_file, env = _build_cmd(engine, model, full_prompt)
    started_at = time.time()
    prompt_marker = str(prompt_file) if prompt_file else full_prompt.strip()[:500]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    holder: dict = {}

    def _reader_thread() -> None:
        import codecs

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                shell=shell,
                stdin=stdin_pipe,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=_agy_cwd() if engine == "agy" else None,
            )
            holder["proc"] = proc
            with _active_lock:
                _active_procs.add(proc)
            if stdin_payload is not None and proc.stdin is not None:
                proc.stdin.write(stdin_payload)
                proc.stdin.close()

            while True:
                raw = proc.stdout.read(256)
                if not raw:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        asyncio.run_coroutine_threadsafe(queue.put(("chunk", tail)), loop)
                    break
                text = decoder.decode(raw)
                if text:
                    asyncio.run_coroutine_threadsafe(queue.put(("chunk", text)), loop)

            stderr_data = proc.stderr.read()
            returncode = proc.wait()
            asyncio.run_coroutine_threadsafe(queue.put(("done", returncode, stderr_data)), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(exc))), loop)
        finally:
            if proc is not None:
                with _active_lock:
                    _active_procs.discard(proc)
            _cleanup_prompt_file(prompt_file)

    threading.Thread(target=_reader_thread, daemon=True).start()
    emitted_any = False
    try:
        while True:
            item = await queue.get()
            kind = item[0]
            if kind == "chunk":
                emitted_any = emitted_any or bool(item[1])
                yield item[1]
            elif kind == "done":
                returncode, stderr_data = item[1], item[2]
                stderr_text = _decode(stderr_data) if stderr_data else ""
                if returncode == 0 and not emitted_any:
                    if engine == "agy":
                        logger.warning("AGY CLI streaming returned empty stdout; recovering stored response.")
                        fallback = await loop.run_in_executor(
                            None,
                            _read_agy_transcript_response,
                            started_at,
                            prompt_marker,
                        )
                        if fallback:
                            yield fallback
                            break
                    raise RuntimeError(_format_empty_response_error(engine, stderr_text, returncode))
                if returncode != 0:
                    stderr_for_user = _clean_cli_stderr(stderr_text)
                    logger.error("%s CLI streaming failed with rc=%s:\n%s", engine, returncode, stderr_text)
                    if engine == "agy" and _is_gemini_unsupported_client_error(stderr_for_user or stderr_text):
                        raise RuntimeError(_gemini_unsupported_message())
                    raise RuntimeError(f"{engine} CLI 실행 실패: {(stderr_for_user or f'종료 코드 {returncode}')[:1000]}")
                break
            elif kind == "error":
                raise RuntimeError(item[1])
    finally:
        proc = holder.get("proc")
        if proc is not None and proc.poll() is None:
            _kill_proc_tree(proc)


def check_cli_available(engine: str) -> bool:
    engine = _normalize_engine(engine)
    if engine in _cli_cache:
        return _cli_cache[engine]
    try:
        if engine == "agy":
            available = shutil.which("agy", path=_cli_env().get("PATH", "")) is not None
        elif engine == "openai":
            available = shutil.which("codex", path=_cli_env().get("PATH", "")) is not None
        else:
            result = subprocess.run(
                f"{engine} --version",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
                env=_cli_env(),
            )
            available = result.returncode == 0
        _cli_cache[engine] = available
    except Exception:
        _cli_cache[engine] = False
    return _cli_cache[engine]


def _safe_model_name(model: str, fallback: str) -> str:
    model = (model or "").strip()
    return model if _SAFE_MODEL_RE.match(model) else fallback


def _probe_cli_ready(engine: str, model: str) -> dict:
    engine = _normalize_engine(engine)
    model = _safe_model_name(model, _DEFAULT_MODEL.get(engine, _DEFAULT_MODEL["claude"]))
    cache_key = f"{engine}:{model}"
    now = time.monotonic()
    cached = _status_probe_cache.get(cache_key)
    if cached and now - cached[0] < 300:
        return cached[1]

    try:
        if engine == "claude":
            cmd, shell, stdin_payload, _stdin_pipe, _prompt_file, env = _build_cmd(engine, model, "Say OK")
            input_text = stdin_payload.decode("utf-8")
        elif engine == "openai":
            cmd, shell, stdin_payload, _stdin_pipe, _prompt_file, env = _build_cmd(engine, model, "Say OK")
            input_text = stdin_payload.decode("utf-8")
        elif engine == "agy":
            cmd, shell, _stdin_payload, _stdin_pipe, _prompt_file, env = _build_cmd(engine, model, "Say OK")
            input_text = None
        else:
            probe = {
                "status": "not_configured",
                "label": "지원하지 않는 엔진",
                "detail": f"지원하지 않는 엔진입니다: {engine}",
            }
            _status_probe_cache[cache_key] = (now, probe)
            return probe
    except Exception as exc:
        probe = {"status": "not_configured", "label": "CLI 미설치", "detail": str(exc)}
        _status_probe_cache[cache_key] = (now, probe)
        return probe

    try:
        started_at = time.time()
        result = subprocess.run(
            cmd,
            shell=shell,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=45,
            env=env,
            cwd=_agy_cwd() if engine == "agy" else None,
        )
    except subprocess.TimeoutExpired:
        probe = {
            "status": "probe_timeout",
            "label": "CLI 응답 지연",
            "detail": "CLI 설치는 확인됐지만 테스트 호출이 제한 시간 안에 끝나지 않았습니다.",
        }
        _status_probe_cache[cache_key] = (now, probe)
        return probe
    except Exception as exc:
        probe = {"status": "probe_failed", "label": "CLI 호출 실패", "detail": str(exc)}
        _status_probe_cache[cache_key] = (now, probe)
        return probe

    stderr = _clean_cli_stderr(result.stderr or "")
    stdout = (result.stdout or "").strip()
    if engine == "agy" and result.returncode == 0 and not stdout:
        stdout = _read_agy_transcript_response(started_at, "Say OK")
    if result.returncode == 0 and stdout:
        probe = {"status": "cli_ready", "label": "CLI 호출 가능", "detail": ""}
    elif result.returncode == 0 and not stdout:
        probe = {
            "status": "probe_failed",
            "label": "CLI 빈 응답",
            "detail": _format_empty_response_error(engine, result.stderr or "", result.returncode),
        }
    elif engine == "agy" and _is_gemini_unsupported_client_error(stderr):
        probe = {"status": "auth_error", "label": "Gemini/AGY 계정 오류", "detail": _gemini_unsupported_message()}
    elif "authentication cancelled" in stderr.lower() or "[Y/n]" in stdout:
        probe = {
            "status": "auth_error",
            "label": "CLI 인증 필요",
            "detail": f"터미널에서 `{engine} \"hello\"`를 실행해 인증을 완료해 주세요.",
        }
    else:
        probe = {
            "status": "probe_failed",
            "label": "CLI 호출 실패",
            "detail": (stderr or stdout or f"종료 코드 {result.returncode}")[:1000],
        }

    _status_probe_cache[cache_key] = (now, probe)
    return probe


def _candidate_account_paths(engine: str) -> list[Path]:
    home = _get_fixed_home()

    appdata_str = _normalize_win_path(os.environ.get("APPDATA", ""), home)
    appdata = Path(appdata_str) if appdata_str else home / "AppData" / "Roaming"

    local_appdata_str = _normalize_win_path(os.environ.get("LOCALAPPDATA", ""), home)
    local_appdata = Path(local_appdata_str) if local_appdata_str else home / "AppData" / "Local"
    if engine == "claude":
        return [
            home / ".claude.json",
            home / ".claude",
            appdata / "Claude",
            appdata / "claude",
            local_appdata / "Claude",
            local_appdata / "claude",
        ]
    if engine == "openai":
        return [
            home / ".codex",
            home / ".codex.json",
            appdata / "OpenAI" / "Codex",
            appdata / "OpenAI",
            local_appdata / "OpenAI" / "Codex",
            local_appdata / "OpenAI",
        ]
    if engine in ("agy", "gemini"):
        return [
            home / ".antigravity",
            appdata / "Antigravity",
            appdata / "antigravity",
            local_appdata / "Antigravity",
            local_appdata / "antigravity",
            home / ".gemini" / "antigravity-cli",
            home / ".gemini",
            appdata / "Gemini",
            appdata / "gemini",
            local_appdata / "Gemini",
            local_appdata / "gemini",
            local_appdata / "Google" / "GeminiCLI",
        ]
    return []


def _email_from_json_value(value) -> str:
    if isinstance(value, dict):
        preferred_keys = ("email", "account", "user", "profile", "login")
        for key, item in value.items():
            if any(part in str(key).lower() for part in preferred_keys):
                found = _email_from_json_value(item)
                if found:
                    return found
        for item in value.values():
            found = _email_from_json_value(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _email_from_json_value(item)
            if found:
                return found
    elif isinstance(value, str):
        match = _EMAIL_RE.search(value)
        if match:
            return match.group(0)
    return ""


def _email_from_text(text: str) -> str:
    try:
        found = _email_from_json_value(json.loads(text))
        if found:
            return found
    except Exception:
        pass
    match = _EMAIL_RE.search(text or "")
    return match.group(0) if match else ""


def _decode_jwt_payload(token: str) -> dict:
    token = (token or "").strip()
    parts = token.split(".")
    if len(parts) < 2 or not parts[1]:
        return {}
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = _decode_json_bytes(__import__("base64").urlsafe_b64decode(payload + padding))
    except Exception:
        return {}
    try:
        parsed = json.loads(decoded)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _decode_json_bytes(raw: bytes) -> str:
    return _decode(raw).strip()


def _extract_codex_auth_email(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""

    direct = _email_from_json_value(payload)
    if direct:
        return direct

    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return ""

    for key in ("id_token", "access_token"):
        claims = _decode_jwt_payload(tokens.get(key, ""))
        if not claims:
            continue
        email = _email_from_json_value(claims)
        if email:
            return email
    return ""


def _iter_small_account_files(root: Path):
    if not root.exists():
        return
    if root.is_file():
        yield root
        return

    allowed_suffixes = {".json", ".toml", ".yaml", ".yml", ".txt", ".env", ""}
    skip_dirs = {"cache", "logs", "sessions", "tmp", "node_modules", "__pycache__"}
    checked = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in skip_dirs]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in allowed_suffixes:
                continue
            try:
                if path.stat().st_size > 512_000:
                    continue
            except OSError:
                continue
            yield path
            checked += 1
            if checked >= 120:
                return


def find_cli_account_email(engine: str) -> str:
    engine = _normalize_engine(engine)
    if engine in _account_cache:
        return _account_cache[engine]

    if engine == "openai":
        auth_path = _get_fixed_home() / ".codex" / "auth.json"
        email = _extract_codex_auth_email(auth_path)
        if email:
            _account_cache[engine] = email
            return email

    for root in _candidate_account_paths(engine):
        for path in _iter_small_account_files(root) or []:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            email = _email_from_text(text)
            if email:
                _account_cache[engine] = email
                return email

    _account_cache[engine] = ""
    return ""


def get_engine_status(settings: Settings) -> dict:
    engine = _normalize_engine(settings.engine)
    cli_available = check_cli_available(engine)
    account_email = find_cli_account_email(engine) if cli_available else ""

    if cli_available:
        probe = _probe_cli_ready(engine, _resolve_model(settings, "parser"))
        return {
            **probe,
            "installed": True,
            "account_email": account_email,
            "account_label": account_email or "연결 계정 확인 불가",
        }

    return {
        "status": "not_configured",
        "label": "CLI 미설치",
        "installed": False,
        "account_email": "",
        "account_label": "CLI를 설치하거나 로그인해 주세요.",
        "detail": f"{engine} CLI를 찾을 수 없습니다. 설치 및 PATH 설정을 확인해 주세요.",
    }
