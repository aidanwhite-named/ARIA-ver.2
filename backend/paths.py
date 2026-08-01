from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
REPORTS_DIR = PROJECT_ROOT / "reports"
CASES_DIR = PROJECT_ROOT / "cases"
# 인용발명 판정 캐시는 사건 폴더 밖에 둔다. 히스토리를 삭제해도 같은 입력에
# 대한 판정이 유지되어야 재현성이 보장되기 때문이다.
ADJUDICATION_CACHE_DIR = PROJECT_ROOT / "data" / "adjudication_cache"
