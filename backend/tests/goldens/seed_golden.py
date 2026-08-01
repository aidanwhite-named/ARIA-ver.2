"""분석이 끝난 사건을 인용발명 선정 골든셋으로 복사한다.

    python backend/tests/goldens/seed_golden.py <CASE-ID> [golden-name]

uploads/<CASE-ID>/comparisons_*.json 과 cases/<CASE-ID>/parsed/claims.json 만
가져오고, PDF 원문·청크·보고서는 복사하지 않는다. 기대값은 현재 알고리즘
출력을 그대로 기록하므로 `verified_by`가 "algorithm_snapshot"이다. 사건을 실제로
검토했다면 손으로 "human"으로 바꾸고 note에 근거를 남긴다.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from backend.models.schemas import ExtractedDocument, ParsedClaim  # noqa: E402
from backend.paths import CASES_DIR, UPLOADS_DIR  # noqa: E402
from backend.services.citation_chain import (  # noqa: E402
    build_citation_chain_from_comparisons,
)

GOLDENS_DIR = Path(__file__).resolve().parent


def seed(case_id: str, golden_name: str | None = None) -> Path:
    job_dir = UPLOADS_DIR / case_id
    parsed_dir = CASES_DIR / case_id / "parsed"
    if not job_dir.exists():
        raise SystemExit(f"업로드 폴더가 없습니다: {job_dir}")
    if not (parsed_dir / "claims.json").exists():
        raise SystemExit(f"claims.json이 없습니다: {parsed_dir}")

    comparisons = sorted(job_dir.glob("comparisons_*.json"),
                         key=lambda path: int(path.stem.split("_")[-1]))
    if not comparisons:
        raise SystemExit(f"comparisons_*.json이 없습니다: {job_dir}")

    target = GOLDENS_DIR / (golden_name or case_id)
    target.mkdir(parents=True, exist_ok=True)

    shutil.copy2(parsed_dir / "claims.json", target / "claims.json")
    for path in comparisons:
        shutil.copy2(path, target / path.name)

    claims = [ParsedClaim(**item) for item in
              json.loads((target / "claims.json").read_text(encoding="utf-8"))]
    prior_docs_raw = json.loads(
        (parsed_dir / "prior_docs.json").read_text(encoding="utf-8")
    )
    prior_docs = [
        ExtractedDocument(
            filename=Path(str(item.get("pdf_path", ""))).name or f"doc{index}.pdf",
            document_type=item.get("document_type", ""),
        )
        for index, item in enumerate(prior_docs_raw)
    ]

    chain = build_citation_chain_from_comparisons(str(target), claims, prior_docs)
    # 체인 빌드는 job_dir에 citation_chain.json을 남기고, 다음 실행에서 그 파일의
    # selection_locks를 승계한다. 픽스처에 남으면 골든 테스트가 선정을 다시
    # 계산하지 않고 잠긴 값을 그대로 통과시키므로 반드시 지운다.
    (target / "citation_chain.json").unlink(missing_ok=True)
    expected = {
        "documents": [doc.filename for doc in prior_docs],
        "families": {
            key: {
                "primary_idx": value.get("primary_idx"),
                "secondary_idx": value.get("secondary_idx"),
            }
            for key, value in (chain.get("families") or {}).items()
        },
        "verified_by": "algorithm_snapshot",
        "note": "",
    }
    (target / "expected.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"seeded {target.relative_to(ROOT)}  families={expected['families']}")
    return target


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    seed(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
