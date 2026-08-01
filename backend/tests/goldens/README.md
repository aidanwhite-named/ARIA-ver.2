# 인용발명 선정 골든셋

`citation_chain`의 주/보조 인용발명 선정이 상수 조정이나 리팩터링으로
의도치 않게 바뀌는 것을 잡기 위한 회귀 픽스처입니다.

## 왜 별도 폴더인가

`cases/`와 `uploads/`는 `DELETE /api/jobs`로 통째로 삭제됩니다
(`backend/routers/analyze.py`의 `delete_job` / `delete_all_jobs`).
골든셋은 히스토리 삭제와 무관해야 하므로 여기에 두고 git으로 관리합니다.

## 파인튜닝이 아닙니다

모델 가중치를 학습하지 않습니다. 새 사건 판단에 이 데이터가 쓰이지도 않습니다.
pytest가 읽는 고정 입력일 뿐입니다.

## 2층 구조

### 1층 — 로직 골든셋 (여기 있는 것)

```
claims.json + comparisons_*.json  →  citation_chain (LLM 호출 없음)  →  주/보조
```

LLM을 한 번도 호출하지 않으므로 결과가 완전히 결정론적이고, **정확 일치**로
검증합니다. 다만 검증 범위는 선정 알고리즘뿐입니다.

### 2층 — 전체 파이프라인 골든셋 (미구축)

PDF 파싱·청킹·LLM 구성대비 판정까지 포함하는 층입니다. LLM 판정이 끼고
CLI 엔진들이 temperature를 노출하지 않으므로 **정확 일치 테스트가 될 수
없습니다.** 허용 범위(핵심 구성 판정 ±1등급, 주인용이 상위 2개 이내)로
설계해야 하며, 회귀 검출이 아니라 대형 붕괴 감지용입니다.

## 디렉터리 형식

```
goldens/
└── <case-name>/
    ├── claims.json          # List[ParsedClaim]
    ├── comparisons_0.json   # 문헌별 구성대비 판정 (문헌 수만큼)
    ├── comparisons_1.json
    └── expected.json
```

`expected.json`:

```json
{
  "documents": ["파일명1.pdf", "파일명2.pdf"],
  "families": {
    "1": {"primary_idx": 1, "secondary_idx": null}
  },
  "verified_by": "algorithm_snapshot",
  "note": "왜 이 답이 맞는지 사람이 적는 칸"
}
```

### `verified_by` 값의 의미 — 중요

| 값 | 뜻 | 잡을 수 있는 것 |
| :--- | :--- | :--- |
| `algorithm_snapshot` | 현재 알고리즘 출력을 그대로 떠온 것 | 의도치 않은 **변경**만 |
| `human` | 사람이 확인한 정답 | 알고리즘의 **오답** |

`algorithm_snapshot`은 알고리즘이 맞다는 근거가 아닙니다. 상수를 고쳤을 때
결과가 흔들리는지만 알려줍니다. 사건을 실제로 검토했다면 `verified_by`를
`human`으로 바꾸고 `note`에 근거를 남겨 주십시오. 그때부터 그 케이스가
비로소 정답지 역할을 합니다.

## 케이스 추가

분석이 끝난 사건을 골든셋에 넣습니다.

```bash
python backend/tests/goldens/seed_golden.py <CASE-ID> [golden-name]
```

`uploads/<CASE-ID>/`와 `cases/<CASE-ID>/parsed/`에서 필요한 파일만 복사하고,
현재 알고리즘 출력을 `expected.json`에 `algorithm_snapshot`으로 기록합니다.
PDF 원문과 청크는 복사하지 않습니다.
