# Codex 지시사항 - ARIA ver.2

## 프로젝트 개요

ARIA ver.2는 특허 청구항과 인용발명 PDF를 비교하여 신규성/진보성 판단 보고서를 생성하는 로컬 AI 보조 프로그램입니다.

## 현재 구현 상태

- UI: React 18, Vite, Tailwind CSS
- Backend: FastAPI
- Frontend port: `5274`
- Backend port: `8200`
- 입력 방식: 인용발명 PDF 최대 7개 업로드 후 사용자가 청구항을 직접 붙여넣기
- LLM: Claude CLI 기본, AGY CLI 선택 지원
- 비교 방식: 붙여넣은 청구항을 구성요소로 분해한 뒤 인용발명 전문과 순차 대비

## 프로젝트 구조

```text
new-patentsearching/
├── AGENTS.md
├── README.md
├── start.ps1
├── backend/
└── frontend/
```

## 코드 작성 원칙

- 요청 범위에 맞춰 작게 수정하고, 불필요한 리팩터링은 피합니다.
- 포트, 브랜딩, 실행 방법이 바뀌면 `README.md`와 관련 코드도 함께 갱신합니다.
- 보고서 양식이나 판정 기준 변경은 프롬프트/백엔드 로직/README 설명이 서로 어긋나지 않게 맞춥니다.
- 기존 사용자 작업으로 보이는 변경은 되돌리지 않습니다.
- 깨진 인코딩 문자열을 발견하면 가능한 범위에서 UTF-8 한국어 또는 명확한 영어로 복구합니다.
