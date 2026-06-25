import os
import sys
import io

# Windows 환경 등에서 시스템 로케일로 인해 stdout, stderr, 파일 입출력 인코딩이 깨지는 문제를 방지하기 위해
# 환경 변수와 표준 출력을 UTF-8 인코딩으로 강제 설정합니다.
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass
if sys.stderr and sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import analyze, settings as settings_router

app = FastAPI(title="ARIA ver.2 특허 신규성/진보성 판단 보고서 생성기", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5274", "http://127.0.0.1:5274", "http://0.0.0.0:5274"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)

app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
app.include_router(settings_router.router, prefix="/settings", tags=["settings"])


@app.get("/")
async def root():
    return {"status": "ok", "message": "ARIA ver.2 Patent Report API"}
