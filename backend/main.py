import os
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
