from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────
    APP_ENV: str = "development"          # development | production
    DEBUG: bool = True

    # ── JWT — dùng CÙNG secret với Node.js để verify token ──────────────
    # Node.js dùng process.env.ACCESS_TOKEN_SECRET
    # FastAPI đọc ACCESS_TOKEN_SECRET từ .env cùng file
    ACCESS_TOKEN_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"

    # ── CORS ─────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",    # React dev
        "http://localhost:5000",    # Node.js gateway
    ]

    # ── MongoDB ──────────────────────────────────────────────────────────
    MONGODB_URI: str = "mongodb://mongo:27017"
    MONGODB_DB: str = "cv_chatbot"

    # MongoDB chứa jobs thực tế (lấy từ Node.js server)
    JOBS_MONGO_URI: str = "mongodb://mongo:27017"
    JOBS_MONGO_DB: str = "ITJOBS"

    # ── Redis ─────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://redis:6379/0"

    # ── ChromaDB ──────────────────────────────────────────────────────────
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8001

    # ── LLM APIs (free tier) ──────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # ── LangFuse observability ────────────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://langfuse:3000"   # self-hosted trong Docker

    # ── Chunking ─────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ── Token quota per user (cảnh báo khi gần hết) ───────────────────────
    DAILY_TOKEN_LIMIT: int = 100_000
    TOKEN_WARNING_THRESHOLD: float = 0.9   # cảnh báo khi dùng 90%

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()