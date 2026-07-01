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
        "https://it-job-finder-client-five.vercel.app",  # Production frontend
        "https://it-job-finder-server.onrender.com",     # Production Node.js
    ]

    # ── MongoDB ──────────────────────────────────────────────────────────
    MONGODB_URI: str = "mongodb://22110434_db_user:Tinvn1201@ac-obmd0vz-shard-00-00.e0lm6xu.mongodb.net:27017,ac-obmd0vz-shard-00-01.e0lm6xu.mongodb.net:27017,ac-obmd0vz-shard-00-02.e0lm6xu.mongodb.net:27017/ITJOBS?ssl=true&replicaSet=atlas-13rwz6-shard-0&authSource=admin&appName=ITJOBS-Cluster"
    MONGODB_DB: str = "cv_chatbot"

    # MongoDB chứa jobs thực tế (lấy từ Node.js server)
    JOBS_MONGO_URI: str = "mongodb://22110434_db_user:Tinvn1201@ac-obmd0vz-shard-00-00.e0lm6xu.mongodb.net:27017,ac-obmd0vz-shard-00-01.e0lm6xu.mongodb.net:27017,ac-obmd0vz-shard-00-02.e0lm6xu.mongodb.net:27017/ITJOBS?ssl=true&replicaSet=atlas-13rwz6-shard-0&authSource=admin&appName=ITJOBS-Cluster"
    JOBS_MONGO_DB: str = "ITJOBS"

    # ── Redis (Upstash ở production, local redis ở dev) ──────────────────
    REDIS_URL: str = "redis://redis:6379/0"

    # ── LLM APIs (free tier) ──────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # ── LangFuse observability ────────────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://langfuse:3000"   # self-hosted trong Docker

    # ── Storage ─────────────────────────────────────────────────────────────────
    STORAGE_BACKEND: str = "local"    # local | s3
    STORAGE_LOCAL_UPLOAD_DIR: str = ".uploads"
    STORAGE_S3_BUCKET: str = ""
    STORAGE_S3_REGION: str = ""
    STORAGE_S3_ENDPOINT_URL: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""

    # ── Chunking ─────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ── Token quota per user (cảnh báo khi gần hết) ───────────────────────
    DAILY_TOKEN_LIMIT: int = 100_000
    TOKEN_WARNING_THRESHOLD: float = 0.9   # cảnh báo khi dùng 90%

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()