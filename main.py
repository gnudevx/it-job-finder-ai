from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from core.config import settings
from core.logger import setup_logging
from routers import chat, cv, health

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    logger.info("🚀 AI Service starting up...")
    
    # ← Initialize MongoDB collections
    try:
        from services.CV.storage.metadata_service import _get_collection
        _get_collection()  # ← Force initialization
        logger.info("✅ MongoDB collections initialized")
    except Exception as e:
        logger.exception(f"❌ MongoDB initialization failed: {e}")
    
    # TODO: khởi tạo ChromaDB client, embedding model ở đây
    yield
    logger.info("🛑 AI Service shutting down...")


app = FastAPI(
    title="CV Chatbot AI Service",
    version="1.0.0",
    docs_url="/docs",           # Swagger UI — tắt ở production
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────
# Chỉ cho phép frontend và Node.js gateway gọi vào
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health Check & Root Route ────────────────────────────────────────────────
@app.get("/")
async def root():
    """Root endpoint để Render port scan thành công với 200 OK."""
    return {"message": "CV Chatbot AI Service is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint trực tiếp tránh redirect 307 cho Render."""
    return {"status": "ok", "service": "cv-chatbot-ai"}


# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/health",  tags=["health"])
app.include_router(cv.router,     prefix="/api/cv",  tags=["cv"])
app.include_router(chat.router,   prefix="/api/chat",tags=["chat"])