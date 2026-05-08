from celery import Celery
from core.config import settings

# Celery dùng Redis làm broker (nơi nhận job)
# và Redis làm backend (lưu kết quả job)

celery_app = Celery(
    "cv_workers",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["workers.cv_worker"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,          # chỉ xóa khỏi queue sau khi hoàn thành
    worker_prefetch_multiplier=1, # mỗi worker chỉ nhận 1 job một lúc (embedding nặng)
    task_time_limit=600,
    task_soft_time_limit=540,
)