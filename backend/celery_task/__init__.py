from celery import Celery
from core.config import default_config


celery_app = Celery(
    "worker",
    backend=default_config.CELERY_BACKEND_URL,
    broker=default_config.CELERY_BROKER_URL,
)

celery_app.conf.task_routes = {"worker.celery_worker.test_celery": "test-queue"}
celery_app.conf.update(task_track_started=True)
