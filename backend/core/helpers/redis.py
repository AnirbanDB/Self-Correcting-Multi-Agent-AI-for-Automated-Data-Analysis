from redis import asyncio as redis

from core.config import default_config

redis_client = redis.from_url(url=f"redis://{default_config.REDIS_HOST}", decode_responses=True)
