REDIS_HOST = "REDIS_HOST", "dummy-redis-host"
REDIS_PORT = "REDIS_PORT", "1234"
REDIS_PASSWORD = "REDIS_PASSWORD", "dummy-password"

def get_redis_url():
    if REDIS_PASSWORD:
        return f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
    else:
        return f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

REDIS_URL = get_redis_url()