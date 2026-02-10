import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "api.log")

logger = logging.getLogger("health_api")
logger.setLevel(logging.INFO)

handler = TimedRotatingFileHandler(
    LOG_FILE,
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

handler.setFormatter(formatter)
logger.addHandler(handler)
