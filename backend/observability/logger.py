import sys
import json
import time
from typing import Any, Dict
from loguru import logger
from config import settings


def configure_logging():
    logger.remove()
    if settings.debug:
        logger.add(sys.stderr, level="DEBUG", format="{time:HH:mm:ss.SSS} | {level:<7} | {name}:{line} | {message}")
    else:
        logger.add(sys.stderr, level=settings.log_level, format="{time:HH:mm:ss} | {level:<7} | {message}")


def log_request(request_id: str, query: str, metadata: Dict[str, Any] = {}):
    logger.info(json.dumps({
        "request_id": request_id,
        "query": query[:100],
        "ts": time.time(),
        **metadata,
    }))


def log_trace(request_id: str, trace: Dict[str, Any]):
    if settings.debug:
        logger.debug(json.dumps({"request_id": request_id, "trace": trace}, indent=2))
