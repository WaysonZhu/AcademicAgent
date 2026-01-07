import logging
import os
from logging.handlers import RotatingFileHandler
from config.settings import settings


def setup_logger(name: str):
    logger = logging.getLogger(name)

    # 如果 logger 已经有 handler，直接返回，避免重复添加导致重复日志
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # 禁止日志传播，阻止日志冒泡到 Root Logger (Chainlit 控制台)，只使用我们自定义的 Handler
    logger.propagate = False

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File Handler (写入文件)
    log_file = os.path.join(settings.LOG_DIR, "app.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler (输出到控制台)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger