# custom_logger.py

import logging
import sys

class StreamToStdout(logging.StreamHandler):
    def __init__(self):
        super().__init__(sys.stdout)

    def emit(self, record):
        if record.levelno < logging.ERROR:
            super().emit(record)

class StreamToStderr(logging.StreamHandler):
    def __init__(self):
        super().__init__(sys.stderr)

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            super().emit(record)

def setup_custom_logger():
    # 获取 root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建输出到标准输出的处理器
    stdout_handler = StreamToStdout()
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s'))

    # 创建输出到标准错误的处理器
    stderr_handler = StreamToStderr()
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s'))

    # 如果 logger 没有处理器，则添加处理器
    if not logger.hasHandlers():
        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)

    return logger

