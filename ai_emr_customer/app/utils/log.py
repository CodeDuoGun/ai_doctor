import logging
import os
import sys
# import redis
from loguru import logger as logurulogger
# import redis.exceptions
from app.config import config
import json
# from redis.retry import Retry
# from redis.backoff import ExponentialBackoff

LOG_FORMAT = (
    "<level>{level: <8}</level> "
    "{process.name} | "  # 进程名
    "{thread.name}  | "
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - "
    "<blue>{process}</blue> "
    "<cyan>{module}</cyan>.<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
LOG_NAME = ["uvicorn", "uvicorn.access", "uvicorn.error", "flask"]

# 配置 Redis 连接池
# redis_pool = redis.ConnectionPool(
#     host=config.LOG_REDIS_HOST,  # Redis 服务器地址
#     port=config.LOG_REDIS_PORT,  # Redis 服务器端口
#     db=config.LOG_REDIS_DB,  # 数据库编号
#     password=config.LOG_REDIS_AUTH,  # 密码
#     max_connections=config.max_connections,  # 最大连接数
#     socket_connect_timeout=config.socket_connect_timeout,  # 连接超时时间
#     socket_timeout=config.socket_timeout,  # 等待超时时间
# )


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logurulogger.level(record.levelname).name
        except AttributeError:
            level = logging._levelToName[record.levelno]

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logurulogger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

class Logging:
    """自定义日志"""

    def __init__(self):
        self.log_path = "logs"
        # self._connect_redis()
        os.makedirs(self.log_path, exist_ok=True)
        self._initlogger()
        self._reset_log_handler()
    
    def _connect_redis(self):
        retry = Retry(ExponentialBackoff(), 3)  # 重试3次，指数退避
        self.redis_client = redis.Redis(connection_pool=redis_pool,retry=retry)  # 使用连接池

    def _initlogger(self):
        """初始化loguru配置"""
        logurulogger.remove()
        logurulogger.add(
            os.path.join(self.log_path, "error.log.{time:YYYY-MM-DD}"),
            format=LOG_FORMAT,
            level=logging.ERROR,
            rotation="00:00",
            retention="1 week",
            backtrace=True,
            diagnose=True,
            enqueue=True
        )
        logurulogger.add(
            os.path.join(self.log_path, "info.log.{time:YYYY-MM-DD}"),
            format=LOG_FORMAT,
            level=logging.INFO,
            rotation="00:00",
            retention="1 week",
            enqueue=True
        )
        logurulogger.add(
            sys.stdout,
            format=LOG_FORMAT,
            level=logging.DEBUG,
            colorize=True,
        )

        # logurulogger.add(self._log_to_redis, level="INFO", format=LOG_FORMAT)
        self.logger = logurulogger
        

    def _log_to_redis(self, message):
        """将日志写入 Redis 列表"""
        try:
            self.redis_client.rpush(f"nlp.logger.{config.env_version}.log", json.dumps({"message": message}))
        except redis.exceptions.ConnectionError as e:
            logger.error(f"write {message} Redis connection error: {e}")
        except redis.exceptions.TimeoutError as e:
            logger.error(f"write {message} Redis operation timed out: {e}")
        except Exception as e:
            logger.error(f"write {message} Unexpected error: {e}")

    def _reset_log_handler(self):
        for log in LOG_NAME:
            logger = logging.getLogger(log)
            logger.handlers = [InterceptHandler()]

    def getlogger(self):
        return self.logger 

logger = Logging().getlogger()

