from functools import wraps
import redis
from app.config import config


def singleton(cls):
    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls.instance is None:
            cls.instance = cls(*args, **kwargs)
        return cls.instance

    cls.instance = None
    return wrapper


@singleton
class RedisSingleton:
    def __init__(self):
        self.client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB, password=config.REDIS_AUTH, decode_responses=True)

    def set(self, key, value):
        self.client.set(key, value)

    def set_expire(self, key, value, expire_time=86400):
        # TODO: 暂时保留7d, 后面写入mongodb来维护
        self.set(key, value)
        self.client.expire(key, expire_time)

    def get(self, key):
        return self.client.get(key)

    def exist(self, key):
        return self.client.exists(key)

    def delete(self, key):
        self.client.delete(key)

    def hexists(self, table, key):
        return self.client.hexists(table, key)

    def hget(self, table, key):
        return self.client.hget(table, key)

    def hgetall(self, table):
        return self.client.hgetall(table)

    def hset(self, table, key, value):
        self.client.hset(table, key, value)

    def hkeys(self, table):
        return self.client.hkeys(table)

    def sadd(self, key, *values):
        self.client.sadd(key, *values)

    def smembers(self, key):
        return self.client.smembers(key)

    def srem(self, key, *values):
        return self.client.srem(key, *values)

    def scan_iter(self, key):
        return self.client.scan_iter(key)

    def spop(self, key):
        return self.client.spop(key)

    def sismember(self, key, value):
        return self.client.sismember(key, value)

    def lpush(self, key, value):
        return self.client.lpush(key, value)

    def rpop(self, key):
        return self.client.rpop(key)

    def llen(self, key):
        return self.client.llen(key)

    def incr(self, key):
        return self.client.incr(key)

    def lrange(self, key, start, end):
        return self.client.lrange(key, start, end)

redis_tool = RedisSingleton()