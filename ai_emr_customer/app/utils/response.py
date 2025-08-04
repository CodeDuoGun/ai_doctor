"""
Custom General Response
"""
from flask import make_response
from typing import TypeVar

T = TypeVar("T")
import json

# TODO: wrap response
class RespException(Exception):
    def __init__(self, value: int, msg: str, msg_cn: str):
        self.value = value
        self.msg = msg
        self.msg_cn = msg_cn

    def __int__(self):
        return self.value

    def __str__(self):
        return self.msg


class RespStatus:
    BASE = 200000
    SUCCESS = RespException(BASE + 1, "Success", "成功")
    FAILED = RespException(BASE + 2, "Faild", "失败")
    UNKNOWN = RespException(BASE + 3, "Unknown error", "未知错误")
    PERMISSION_DENIED = RespException(BASE + 4, "Permission denied", "权限不足")
    NOT_LOGIN = RespException(BASE + 5, "Not login", "未登录")
    PARAM_ERROR = RespException(BASE + 6, "Params error", "参数错误")
    NOT_FOUND = RespException(BASE + 7, "Not found", "未找到")
    USER_BANNED = RespException(BASE + 8, "User is banned", "用户被禁用")
    METHOD_NOT_ALLOWED = RespException(BASE + 9, "Method not allowed", "请求方法不被允许")
    TYPEERROR = RespException(BASE + 10, "Type error", "格式错误")
    EXISTS = RespException(BASE + 11, "Exists", "已存在")
    NOT_EXISTS = RespException(BASE + 12, "Not exists", "不存在")
    DATA_PERMISSION_DENIED = RespException(
        BASE + 13, "Data permission denied", "数据权限不足"
    )


def general_response(code: int = 200, msg: str = "", data: T = None):
    """
    @param code: int, 状态码
    @param msg: str, 成功/错误信息
    @param data: T, 返回数据
    """
    resp = {"code": code, "msg": msg, "data": data}
    return make_response(json.dumps(resp, ensure_ascii=False), 200, {"Content-Type": "application/json; charset=utf-8"})