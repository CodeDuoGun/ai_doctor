"""启动一个ws服务，异步接收前端消息
# 文本类型
{
    "msg_type": "text",
    "msg": {""}
}

# 音频类型
{
    "msg_type": "text",
    "msg": b''
}

# 上传舌照面照图片
{
    "msg_type": "tongue_face_img",
    "msg": ["url"] or []
}

# 上传检查资料
{
    "msg_type": "check_img",
    "msg": ["url"] or []
}

"""
#!/usr/bin/env python
