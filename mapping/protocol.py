"""mapping 服务端/客户端共用的极简 TCP 协议。

消息格式：8 字节大端 header 长度 + JSON header + 二进制 payload。
header 中 ``payload_len`` 字段由发送方自动填写。

只依赖标准库，保证在 habitat (py3.9) 与 vggtslam (py3.11) 两个
conda 环境中都可以 import。
"""

import json
import struct

_HEADER_STRUCT = struct.Struct(">Q")


def send_msg(sock, header, payload=b""):
    """发送一条消息。header 为可 JSON 序列化的 dict，payload 为 bytes。"""
    header = dict(header)
    header["payload_len"] = len(payload)
    data = json.dumps(header).encode("utf-8")
    sock.sendall(_HEADER_STRUCT.pack(len(data)) + data + payload)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed by peer")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    """接收一条消息，返回 (header: dict, payload: bytes)。"""
    (hlen,) = _HEADER_STRUCT.unpack(_recv_exact(sock, _HEADER_STRUCT.size))
    header = json.loads(_recv_exact(sock, hlen).decode("utf-8"))
    payload = _recv_exact(sock, header.get("payload_len", 0))
    return header, payload
