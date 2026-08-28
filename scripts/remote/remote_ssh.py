"""临时远端操作助手：python remote_ssh.py run "cmd" [timeout] | put local remote | get remote local"""
import re
import sys

import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 39359
USER = "root"
PASS = "grryKuINrbKQ"


def _rp(path):
    """撤销 Git Bash 对命令行参数的路径转换。"""
    m = re.match(r"^[A-Za-z]:/Program Files/Git(/.*)$", str(path))
    return m.group(1) if m else str(path)


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return c


def main():
    mode = sys.argv[1]
    client = connect()
    if mode == "run":
        cmd = re.sub(r"[A-Za-z]:/Program Files/Git(/[^\s\"']*)", r"\1", sys.argv[2])
        _, stdout, stderr = client.exec_command(
            cmd, timeout=int(sys.argv[3]) if len(sys.argv) > 3 else 120)
        out_bytes = stdout.read()
        sys.stdout.buffer.write(out_bytes)
        sys.stdout.buffer.flush()
        err = stderr.read().decode("utf-8", "replace")
        if err:
            sys.stderr.write(err)
        sys.exit(stdout.channel.recv_exit_status())
    elif mode == "put":
        sftp = client.open_sftp()
        sftp.put(sys.argv[2], _rp(sys.argv[3]))
        print("uploaded", sys.argv[2])
    elif mode == "get":
        sftp = client.open_sftp()
        sftp.get(_rp(sys.argv[2]), sys.argv[3])
        print("downloaded", sys.argv[3])
    client.close()


if __name__ == "__main__":
    main()
