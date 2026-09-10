"""Execute a command on the benchmark remote host via paramiko (password auth).

Usage:
    python remote_exec.py "command here"

Prints remote stdout to local stdout. Exit code mirrors a trimmed remote exit code.
"""
import sys

import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 48455
USER = "root"
PASS = "cF/hGEtNEHn0"


def main():
    cmd = " ".join(sys.argv[1:])
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    except Exception as exc:  # noqa: BLE001
        print(f"[SSH-CONNECT-FAIL] {exc}", file=sys.stderr)
        sys.exit(2)

    stdin, stdout, stderr = client.exec_command(cmd, timeout=180)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    client.close()
    sys.exit(code if code is not None else 1)


if __name__ == "__main__":
    main()