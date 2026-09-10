"""Event-driven monitor for the benchmark remote.

No polling from this side: one long-lived SSH connection runs a watcher on the
remote that checks the episode checkpoint dir every few seconds and only pushes
a line when something happens (episode completed / worker count changed /
files appeared or gone / heartbeat). Each pushed line is forwarded to stdout,
which under the Monitor tool becomes one notification.

State (byte offsets per run file) is persisted so reconnects never re-report
rows that were already reported.

Usage:
    python monitor_remote.py [--log <statefile>]
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time

import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 48455
USER = "root"
PASS = "cF/hGEtNEHn0"

RESULTS_DIR = "/root/autodl-tmp/habitat_benchmark_eval_20260827/eval_results"
PY = "/root/miniconda3/bin/python"
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".monitor_state.json")

CHECK_INTERVAL = 3     # remote internal directory scan (s)
PROC_INTERVAL = 60     # remote internal worker-process check (s)
HB_INTERVAL = 1800     # remote heartbeat (s)

# Remote watcher. Emits one JSON line per event, flushed immediately:
#   {"t":"sync"}                      initial: current worker counts
#   {"t":"new", "f":..., "n":rows}    run file appeared with rows already in it
#   {"t":"ep", "f":..., "off":bytes}  episode completed (+ metrics summary)
#   {"t":"reset","f":...}             file shrank / replaced
#   {"t":"gone", "f":...}             run file disappeared
#   {"t":"proc","n":N,"d":[...],...}  worker process count changed
#   {"t":"hb","n":N,"ep":M,...}       heartbeat
# Event rows carry "off" = byte offset *after* that row, which the local side
# persists so a reconnect never re-extracts it.
WATCHER = r"""
import json, os, re, subprocess, sys, time
RESULTS = @RESULTS@
CHECK = @CHECK@
PROC_DT = @PROC_DT@
HB_DT = @HB_DT@
KNOWN = @KNOWN@            # {file: bytes_consumed}

def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)

def rows_after(path, off):
    # yield (json_obj, byte_offset_after_row) for rows past `off`
    sz = os.path.getsize(path)
    pos = off
    if sz <= off:
        return
    with open(path, encoding='utf-8', errors='replace') as fh:
        fh.seek(off)
        while True:
            line = fh.readline()
            if not line:
                break
            pos = fh.tell()
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                emit({"t": "badjson", "f": os.path.basename(path)})
                continue
            yield d, pos

def ep_summary(d, f):
    ep = d.get('episode') or {}
    m = ep.get('metrics') or ep
    fin = bool(m.get('finished_by_agent', ep.get('finished_by_agent')))
    to = bool(m.get('timed_out', ep.get('timed_out')))
    failed = m.get('failed', ep.get('failed'))
    s = {
        "t": "ep", "f": f, "ep": ep.get('episode_id'),
        "scene": (ep.get('scene_id') or '').split('/')[-1],
        "sr": m.get('sr'), "f1": m.get('f1'), "spl": m.get('spl_multi'),
        "p": m.get('precision'), "r": m.get('recall'),
        "finished": fin, "timed_out": to, "failed": failed,
        "reason": ep.get('failure_reason'),
        "steps": m.get('steps', m.get('finish_step', ep.get('finish_step'))),
    }
    return s

cur_proc = None
gone = set()
last_proc_t = last_hb = time.time()

while True:
    now = time.time()
    try:
        names = sorted(os.listdir(RESULTS))
    except Exception as e:
        emit({"t": "warn", "msg": "listdir: %s" % (e,)})
        time.sleep(CHECK)
        continue

    files = {}
    for fe in names:
        if not re.fullmatch(r'eval_\d{8}_\d{6}\.episodes\.jsonl', fe):
            continue
        try:
            p = os.path.join(RESULTS, fe)
            files[fe] = os.path.getsize(p)
        except OSError:
            pass

    # files that vanished
    for fe in KNOWN:
        if fe not in files and fe not in gone:
            emit({"t": "gone", "f": fe})
            gone.add(fe)

    for fe, sz in files.items():
        p = os.path.join(RESULTS, fe)
        if fe not in KNOWN:
            KNOWN[fe] = 0
            for d, pos in rows_after(p, 0):
                s = ep_summary(d, fe); s["off"] = pos
                emit(s)
                KNOWN[fe] = pos
            continue
        if sz < KNOWN[fe]:
            emit({"t": "reset", "f": fe})
            KNOWN[fe] = sz
            continue
        if sz > KNOWN[fe]:
            for d, pos in rows_after(p, KNOWN[fe]):
                s = ep_summary(d, fe); s["off"] = pos
                emit(s)
                KNOWN[fe] = pos

    if now - last_proc_t >= PROC_DT:
        last_proc_t = now
        out = subprocess.run(
            ["ps", "-eo", "etime,args"],
            capture_output=True, text=True).stdout
        procs = []
        for line in out.splitlines():
            if "run_eval.py" in line and "grep" not in line:
                m = re.search(r"--episode-id (\S+)", line)
                procs.append({"k": "eval", "ep": m.group(1) if m else None,
                              "run": re.search(r"eval_(\d{8}_\d{6})", line).group(1) if re.search(r"eval_(\d{8}_\d{6})", line) else None})
            elif "mapping.server" in line and "grep" not in line:
                m = re.search(r"--port (\d+)", line)
                procs.append({"k": "map", "port": m.group(1) if m else None})
        n = len(procs)
        if n != cur_proc:
            emit({"t": "proc", "n": n, "d": procs})
            if n == 0 and cur_proc is not None and cur_proc > 0:
                emit({"t": "batch", "msg": "all workers gone - test batch may be finished"})
            cur_proc = n

    if now - last_hb >= HB_DT:
        last_hb = now
        nrows = sum(1 for fe in files if re.fullmatch(r'eval_\d{8}_\d{6}\.episodes\.jsonl', fe))
        emit({"t": "hb", "files": nrows,
              "ep": sum(1 for fe in KNOWN if fe in files),
              "proc": cur_proc,
              "at": time.strftime("%H:%M:%S")})
    time.sleep(CHECK)
"""


def fmt_disp(v, nd=2):
    if v is None:
        return "?"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def load_state(path):
    # values are byte offsets (int). Older versions stored {lines,mtime,bytes}
    # dicts — normalize to the byte offset kept there.
    try:
        files = json.load(open(path, encoding="utf-8")).get("files", {})
    except Exception:
        return {}
    out = {}
    for f, v in files.items():
        if isinstance(v, dict):
            if isinstance(v.get("bytes"), int):
                out[f] = v["bytes"]
            elif isinstance(v.get("mtime"), int):
                out[f] = v["mtime"]
            else:
                out[f] = 0
        elif isinstance(v, int):
            out[f] = v
    changed = out != files
    if changed:
        save_state(path, out)
    return out


def save_state(path, state):
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"files": state, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=1)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=STATE_PATH)
    args = ap.parse_args()

    state = load_state(args.log)
    last_gone = {}
    print(f"[monitor] event-driven; watching {RESULTS_DIR} (state: {args.log})")

    while True:
        client = None
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=45, banner_timeout=45)
            client.get_transport().set_keepalive(30)

            watcher = WATCHER.replace("@RESULTS@", json.dumps(RESULTS_DIR)).replace(
                "@CHECK@", str(CHECK_INTERVAL)
            ).replace("@PROC_DT@", str(PROC_INTERVAL)).replace(
                "@HB_DT@", str(HB_INTERVAL)
            ).replace("@KNOWN@", json.dumps(state))
            _si, so, _se = client.exec_command(f"{PY} - <<'PYEOF'\n{watcher}\nPYEOF")
            # stream lines; each is one event. Channel EOF ends the stream.
            for line in iter(so.readline, ""):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    print(f"[RAW] {line[:200]}")
                    continue
                t = ev.get("t")
                if t == "ep":
                    fin = "FINISHED" if ev.get("finished") else ("TIMED_OUT" if ev.get("timed_out") else "ENDED")
                    fail = "" if ev.get("finished") else " <-- FAIL"
                    reason = (f" ({ev.get('reason')})" if ev.get("reason") else "")
                    print(
                        f"[EP] {ev.get('f')} | {ev.get('ep')} (scene {ev.get('scene')}) | {fin} | "
                        f"SR={fmt_disp(ev.get('sr'))} F1={fmt_disp(ev.get('f1'))} SPLm={fmt_disp(ev.get('spl'))} "
                        f"P={fmt_disp(ev.get('p'))} R={fmt_disp(ev.get('r'))} steps={ev.get('steps')}"
                        f"{reason}{fail}"
                    )
                    if "off" in ev:
                        state[ev["f"]] = ev["off"]
                    save_state(args.log, state)
                elif t == "reset":
                    print(f"[RUN_RESET] {ev.get('f')} shrank/replaced on remote")
                elif t == "gone":
                    if last_gone.get(ev.get("f")) != True:
                        print(f"[RUN_GONE] {ev.get('f')}")
                        last_gone[ev.get("f")] = True
                    if ev.get("f") in state:
                        del state[ev.get("f")]
                        save_state(args.log, state)
                elif t == "proc":
                    d = ev.get("d", [])
                    evals = [x["ep"] for x in d if x.get("k") == "eval"]
                    maps = [x["port"] for x in d if x.get("k") == "map"]
                    print(
                        f"[PROC] workers={ev.get('n')} "
                        f"(eval: {', '.join(evals) if evals else '-'} | map ports: {', '.join(maps) if maps else '-'})"
                    )
                elif t == "batch":
                    print(f"[BATCH] {ev.get('msg')}")
                elif t == "hb":
                    print(f"[HB] files={ev.get('files')} episodes_done={ev.get('ep')} workers={ev.get('proc')} @ {ev.get('at')}")
                elif t == "badjson":
                    print(f"[ERR] unparseable row in {ev.get('f')}")
                elif t == "warn":
                    print(f"[WARN] {ev.get('msg')}")
            # stream ended: surface the remote exit status instead of silently reconnecting
            rc = so.channel.recv_exit_status() if so.channel.exit_status_ready() else None
            if rc not in (None, 0):
                print(f"[WATCHER-DIED] remote watcher exited rc={rc}: {_se.read().decode('utf-8', 'replace')[-300:]}")
            elif rc == 0:
                print(f"[WATCHER-DIED] remote watcher exited rc=0 (unexpected)")
            else:
                print("[LOST] remote stream ended without exit status")
        except Exception as exc:  # noqa: BLE001
            print(f"[LOST] {str(exc)[:160]}")
        finally:
            try:
                if client:
                    client.close()
            except Exception:  # noqa: BLE001
                pass
        # reconnect with backoff; state is persisted so nothing is re-reported
        time.sleep(10)


if __name__ == "__main__":
    main()