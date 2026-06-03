"""
Competition control helper — talk to the Teacher Server.

NOTE: The Teacher Server is offline right now. This script is pre-configured so
that the moment it comes online you can run these commands without edits — all
values are read from .env via config.py.

Usage:
    python scripts/compete.py register          # auto-detect LAN IP and register
    python scripts/compete.py register --ip 192.168.1.15 --port 5000
    python scripts/compete.py evaluate          # start the exam (blocks until done)
    python scripts/compete.py result            # check current score/status
    python scripts/compete.py reset             # reset your session
    python scripts/compete.py run               # register + evaluate in one go

All requests send the required  X-Student-ID  header.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys

import requests

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
import config

HEADERS = {"X-Student-ID": config.STUDENT_ID, "Content-Type": "application/json"}


def detect_lan_ip() -> str:
    """Best-effort detection of this machine's LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        host = config.TEACHER_BASE_URL.split("//")[1].split("/")[0].split(":")[0]
        s.connect((host, 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = socket.gethostbyname(socket.gethostname())
    finally:
        s.close()
    return ip


def _post(path: str, body: dict | None = None) -> dict:
    url = f"{config.TEACHER_BASE_URL}{path}"
    r = requests.post(url, headers=HEADERS, json=body or {}, timeout=900)
    print(f"POST {url} -> {r.status_code}")
    try:
        data = r.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return data
    except Exception:
        print(r.text)
        return {}


def _get(path: str) -> dict:
    url = f"{config.TEACHER_BASE_URL}{path}"
    r = requests.get(url, headers=HEADERS, timeout=60)
    print(f"GET {url} -> {r.status_code}")
    try:
        data = r.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return data
    except Exception:
        print(r.text)
        return {}


def cmd_register(ip: str | None, port: int) -> None:
    ip = ip or detect_lan_ip()
    server_url = f"http://{ip}:{port}"
    print(f"[register] student_id={config.STUDENT_ID} server_url={server_url}")
    _post("/competition/register", {"server_url": server_url})


def cmd_evaluate() -> None:
    print("[evaluate] Starting exam. Teacher will call /upload then 10x /ask.")
    _post("/competition/evaluate")


def cmd_result() -> None:
    _get("/competition/result")


def cmd_reset() -> None:
    _post("/competition/reset")


def main() -> None:
    p = argparse.ArgumentParser(description="Offline RAG competition controller")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("register", help="register this server with the teacher")
    pr.add_argument("--ip", default=None, help="LAN IP (auto-detected if omitted)")
    pr.add_argument("--port", type=int, default=config.PORT)

    sub.add_parser("evaluate", help="start the exam")
    sub.add_parser("result", help="check score/status")
    sub.add_parser("reset", help="reset your session")

    pn = sub.add_parser("run", help="register then evaluate")
    pn.add_argument("--ip", default=None)
    pn.add_argument("--port", type=int, default=config.PORT)

    args = p.parse_args()

    if args.cmd == "register":
        cmd_register(args.ip, args.port)
    elif args.cmd == "evaluate":
        cmd_evaluate()
    elif args.cmd == "result":
        cmd_result()
    elif args.cmd == "reset":
        cmd_reset()
    elif args.cmd == "run":
        cmd_register(args.ip, args.port)
        cmd_evaluate()
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
