"""
Competition control helper — talk to the Teacher Server.

NOTE: The Teacher Server is offline right now. This script is pre-configured so
that the moment it comes online you can run these commands without edits — all
values are read from .env via config.py.

Usage:
    python scripts/compete.py register          # auto-detect LAN IP and register
    python scripts/compete.py register --ip 192.168.1.15 --port 5000

    # FIRST run (Teacher sends the document via /upload, then asks):
    python scripts/compete.py evaluate

    # SUBSEQUENT runs (document already embedded + cached -> skip /upload):
    python scripts/compete.py evaluate --document-received

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

        # Parse response based on endpoint (Modified / Modified 2 schema).
        if "/register" in path and "message" in data:
            print(f"✓ Registration: {data['message']}")
        elif "/evaluate" in path and "final_score" in data:
            print(f"✓ Evaluation complete! Final score: {data['final_score']}")
        elif "/reset" in path and "score" in data:
            print(f"✓ Reset complete. Score: {data['score']}")

        return data
    except Exception as e:
        print(f"Error parsing response: {e}")
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


def cmd_evaluate(document_received: bool = False) -> None:
    """
    Start the exam (Modified 2: body carries document_received).

    document_received=False (default, FIRST run):
        Teacher will call /upload (send the document) then the questions.
        Use this the first time so your server receives + embeds + saves the
        document. Even if the Teacher reports an /upload timeout, your server
        keeps embedding in the background and persists the vector cache.

    document_received=True (SUBSEQUENT runs):
        Tells the Teacher you ALREADY have the document, so it SKIPS /upload
        and goes straight to the questions. Use this after the document has
        been embedded + cached (and your server reloaded it on startup), to
        avoid the slow upload/embed step and the 120s timeout.
    """
    if document_received:
        print("[evaluate] document_received=True -> Teacher SKIPS /upload, asks questions directly.")
    else:
        print("[evaluate] document_received=False -> Teacher will call /upload first, then ask.")
    _post("/competition/evaluate", {"document_received": document_received})


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

    pe = sub.add_parser("evaluate", help="start the exam")
    pe.add_argument(
        "--document-received",
        action="store_true",
        help="tell Teacher you already have the document (skips /upload).",
    )

    sub.add_parser("result", help="check score/status")
    sub.add_parser("reset", help="reset your session")

    pn = sub.add_parser("run", help="register then evaluate")
    pn.add_argument("--ip", default=None)
    pn.add_argument("--port", type=int, default=config.PORT)
    pn.add_argument(
        "--document-received",
        action="store_true",
        help="tell Teacher you already have the document (skips /upload).",
    )

    args = p.parse_args()

    if args.cmd == "register":
        cmd_register(args.ip, args.port)
    elif args.cmd == "evaluate":
        cmd_evaluate(document_received=args.document_received)
    elif args.cmd == "result":
        cmd_result()
    elif args.cmd == "reset":
        cmd_reset()
    elif args.cmd == "run":
        cmd_register(args.ip, args.port)
        cmd_evaluate(document_received=args.document_received)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
