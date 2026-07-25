"""Launch every LIGERBOT module as a subprocess and stream their logs together.

    python run_all.py                 # live feed (needs Kotak Neo creds)
    python run_all.py --simulate      # synthetic feed, no broker required

Each module still runs as an isolated process — this is just a convenience wrapper
so you don't need five terminals. Ctrl-C shuts them all down cleanly.

For real production deployment prefer a proper supervisor (systemd, Docker Compose
services, or a process manager) over this script.
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from typing import List

import config
from src import event_bus

# Start order: consumers first so they're ready when ingestion produces ticks.
# Consumer groups make the order far less critical than it once was — a module that
# starts late now picks up the backlog rather than skipping it (defect B6) — but
# starting consumers first still avoids an unnecessary burst of catch-up work.
BASE_MODULES = [
    ("storage",   [sys.executable, "-m", "src.storage_logger"]),
    ("positions", [sys.executable, "-m", "src.position_manager"]),
    ("risk",      [sys.executable, "-m", "src.risk_manager"]),
    ("strategy",  [sys.executable, "-m", "src.strategy_engine"]),
    ("bars",      [sys.executable, "-m", "src.bar_builder"]),
    ("ingestion", [sys.executable, "-m", "src.data_ingestion"]),
]

# Exactly one of these fills orders. Running both would fill every order twice and
# double-count every trade, so the mode selects one and only one.
EXECUTORS = {
    "paper": ("paper", [sys.executable, "-m", "src.paper_broker"]),
    "live": ("execution", [sys.executable, "-m", "src.execution_engine"]),
    "dry_run": ("execution", [sys.executable, "-m", "src.execution_engine"]),
}


def modules_for_mode(mode: str) -> List[tuple]:
    executor = EXECUTORS.get(mode, EXECUTORS["dry_run"])
    # Insert the executor after storage so it is listening before signals flow.
    return [BASE_MODULES[0], executor] + BASE_MODULES[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all LIGERBOT modules together")
    parser.add_argument("--simulate", action="store_true",
                        help="run the data feed in simulation mode (no broker)")
    args = parser.parse_args()

    print(config.summary())
    if not event_bus.ping():
        print("ERROR: Redis not reachable. Start infra with `docker compose up -d`.",
              file=sys.stderr)
        sys.exit(1)

    modules = modules_for_mode(config.TRADING_MODE)
    print(f"Trading mode: {config.TRADING_MODE}")
    if config.TRADING_MODE == "live":
        print("  *** LIVE — real orders will be sent to the broker. ***")
    elif config.TRADING_MODE == "paper":
        print("  Paper: realistic simulated fills (next-bar open, slippage, costs).")
    else:
        print("  Dry run: orders are logged, never filled. Wiring check only.")

    procs: List[subprocess.Popen] = []

    def shutdown(*_):
        print("\nShutting down all modules...")
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
        for p in reversed(procs):
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for name, cmd in modules:
        if name == "ingestion" and args.simulate:
            cmd = cmd + ["--simulate"]
        print(f"Starting {name}: {' '.join(cmd)}")
        procs.append(subprocess.Popen(cmd))
        time.sleep(1)  # small stagger so consumers subscribe before producers start

    print("\nAll modules running. Press Ctrl-C to stop.\n")
    # Wait; if any module dies, report it but keep the others alive.
    while True:
        for (name, _cmd), proc in zip(modules, procs):
            code = proc.poll()
            if code is not None:
                print(f"[warn] module '{name}' exited with code {code}")
        time.sleep(2)


if __name__ == "__main__":
    main()
