"""Probe what Kotak Neo actually gives us. **Run this before anything else in Phase 1.**

D5 chose Kotak as the only history source, and the plan explicitly depends on a number
nobody has measured yet: how much history the API will actually return, at what
granularity. Everything downstream — whether the 200-trade / 2-year validation gates in
DESIGN.md 2.6 are reachable at all, or whether the paper period must stretch to 40+
sessions instead — turns on the answer.

It also dumps a real ``limits()`` response, which settles the open item in DESIGN.md 5.3:
the field names in ``src/account.py`` are currently **guesses**, and equity sizing depends
on them being right.

This is a throwaway diagnostic, not part of the trading path. It places no orders and
changes no state — it only reads. Run it, read the summary, and paste the findings back::

    python -m tools.probe_kotak_history
    python -m tools.probe_kotak_history --token 2885 --days 400

Requires working credentials in ``.env`` and the ``neo_api_client`` package.
"""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import market_calendar as cal  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [probe] %(message)s")
log = logging.getLogger("ligerbot.probe")

RULE = "=" * 78

# Method names that have appeared on the Neo SDK across versions. We do not know which
# (if any) this build exposes, so we discover rather than assume.
HISTORY_METHOD_CANDIDATES = [
    "history", "get_history", "historical_data", "candle_data", "get_candle_data",
    "chart", "ohlc", "get_ohlc", "intraday_data",
]


def _describe_callable(obj: Any, name: str) -> str:
    member = getattr(obj, name, None)
    if member is None or not callable(member):
        return f"  {name}: (absent)"
    try:
        return f"  {name}{inspect.signature(member)}"
    except (TypeError, ValueError):
        return f"  {name}(...)"


def introspect(client: Any) -> Dict[str, Any]:
    """List what the SDK actually exposes.

    The whole point of this script is that we do not know the API surface, so the first
    thing it does is look rather than guess.
    """
    print(f"\n{RULE}\n1. SDK surface\n{RULE}")
    public = sorted(n for n in dir(client) if not n.startswith("_") and callable(getattr(client, n, None)))
    print(f"Callable methods on the client ({len(public)}):")
    for name in public:
        print(f"  - {name}")

    print("\nHistory-shaped candidates:")
    found: List[str] = []
    for name in HISTORY_METHOD_CANDIDATES:
        line = _describe_callable(client, name)
        print(line)
        if "(absent)" not in line:
            found.append(name)

    if not found:
        print("\n  !! No history-shaped method found on this SDK build.")
        print("     If this holds, Kotak cannot serve historical bars at all and D5 must")
        print("     be revisited — self-recording forward becomes the ONLY source, which")
        print("     means Phase 2 validation waits months. Report this back immediately.")
    return {"all_methods": public, "history_candidates": found}


def dump_limits(client: Any) -> Dict[str, Any]:
    """Dump the real limits() response to pin down the equity field names."""
    print(f"\n{RULE}\n2. limits() — resolves the account.py field mapping\n{RULE}")
    try:
        payload = client.limits()
    except Exception as exc:
        print(f"  limits() failed: {exc}")
        return {"error": str(exc)}

    print(json.dumps(payload, indent=2, default=str)[:4000])

    from src.account import CASH_FIELDS, MTM_FIELDS, parse_limits

    keys = set(payload) if isinstance(payload, dict) else set()
    for wrapper in ("data", "Data", "result"):
        inner = payload.get(wrapper) if isinstance(payload, dict) else None
        if isinstance(inner, dict):
            keys |= set(inner)
        elif isinstance(inner, list) and inner and isinstance(inner[0], dict):
            keys |= set(inner[0])

    print(f"\n  Cash field candidates {CASH_FIELDS}")
    print(f"    matched: {sorted(keys & set(CASH_FIELDS)) or 'NONE'}")
    print(f"  MTM field candidates {MTM_FIELDS}")
    print(f"    matched: {sorted(keys & set(MTM_FIELDS)) or 'NONE'}")

    try:
        snapshot = parse_limits(payload)
        print(f"\n  Parsed OK -> equity {snapshot.equity:,.2f} "
              f"(cash {snapshot.cash:,.2f} + MTM {snapshot.unrealized_mtm:,.2f})")
        print("  Sanity-check this against the app. If it is wrong, every position this")
        print("  bot ever sizes will be wrong by the same factor.")
    except Exception as exc:
        print(f"\n  !! parse_limits FAILED: {exc}")
        print("     Update CASH_FIELDS / MTM_FIELDS in src/account.py with the real names")
        print("     from the dump above. Until then the bot correctly refuses to trade.")
    return {"raw": payload}


def dump_shapes(client: Any) -> Dict[str, Any]:
    """Dump the response shape of every endpoint whose fields we still guess at.

    One credentialed run should settle every remaining unknown, not just history depth.
    Two of these are the last places in the codebase where field names are unverified:

    ``limits()``
        The equity mapping in ``src/account.py``. Checked across four projects on this
        machine and **none** of them had identified these fields — one returns the raw
        dict unexamined, another flags them as unverified in its README. A wrong mapping
        mis-sizes every trade by the same factor, silently.
    ``scrip_master()`` / ``search_scrip()``
        The instrument master (defect B4). No project here has a working implementation —
        the nearest is a ``NotImplementedError`` with a TODO.

    ``order_report()`` field names *are* corroborated by two independent integrations, so
    this confirms rather than discovers.
    """
    print(f"\n{RULE}\n4. Response shapes for the fields we still guess at\n{RULE}")
    out: Dict[str, Any] = {}

    probes = [
        ("positions", lambda: client.positions(), "src/position_manager.py mapping"),
        ("order_report", lambda: client.order_report(),
         "src/order_state.py mapping (corroborated, confirming)"),
        ("holdings", lambda: client.holdings(), "informational"),
    ]
    for name, call, why in probes:
        print(f"\n--- {name}()  [{why}] ---")
        try:
            response = call()
        except Exception as exc:
            print(f"    failed: {str(exc)[:200]}")
            out[name] = {"error": str(exc)[:300]}
            continue
        out[name] = response
        _describe_shape(response)

    # scrip_master returns file URLs per segment in some versions and rows in others.
    print(f"\n--- scrip_master(exchange_segment='nse_cm')  [instrument master, B4] ---")
    try:
        response = client.scrip_master(exchange_segment="nse_cm")
        out["scrip_master"] = response
        _describe_shape(response)
        print("    ^ if this is a URL, the instrument master downloads and parses it;")
        print("      if rows, parse them directly. Either way B4 becomes buildable.")
    except Exception as exc:
        print(f"    failed: {str(exc)[:200]}")
        out["scrip_master"] = {"error": str(exc)[:300]}

    print(f"\n--- search_scrip(exchange_segment='nse_cm', symbol='RELIANCE') ---")
    try:
        response = client.search_scrip(exchange_segment="nse_cm", symbol="RELIANCE")
        out["search_scrip"] = response
        _describe_shape(response)
    except Exception as exc:
        print(f"    failed: {str(exc)[:200]}")
        out["search_scrip"] = {"error": str(exc)[:300]}

    return out


def _describe_shape(response: Any, indent: str = "    ") -> None:
    """Print enough structure to pin field names, without dumping account data.

    Values are shown only for the first row, and truncated — the goal is the *keys*.
    """
    if isinstance(response, dict):
        print(f"{indent}dict with keys: {sorted(response)[:20]}")
        for wrapper in ("data", "Data", "result"):
            inner = response.get(wrapper)
            if isinstance(inner, dict):
                print(f"{indent}  ['{wrapper}'] dict keys: {sorted(inner)[:25]}")
            elif isinstance(inner, list) and inner:
                print(f"{indent}  ['{wrapper}'] list[{len(inner)}]")
                if isinstance(inner[0], dict):
                    print(f"{indent}    row keys: {sorted(inner[0])[:25]}")
    elif isinstance(response, list):
        print(f"{indent}list[{len(response)}]")
        if response and isinstance(response[0], dict):
            print(f"{indent}  row keys: {sorted(response[0])[:25]}")
    else:
        text = str(response)
        print(f"{indent}{type(response).__name__}: {text[:200]}")


def probe_depth(
    client: Any, method_name: str, token: str, segment: str, max_days: int
) -> Dict[str, Any]:
    """Binary-search how far back the endpoint will actually serve data."""
    print(f"\n{RULE}\n3. Historical depth via {method_name}()\n{RULE}")
    method = getattr(client, method_name)
    today = dt.date.today()
    findings: Dict[str, Any] = {"method": method_name, "attempts": []}

    def attempt(days_back: int) -> Optional[int]:
        """Rows returned for a window ending today and starting ``days_back`` ago."""
        start = today - dt.timedelta(days=days_back)
        started = time.monotonic()
        for kwargs in (
            {"exchange_segment": segment, "instrument_token": token,
             "from_date": start.isoformat(), "to_date": today.isoformat(),
             "interval": "1"},
            {"exchange_segment": segment, "instrument_token": token,
             "from_date": start.strftime("%d-%m-%Y"), "to_date": today.strftime("%d-%m-%Y"),
             "interval": "1minute"},
            {"instrument_token": token, "from_date": start.isoformat(),
             "to_date": today.isoformat()},
        ):
            try:
                response = method(**kwargs)
            except TypeError:
                continue  # wrong signature; try the next shape
            except Exception as exc:
                findings["attempts"].append(
                    {"days_back": days_back, "error": str(exc)[:200]})
                return None
            elapsed = time.monotonic() - started
            rows = _row_count(response)
            findings["attempts"].append({
                "days_back": days_back, "rows": rows,
                "seconds": round(elapsed, 2), "kwargs": list(kwargs),
            })
            print(f"  {days_back:>4}d back -> {rows if rows is not None else '?':>7} rows "
                  f"in {elapsed:.2f}s")
            return rows
        findings["attempts"].append(
            {"days_back": days_back, "error": "no accepted call signature"})
        print(f"  {days_back:>4}d back -> no accepted call signature")
        return None

    deepest = 0
    for days_back in (1, 5, 30, 90, 180, 365, 730, max_days):
        if days_back > max_days:
            break
        rows = attempt(days_back)
        if rows:
            deepest = days_back
        time.sleep(0.5)  # be polite; we are also measuring rate limits

    findings["deepest_days_with_data"] = deepest
    print(f"\n  Deepest window that returned data: {deepest} calendar days")
    _interpret_depth(deepest)
    return findings


def _row_count(response: Any) -> Optional[int]:
    if response is None:
        return 0
    if isinstance(response, list):
        return len(response)
    if isinstance(response, dict):
        for key in ("data", "Data", "result", "candles", "records"):
            value = response.get(key)
            if isinstance(value, list):
                return len(value)
        return 0
    return None


def _interpret_depth(days: int) -> None:
    """Translate the measurement into the decision it drives."""
    trading_days = int(days * 250 / 365)
    print(f"  ~= {trading_days} trading days\n")
    if days == 0:
        print("  VERDICT: no usable history. D5 must be revisited — self-recording")
        print("  forward becomes the only source and Phase 2 waits months for data.")
    elif trading_days < 125:
        print("  VERDICT: under ~6 months. Below what DESIGN.md D5 assumed. The 200-trade")
        print("  out-of-sample gate is unlikely to be reachable from backfill alone;")
        print("  lean on universe breadth and plan for the 40+ session paper period.")
    elif trading_days < 250:
        print("  VERDICT: roughly 6-12 months. Workable but single-regime. Widen the")
        print("  universe for trade count and keep the extended paper period.")
    else:
        print("  VERDICT: a year or more. Better than D5 assumed — the standard gates in")
        print("  DESIGN.md 2.6 may be reachable from backfill. Confirm regime diversity.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Kotak Neo's historical data API")
    parser.add_argument("--token", default="2885", help="instrument token (default: RELIANCE)")
    parser.add_argument("--segment", default="nse_cm")
    parser.add_argument("--days", type=int, default=1095, help="deepest window to try")
    parser.add_argument("--out", default="state/kotak_probe.json")
    args = parser.parse_args()

    print(RULE)
    print("Kotak Neo capability probe — read-only, places no orders")
    print(f"Today: {dt.date.today()}  |  last trading day: "
          f"{max(cal.trading_days_between(dt.date.today() - dt.timedelta(days=10), dt.date.today()), default='?')}")
    print(RULE)

    try:
        from src.auth import authenticate_neo
        client = authenticate_neo()
    except Exception as exc:
        print(f"\nAuthentication failed: {exc}\n")
        print("Set KOTAK_* credentials in .env and install the SDK:")
        print('  pip install "git+https://github.com/Kotak-Neo/Kotak-neo-api-v2.git'
              '@v2.0.2#egg=neo_api_client"')
        raise SystemExit(1)

    findings: Dict[str, Any] = {"probed_at": dt.datetime.now().isoformat()}
    findings["sdk"] = introspect(client)
    findings["limits"] = dump_limits(client)

    findings["shapes"] = dump_shapes(client)

    candidates = findings["sdk"]["history_candidates"]
    if candidates:
        findings["history"] = probe_depth(
            client, candidates[0], args.token, args.segment, args.days)
    else:
        findings["history"] = {"available": False}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")

    print(f"\n{RULE}\nFindings written to {out}")
    print("Report back: (1) the depth verdict, (2) whether parse_limits succeeded,")
    print("(3) your broker's actual brokerage plan (DESIGN.md 5.3 item 1).")
    print(RULE)


if __name__ == "__main__":
    main()
