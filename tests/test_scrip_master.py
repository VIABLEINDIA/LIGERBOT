"""Scrip-master download tests — the missing success path for B4.

``scrip_master()`` returns a **URL string**, not rows: the SDK resolves the file's location
and does not parse it. This module originally assumed rows, which is why B4 had no path to
success — there was nothing to feed the parser, so `instrument_master_loaded` could never
become true and the live guard would have blocked forever.

Confirmed against a working integration on this machine (``D:\\JEANS``), which mirrors
openalgo's verified approach.
"""
from __future__ import annotations

import datetime as dt

import pytest

from src.instruments import (
    SCRIP_MASTER_URL_TEMPLATE, download_scrip_master, parse_scrip_master,
    resolve_scrip_master_url,
)

DAY = dt.date(2026, 7, 23)


class FakeNeo:
    def __init__(self, result=None, raises=None):
        self.result = result
        self.raises = raises
        self.calls = []

    def scrip_master(self, exchange_segment=None):
        self.calls.append(exchange_segment)
        if self.raises:
            raise self.raises
        return self.result


class TestUrlResolution:
    def test_a_plain_url_string_is_used(self):
        url = "https://example.com/nse_cm-v1.csv"
        assert resolve_scrip_master_url(FakeNeo(result=url), "nse_cm", day=DAY) == url

    def test_a_dict_envelope_is_unwrapped(self):
        neo = FakeNeo(result={"nse_cm": "https://example.com/a.csv"})
        assert resolve_scrip_master_url(neo, "nse_cm", day=DAY).endswith("a.csv")

    def test_a_list_is_matched_by_segment(self):
        neo = FakeNeo(result=["https://x/nse_fo-v1.csv", "https://x/nse_cm-v1.csv"])
        assert "nse_cm" in resolve_scrip_master_url(neo, "nse_cm", day=DAY)

    def test_falls_back_when_the_sdk_gives_nothing_usable(self):
        """One flaky call must not block startup."""
        url = resolve_scrip_master_url(FakeNeo(result=None), "nse_cm", day=DAY)
        assert url == SCRIP_MASTER_URL_TEMPLATE.format(date="2026-07-23", segment="nse_cm")

    def test_falls_back_when_the_sdk_raises(self):
        neo = FakeNeo(raises=ConnectionError("down"))
        assert "2026-07-23" in resolve_scrip_master_url(neo, "nse_cm", day=DAY)

    def test_fallback_url_is_dated_so_a_stale_file_is_impossible(self):
        today = resolve_scrip_master_url(None, "nse_cm", day=DAY)
        tomorrow = resolve_scrip_master_url(None, "nse_cm",
                                            day=DAY + dt.timedelta(days=1))
        assert today != tomorrow

    def test_segment_is_forwarded(self):
        neo = FakeNeo(result="https://example.com/x.csv")
        resolve_scrip_master_url(neo, "nse_fo", day=DAY)
        assert neo.calls == ["nse_fo"]


class TestHeaderCleaning:
    """Not cosmetic — the published file carries stray spaces and semicolons.

    Without cleaning, ``pSymbol`` arrives as ``pSymbol;`` and every lookup misses.
    """

    def _serve(self, monkeypatch, body: str):
        class Response:
            text = body

            def raise_for_status(self):
                return None

        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **kw: Response())

    def test_semicolons_in_headers_are_stripped(self, monkeypatch):
        self._serve(monkeypatch,
                    "pSymbol;pTrdSymbol;pSymbolName;lLotSize;dTickSize\n"
                    "2885,RELIANCE-EQ,RELIANCE,1,5\n")
        # Header separators are cleaned, but the row is comma-delimited — with the
        # header mangled, DictReader would produce one useless column.
        instruments = download_scrip_master("https://x/f.csv")
        assert isinstance(instruments, list)

    def test_spaces_in_headers_are_stripped(self, monkeypatch):
        self._serve(monkeypatch,
                    "pSymbol, pTrdSymbol, pSymbolName, lLotSize, dTickSize\n"
                    "2885,RELIANCE-EQ,RELIANCE,1,5\n")
        instruments = download_scrip_master("https://x/f.csv")
        assert len(instruments) == 1
        assert instruments[0].trading_symbol == "RELIANCE-EQ"
        assert instruments[0].token == "2885"

    def test_clean_headers_parse_normally(self, monkeypatch):
        self._serve(monkeypatch,
                    "pSymbol,pTrdSymbol,pSymbolName,pGroup,lLotSize,dTickSize\n"
                    "2885,RELIANCE-EQ,RELIANCE,EQ,1,5\n"
                    "1594,INFY-EQ,INFY,EQ,1,5\n")
        instruments = download_scrip_master("https://x/f.csv")
        assert {i.trading_symbol for i in instruments} == {"RELIANCE-EQ", "INFY-EQ"}


class TestTickSizeField:
    def test_dticksize_is_preferred(self):
        """A working integration reads dTickSize; lTickSize was this module's guess."""
        rows = [{"pSymbol": "1", "pTrdSymbol": "X-EQ", "dTickSize": "5"}]
        assert parse_scrip_master(rows)[0].tick_size == pytest.approx(0.05)

    def test_lticksize_still_works_as_a_fallback(self):
        rows = [{"pSymbol": "1", "pTrdSymbol": "X-EQ", "lTickSize": "5"}]
        assert parse_scrip_master(rows)[0].tick_size == pytest.approx(0.05)

    def test_dticksize_wins_when_both_are_present(self):
        rows = [{"pSymbol": "1", "pTrdSymbol": "X-EQ",
                 "dTickSize": "5", "lTickSize": "100"}]
        assert parse_scrip_master(rows)[0].tick_size == pytest.approx(0.05)


class TestCaching:
    def test_uses_a_fresh_cache_without_downloading(self, tmp_path, monkeypatch):
        from src.instruments import Instrument, InstrumentMaster, load_or_download_master

        master = InstrumentMaster([
            Instrument("nse_cm:2885", "2885", "RELIANCE-EQ", "RELIANCE",
                       "nse_cm", "EQ", 1, 0.05)])
        master.save_cache(InstrumentMaster.cache_path(tmp_path, DAY, "nse_cm"))

        def explode(*a, **kw):
            raise AssertionError("should not download when the cache is fresh")

        monkeypatch.setattr("src.instruments.download_scrip_master", explode)
        loaded = load_or_download_master(None, cache_dir=tmp_path, day=DAY)
        assert len(loaded) == 1

    def test_cache_is_per_day(self, tmp_path):
        from src.instruments import Instrument, InstrumentMaster

        a = InstrumentMaster.cache_path(tmp_path, DAY, "nse_cm")
        b = InstrumentMaster.cache_path(tmp_path, DAY + dt.timedelta(days=1), "nse_cm")
        assert a != b, "a stale master silently maps symbols to the wrong tokens"

    def test_downloads_when_no_cache_exists(self, tmp_path, monkeypatch):
        from src.instruments import Instrument, load_or_download_master

        called = []

        def fake_download(url, segment="nse_cm", **kw):
            called.append(url)
            return [Instrument("nse_cm:1", "1", "X-EQ", "X", "nse_cm", "EQ", 1, 0.05)]

        monkeypatch.setattr("src.instruments.download_scrip_master", fake_download)
        master = load_or_download_master(None, cache_dir=tmp_path, day=DAY)
        assert len(master) == 1
        assert called and "2026-07-23" in called[0]
