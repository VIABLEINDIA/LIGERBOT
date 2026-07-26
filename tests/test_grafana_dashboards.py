"""Grafana dashboards checked against the schema that actually reaches InfluxDB.

A dashboard is normally untestable: it renders against live data, and nobody has live data
in CI. But the way dashboards fail is not subtle and *is* testable — **a panel that queries
a field nobody writes is silently, permanently empty.** It renders, it has a title, it has
axes, and it shows nothing forever. Worse, an empty panel reads as *"nothing is wrong"*,
which is precisely the wrong message on a risk dashboard.

So these tests do not check that the dashboards look right. They check that every
measurement and field the panels query is one the writer actually archives, and that the
writer archives everything the position book publishes.

**That check found a real defect the moment it was written.** `BatchingInfluxWriter` filters
against a fixed allow-list and discards anything absent without a word, and seven of the
nine fields in a `position_updates` snapshot were being dropped — including
``open_positions`` and ``total_open_risk``, both of which §3.9 explicitly asks the dashboard
to plot. Every one of those panels would have been empty from the day it shipped.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.influx_writer import NUMERIC_FIELDS, TAG_FIELDS, TEXT_FIELDS
from src.storage_logger import STREAM_MEASUREMENTS

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "grafana" / "dashboards"
PROVISIONING = Path(__file__).resolve().parent.parent / "grafana" / "provisioning"

ARCHIVED_FIELDS = set(NUMERIC_FIELDS) | set(TEXT_FIELDS)
ARCHIVED_TAGS = set(TAG_FIELDS)
MEASUREMENTS = set(STREAM_MEASUREMENTS.values())

# Flux `r._field == "x"` / `r._measurement == "x"`.
FIELD_RE = re.compile(r'r\._field\s*==\s*"([^"]+)"')
MEASUREMENT_RE = re.compile(r'r\._measurement\s*==\s*"([^"]+)"')
TAG_RE = re.compile(r'columns:\s*\[([^\]]*)\]')


def dashboards() -> list[tuple[str, dict]]:
    return [(p.name, json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(DASHBOARD_DIR.glob("*.json"))]


def queries(dashboard: dict):
    for panel in dashboard.get("panels", []):
        for target in panel.get("targets", []):
            if target.get("query"):
                yield panel["title"], target["query"]


def all_queries():
    for name, dashboard in dashboards():
        for title, query in queries(dashboard):
            yield name, title, query


# ---------------------------------------------------------------------------
class TestTheyExistAndParse:
    def test_dashboards_are_present(self):
        assert dashboards(), "no dashboards found — provisioning would mount nothing"

    def test_every_dashboard_is_valid_json(self):
        for path in DASHBOARD_DIR.glob("*.json"):
            json.loads(path.read_text(encoding="utf-8"))

    def test_uids_are_unique(self):
        """Two dashboards sharing a uid means provisioning silently keeps one."""
        uids = [d["uid"] for _, d in dashboards()]
        assert len(uids) == len(set(uids))

    def test_every_panel_has_a_title(self):
        for name, dashboard in dashboards():
            for panel in dashboard["panels"]:
                assert panel.get("title"), f"{name}: untitled panel {panel.get('id')}"

    def test_panels_do_not_overlap_or_share_ids(self):
        for name, dashboard in dashboards():
            ids = [p["id"] for p in dashboard["panels"]]
            assert len(ids) == len(set(ids)), f"{name}: duplicate panel ids"


class TestQueriesMatchTheArchive:
    """The check that matters. A field nobody writes renders as an empty panel forever,
    and an empty panel on a risk dashboard reads as "nothing is wrong"."""

    def test_every_queried_measurement_is_archived(self):
        for name, title, query in all_queries():
            for measurement in MEASUREMENT_RE.findall(query):
                assert measurement in MEASUREMENTS, (
                    f"{name} / {title!r} queries measurement {measurement!r}, which "
                    f"storage_logger never writes. Known: {sorted(MEASUREMENTS)}")

    def test_every_queried_field_is_archived(self):
        for name, title, query in all_queries():
            for field in FIELD_RE.findall(query):
                assert field in ARCHIVED_FIELDS, (
                    f"{name} / {title!r} queries field {field!r}, which the Influx writer "
                    f"drops. This panel would be permanently empty.")

    def test_every_grouped_column_is_a_tag(self):
        """Grouping by a field rather than a tag silently returns nothing useful."""
        for name, title, query in all_queries():
            for group in TAG_RE.findall(query):
                for raw in group.split(","):
                    column = raw.strip().strip('"')
                    if not column or column.startswith("_"):
                        continue
                    assert column in ARCHIVED_TAGS, (
                        f"{name} / {title!r} groups by {column!r}, which is not a tag. "
                        f"Tags: {sorted(ARCHIVED_TAGS)}")


class TestTheArchiveCarriesWhatTheBookPublishes:
    """The other direction, and the one that found the defect.

    The writer's allow-list drops anything not named, silently. Nothing was checking it
    against what the position book actually emits.
    """

    def test_no_position_snapshot_field_is_silently_dropped(self):
        from src.position_manager import PositionBook

        snapshot = PositionBook().snapshot()
        # `positions` is a nested list and `session_day` a date — neither belongs in a
        # numeric field, and both are recoverable from the point's own timestamp.
        structural = {"positions", "session_day"}
        dropped = set(snapshot) - ARCHIVED_FIELDS - ARCHIVED_TAGS - structural
        assert not dropped, (
            f"the position book publishes {sorted(dropped)} and the Influx writer "
            f"discards them without a word — any dashboard panel using them is empty")

    @pytest.mark.parametrize("field", [
        "open_positions", "total_open_risk", "gross_exposure", "net_pnl_today",
        "realized_pnl_today", "costs_today", "trade_count",
    ])
    def test_the_fields_design_asks_the_dashboard_to_plot_are_archived(self, field):
        assert field in ARCHIVED_FIELDS

    def test_the_correlation_id_survives_to_the_archive(self):
        """Threading it through five processes is pointless if the permanent record
        drops it — reconstructing a trade afterwards is the whole purpose."""
        assert "correlation_id" in ARCHIVED_FIELDS

    def test_it_is_a_field_not_a_tag(self):
        """Influx tags are indexed. One distinct value per order would explode series
        cardinality, so it is filterable but deliberately not indexed."""
        assert "correlation_id" not in ARCHIVED_TAGS


class TestProvisioningIsSafe:
    def test_no_credential_is_committed(self):
        """A provisioning file with a real token in it is a credential in version
        control — exactly what the CI hygiene job scans for."""
        for path in PROVISIONING.rglob("*.yml"):
            text = path.read_text(encoding="utf-8")
            if "token:" in text:
                assert "${" in text, f"{path.name}: token is not an env reference"

    def test_the_datasource_uses_flux(self):
        text = (PROVISIONING / "datasources" / "influxdb.yml").read_text(encoding="utf-8")
        assert "version: Flux" in text, "the dashboards are written in Flux"

    def test_dashboards_are_read_only(self):
        """Browser edits that are never written back are how a provisioned setup drifts
        from its repository, invisibly, until a rebuild discards them."""
        text = (PROVISIONING / "dashboards" / "dashboards.yml").read_text(encoding="utf-8")
        assert "allowUiUpdates: false" in text

    def test_grafana_is_bound_to_loopback(self):
        """These dashboards show positions, P&L and open risk for a live account."""
        compose = (Path(__file__).resolve().parent.parent
                   / "docker-compose.yml").read_text(encoding="utf-8")
        assert '"127.0.0.1:3000:3000"' in compose

    def test_anonymous_access_is_disabled(self):
        compose = (Path(__file__).resolve().parent.parent
                   / "docker-compose.yml").read_text(encoding="utf-8")
        assert 'GF_AUTH_ANONYMOUS_ENABLED: "false"' in compose


class TestTheGapsAreDeclared:
    """§3.9 asks for six things. Three have no Influx data source, and saying so on the
    dashboard is the difference between an honest gap and a panel that lies by being
    empty."""

    def test_the_operations_dashboard_says_what_it_cannot_show(self):
        text = (DASHBOARD_DIR / "operations.json").read_text(encoding="utf-8")
        for missing in ("Consumer lag", "feed staleness", "health endpoint"):
            assert missing in text, f"the {missing!r} gap is not declared anywhere"

    def test_it_points_at_where_those_actually_live(self):
        text = (DASHBOARD_DIR / "operations.json").read_text(encoding="utf-8")
        assert "9800" in text, "the health endpoint port is not given"
        assert "briefing evening" in text, "reconciliation output is not pointed at"
