"""Per-radio airtime attribution for Fabric bridges.

Covers the reporting contract of ``get_airtime_buckets``: which radio each
stored packet is charged to, that each radio's own LoRa profile is used, and
that single-radio installations keep the response they had before.
"""

from __future__ import annotations

import time

import pytest
from openhop_core.protocol.packet_utils import calculate_lora_airtime_ms

from repeater.config import build_radio_profiles
from repeater.data_acquisition.sqlite_handler import AIRTIME_BUCKETS_QUERY, SQLiteHandler

LOCAL = {
    "radio_id": "local",
    "frequency_hz": 869618000,
    "bandwidth_hz": 62500,
    "spreading_factor": 8,
    "coding_rate": 8,
    "preamble_length": 32,
}
LINK = {
    "radio_id": "link",
    "frequency_hz": 864200000,
    "bandwidth_hz": 62500,
    "spreading_factor": 11,
    "coding_rate": 8,
    "preamble_length": 32,
}
BRIDGE = [LOCAL, LINK]

BASE_TS = 1_700_000_000  # fixed so bucket boundaries are deterministic


def _store(handler: SQLiteHandler, **overrides) -> None:
    record = {
        "timestamp": BASE_TS,
        "type": 1,
        "route": 0,
        "length": 64,
        "transmitted": False,
        "is_duplicate": False,
    }
    record.update(overrides)
    handler.store_packet(record)


def _buckets(handler: SQLiteHandler, profiles=None, bucket_seconds: int = 60) -> dict:
    return handler.get_airtime_buckets(
        start_timestamp=BASE_TS - 3600,
        end_timestamp=BASE_TS + 3600,
        bucket_seconds=bucket_seconds,
        radio_profiles=profiles,
    )


def _radio(result: dict, radio_id: str) -> dict:
    return next(r for r in result["radios"] if r["radio_id"] == radio_id)


def _expected_ms(profile: dict, length: int) -> float:
    return calculate_lora_airtime_ms(
        length,
        profile["spreading_factor"],
        profile["bandwidth_hz"],
        profile["coding_rate"],
        profile["preamble_length"],
    )


@pytest.fixture
def handler(tmp_path) -> SQLiteHandler:
    return SQLiteHandler(tmp_path)


def test_single_radio_response_stays_backward_compatible(handler):
    _store(handler, length=40)
    _store(handler, length=40, transmitted=True)

    result = _buckets(handler)

    # Legacy fields unchanged in shape and meaning.
    assert result["bucket_seconds"] == 60
    assert result["rx_total"] == 1
    assert result["tx_total"] == 1
    assert len(result["buckets"]) == 1
    bucket = result["buckets"][0]
    assert bucket["timestamp"] == (BASE_TS // 60) * 60
    assert bucket["rx_count"] == 1 and bucket["tx_count"] == 1
    assert bucket["rx_ms"] > 0 and bucket["tx_ms"] > 0

    # ...and the additive fields describe the same single radio.
    assert len(result["radios"]) == 1
    only = result["radios"][0]
    assert only["rx_total"] == 1 and only["tx_total"] == 1
    assert only["buckets"] == result["buckets"]
    assert result["unattributed_rx_count"] == 0
    assert result["unattributed_tx_count"] == 0


def test_null_radio_ids_attribute_to_the_only_radio(handler):
    """Pre-Fabric rows carry no radio ids; one radio leaves nothing to infer."""
    _store(handler, rx_radio_id=None, tx_radio_id=None, tx_radio_ids=None)

    result = _buckets(handler, [LOCAL])

    assert _radio(result, "local")["rx_total"] == 1
    assert result["unattributed_rx_count"] == 0


def test_forwarded_packet_charges_ingress_rx_and_every_successful_egress(handler):
    _store(
        handler,
        length=50,
        transmitted=True,
        rx_radio_id="local",
        tx_radio_id="link",
        tx_radio_ids=["link", "local"],
    )

    result = _buckets(handler, BRIDGE)
    local = _radio(result, "local")
    link = _radio(result, "link")

    assert local["rx_total"] == 1 and local["tx_total"] == 1
    assert link["rx_total"] == 0 and link["tx_total"] == 1
    assert local["buckets"][0]["rx_ms"] == pytest.approx(_expected_ms(LOCAL, 50))
    assert local["buckets"][0]["tx_ms"] == pytest.approx(_expected_ms(LOCAL, 50))
    assert link["buckets"][0]["tx_ms"] == pytest.approx(_expected_ms(LINK, 50))
    # Combined legacy series is the sum of both radios.
    assert result["rx_total"] == 1
    assert result["tx_total"] == 2


def test_duplicate_reception_charges_only_the_radio_that_heard_it(handler):
    _store(handler, rx_radio_id="link", is_duplicate=True, drop_reason="duplicate")

    result = _buckets(handler, BRIDGE)

    assert _radio(result, "link")["rx_total"] == 1
    assert _radio(result, "local")["rx_total"] == 0
    assert _radio(result, "local")["buckets"] == []


def test_local_origin_packet_contributes_tx_only(handler):
    _store(handler, transmitted=True, rx_radio_id=None, tx_radio_ids=["local"])

    result = _buckets(handler, BRIDGE)

    assert _radio(result, "local")["tx_total"] == 1
    assert _radio(result, "local")["rx_total"] == 0
    assert result["rx_total"] == 0


def test_partial_fanout_counts_only_successful_egresses(handler):
    """``tx_radio_ids`` holds the radios that actually sent; a refused radio is absent."""
    _store(
        handler,
        transmitted=True,
        rx_radio_id="link",
        tx_radio_id="local",
        tx_radio_ids=["local"],
    )

    result = _buckets(handler, BRIDGE)

    assert _radio(result, "local")["tx_total"] == 1
    assert _radio(result, "link")["tx_total"] == 0
    assert _radio(result, "link")["rx_total"] == 1


def test_scalar_tx_radio_id_supports_pre_migration_rows(handler):
    _store(handler, transmitted=True, rx_radio_id="local", tx_radio_id="link", tx_radio_ids=None)

    result = _buckets(handler, BRIDGE)

    assert _radio(result, "link")["tx_total"] == 1
    assert _radio(result, "local")["rx_total"] == 1
    assert result["unattributed_tx_count"] == 0


def test_unknown_radio_attribution_is_reported_not_guessed(handler):
    _store(handler, rx_radio_id=None)  # Fabric history with no ingress id
    _store(handler, transmitted=True, rx_radio_id=None, tx_radio_id=None, tx_radio_ids=None)
    _store(handler, rx_radio_id="decommissioned")  # radio no longer configured

    result = _buckets(handler, BRIDGE)

    assert result["unattributed_rx_count"] == 2
    assert result["unattributed_tx_count"] == 1
    for radio in result["radios"]:
        assert radio["rx_total"] == 0 and radio["tx_total"] == 0
    # Counted in the legacy totals, but with no airtime invented for them.
    assert result["rx_total"] == 2 and result["tx_total"] == 1
    assert result["buckets"][0]["rx_ms"] == 0.0


def test_equal_lengths_at_sf8_and_sf11_differ_in_airtime(handler):
    _store(handler, length=64, rx_radio_id="local")
    _store(handler, length=64, rx_radio_id="link")

    result = _buckets(handler, BRIDGE)
    local_ms = _radio(result, "local")["buckets"][0]["rx_ms"]
    link_ms = _radio(result, "link")["buckets"][0]["rx_ms"]

    assert local_ms == pytest.approx(_expected_ms(LOCAL, 64))
    assert link_ms == pytest.approx(_expected_ms(LINK, 64))
    assert link_ms > local_ms * 4  # SF11 is far slower than SF8 at the same bandwidth


def test_per_entry_bandwidth_override_affects_only_that_radio(handler):
    wide_link = dict(LINK, bandwidth_hz=250000)
    _store(handler, length=64, rx_radio_id="local")
    _store(handler, length=64, rx_radio_id="link")

    narrow = _buckets(handler, BRIDGE)
    wide = _buckets(handler, [LOCAL, wide_link])

    assert _radio(wide, "local")["buckets"][0]["rx_ms"] == pytest.approx(
        _radio(narrow, "local")["buckets"][0]["rx_ms"]
    )
    assert _radio(wide, "link")["buckets"][0]["rx_ms"] == pytest.approx(_expected_ms(wide_link, 64))
    assert (
        _radio(wide, "link")["buckets"][0]["rx_ms"] < _radio(narrow, "link")["buckets"][0]["rx_ms"]
    )


def test_unreadable_profile_reports_counts_without_inventing_airtime(handler):
    broken = dict(LINK, spreading_factor=None)
    _store(handler, rx_radio_id="link")

    result = _buckets(handler, [LOCAL, broken])
    link = _radio(result, "link")

    assert link["rx_total"] == 1
    assert link["buckets"][0]["rx_ms"] == 0.0
    assert link["profile"]["spreading_factor"] is None


def test_buckets_are_keyed_by_the_same_boundaries_for_every_radio(handler):
    """Both panels must be directly comparable, so both must share bucket edges."""
    _store(handler, timestamp=BASE_TS + 5, rx_radio_id="local")
    _store(handler, timestamp=BASE_TS + 15, rx_radio_id="link")
    _store(handler, timestamp=BASE_TS + 125, rx_radio_id="link")

    result = _buckets(handler, BRIDGE)
    local_edges = [b["timestamp"] for b in _radio(result, "local")["buckets"]]
    link_edges = [b["timestamp"] for b in _radio(result, "link")["buckets"]]

    assert local_edges == [(BASE_TS // 60) * 60]
    assert link_edges[0] == local_edges[0]
    assert all(edge % 60 == 0 for edge in local_edges + link_edges)
    # Edges come from the same grid, so the UI can index both series by timestamp.
    assert set(link_edges) <= {b["timestamp"] for b in result["buckets"]}


def test_realistic_day_of_traffic_is_served_index_only(handler):
    """A 24 h window is a range scan of the covering index, never the row heap."""
    day_start = BASE_TS - 86400
    for i in range(5000):
        _store(
            handler,
            timestamp=day_start + i * 17,
            length=32 + (i % 40),
            rx_radio_id="local" if i % 2 else "link",
        )
    # Traffic outside the window that must not be read.
    for i in range(2000):
        _store(handler, timestamp=day_start - 10_000 - i, rx_radio_id="local")

    with handler._connect() as conn:
        # The exact query get_airtime_buckets runs, so a new column cannot slip
        # out of the index again without this failing.
        plan = conn.execute(
            "EXPLAIN QUERY PLAN " + AIRTIME_BUCKETS_QUERY, (day_start, BASE_TS)
        ).fetchall()
        legacy_plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT timestamp, length, payload_length, transmitted "
            "FROM packets WHERE timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (day_start, BASE_TS, 50000),
        ).fetchall()
    assert any("COVERING INDEX idx_packets_airtime" in str(tuple(row)) for row in plan), plan
    assert any("COVERING INDEX idx_packets_airtime" in str(tuple(row)) for row in legacy_plan), (
        legacy_plan
    )

    started = time.perf_counter()
    result = handler.get_airtime_buckets(
        start_timestamp=day_start,
        end_timestamp=BASE_TS,
        bucket_seconds=60,
        radio_profiles=BRIDGE,
    )
    elapsed = time.perf_counter() - started

    assert result["rx_total"] == 5000
    assert _radio(result, "local")["rx_total"] + _radio(result, "link")["rx_total"] == 5000
    assert elapsed < 5.0, f"24 h aggregation took {elapsed:.2f}s"


def test_config_profiles_feed_the_aggregator_in_configured_order():
    config = {
        "radios": [
            {
                "id": "local",
                "radio_type": "sx1262",
                "radio": {
                    "frequency": 869618000,
                    "bandwidth": 62500,
                    "spreading_factor": 8,
                    "coding_rate": 8,
                    "preamble_length": 32,
                },
            },
            {
                "id": "link",
                "radio_type": "sx1262",
                "radio": {
                    "frequency": 864200000,
                    "bandwidth": 62500,
                    "spreading_factor": 11,
                    "coding_rate": 8,
                    "preamble_length": 32,
                },
            },
        ],
        "fabric": {"tx_mode": "bridge", "default_radio": "local"},
    }

    profiles = build_radio_profiles(config)

    assert [p["radio_id"] for p in profiles] == ["local", "link"]
    assert profiles[0]["spreading_factor"] == 8
    assert profiles[1]["spreading_factor"] == 11
    assert profiles[1]["frequency_hz"] == 864200000


def test_partial_radio_config_reports_no_profiles_rather_than_a_smaller_fabric():
    """A one-profile answer would claim every packet belongs to that radio."""
    config = {
        "radios": [
            {"id": "local", "radio_type": "sx1262", "radio": {"spreading_factor": 8}},
            {"radio_type": "sx1262", "radio": {"spreading_factor": 11}},  # no id
        ],
    }

    assert build_radio_profiles(config) == []


def test_unsupported_radio_type_has_no_profile():
    """get_radio_for_board would reject it, so no radio was built from it."""
    assert build_radio_profiles({"radio_type": "sx1280", "radio": {"spreading_factor": 8}}) == []
    assert build_radio_profiles({"radio_type": "none"}) == []


# --- Stored airtime: history keeps the settings each packet was carried on ----


def test_retuning_the_radio_does_not_restate_stored_history(handler):
    """The reported bug: history rescaled itself to whatever the radio is now.

    A packet stored with its measured time on air keeps that figure when the
    node is retuned, instead of yesterday's traffic quadrupling because the
    operator moved from SF8 to SF11.
    """
    _store(handler, rx_radio_id="local", airtime_ms=123.5)

    as_stored = _buckets(handler, [LOCAL])
    retuned = _buckets(handler, [dict(LOCAL, spreading_factor=11)])

    assert _radio(as_stored, "local")["buckets"][0]["rx_ms"] == pytest.approx(123.5)
    assert _radio(retuned, "local")["buckets"][0]["rx_ms"] == pytest.approx(123.5)


def test_rows_without_a_stored_figure_are_still_estimated(handler):
    """Databases written before the column keep the only answer their data supports."""
    _store(handler, rx_radio_id="local")

    result = _buckets(handler, [LOCAL])

    assert _radio(result, "local")["buckets"][0]["rx_ms"] == pytest.approx(_expected_ms(LOCAL, 64))


def test_an_unmeasurable_packet_is_estimated_rather_than_counted_free(handler):
    """The engine reports 0.0 when it could not measure; that is not a free packet."""
    _store(handler, rx_radio_id="local", airtime_ms=0.0)

    result = _buckets(handler, [LOCAL])

    assert _radio(result, "local")["buckets"][0]["rx_ms"] == pytest.approx(_expected_ms(LOCAL, 64))


def test_a_bridged_relay_charges_each_side_its_own_transmission(handler):
    """The stored figure measures the ingress only; the far egress is estimated.

    One packet heard on ``local`` and relayed out of ``link`` is two different
    lengths of transmission. Charging the far side the ingress measurement would
    understate it by the ratio of their modulations.
    """
    _store(
        handler,
        transmitted=True,
        rx_radio_id="local",
        tx_radio_id="link",
        tx_radio_ids=["link"],
        airtime_ms=123.5,
    )

    result = _buckets(handler, BRIDGE)

    assert _radio(result, "local")["buckets"][0]["rx_ms"] == pytest.approx(123.5)
    assert _radio(result, "link")["buckets"][0]["tx_ms"] == pytest.approx(_expected_ms(LINK, 64))


def test_a_node_originated_packet_is_measured_on_the_radio_that_sent_it(handler):
    """With no ingress, the stored figure belongs to the primary egress."""
    _store(
        handler,
        transmitted=True,
        rx_radio_id=None,
        tx_radio_id="link",
        tx_radio_ids=["link"],
        airtime_ms=456.25,
    )

    result = _buckets(handler, BRIDGE)

    assert _radio(result, "link")["buckets"][0]["tx_ms"] == pytest.approx(456.25)
    assert _radio(result, "local")["buckets"] == []


def test_a_single_radio_node_measures_both_directions(handler):
    """One radio carries the reception and the relay, so one figure covers both."""
    _store(
        handler,
        transmitted=True,
        rx_radio_id=None,
        tx_radio_id=None,
        airtime_ms=200.0,
    )

    result = _buckets(handler, [LOCAL])
    bucket = _radio(result, "local")["buckets"][0]

    assert bucket["tx_ms"] == pytest.approx(200.0)
    assert result["buckets"][0]["tx_ms"] == pytest.approx(200.0)


def test_an_unattributable_packet_still_counts_as_zero(handler):
    """A stored figure does not rescue a row no configured radio claims."""
    _store(handler, rx_radio_id="retired", airtime_ms=999.0)

    result = _buckets(handler, BRIDGE)

    assert result["unattributed_rx_count"] == 1
    assert result["buckets"][0]["rx_ms"] == 0.0
