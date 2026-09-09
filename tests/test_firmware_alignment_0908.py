"""Firmware telemetry parity, retaining the intentional remote MCU policy."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openhop_core.companion.constants import PUSH_CODE_TELEMETRY_RESPONSE

from repeater.companion.frame_server import CompanionFrameServer
from repeater.handler_helpers.protocol_request import ProtocolRequestHelper

GPS = bytes.fromhex("01880030d4ff9e58fffe89")  # channel 1
VOLTAGE_FLOOR = bytes.fromhex("01740000")  # 0.0 V, no battery getter
VOLTAGE_4200 = bytes.fromhex("017401a3")  # 4.2 V, float32-truncated like the firmware
MCU = bytes.fromhex("016700fa")  # channel 1, 25.0 C
MODEM_TEMP_CH2 = bytes.fromhex("026700fa")  # die temp encoded as a sensor channel
EXTERNAL_CH2 = bytes.fromhex("026700c8")  # 20.0 C
EXTERNAL_CH3 = bytes.fromhex("036700c8")


def location_reading():
    return {
        "ok": True,
        "data": {"latitude": 1.25, "longitude": -2.5, "altitude_m": -3.75, "fix_valid": True},
    }


def modem_reading():
    """Production shape: openhop_modem.py aliases die_temperature_c onto temperature_c."""
    reading = location_reading()
    reading["data"].update({"die_temperature_c": 25.0, "temperature_c": 25.0})
    return reading


def readings():
    return [modem_reading(), {"ok": True, "data": {"temperature_c": 20.0}}]


def server_for(sensor_readings, batt_getter=None):
    bridge = Mock()
    bridge.get_public_key.return_value = bytes(range(32))
    return CompanionFrameServer(
        bridge,
        "hash",
        port=0,
        batt_getter=batt_getter,
        sensor_manager=SimpleNamespace(get_summary=lambda: {"readings": sensor_readings}),
    )


@pytest.mark.parametrize(
    "guest,mask,expected",
    [
        (False, 0, GPS + MODEM_TEMP_CH2 + EXTERNAL_CH3),
        (False, 2, MODEM_TEMP_CH2 + EXTERNAL_CH3),
        (False, 4, GPS),
        (False, 6, b""),
        (True, 0, b""),
    ],
)
def test_remote_location_permission_and_order(guest, mask, expected):
    """Remote replies keep the die temp on its sensor channel: no channel-1 MCU slot."""
    manager = SimpleNamespace(get_summary=lambda: {"readings": readings()})
    helper = ProtocolRequestHelper(Mock(), Mock(), sensor_manager=manager)
    client = SimpleNamespace(is_guest=lambda: guest)
    assert helper._handle_get_telemetry(client, 1, bytes([mask])) == VOLTAGE_FLOOR + expected


def test_companion_self_telemetry_reports_die_temperature_once():
    """Channel 1 owns the MCU slot, so the temperature_c alias must not repeat it."""
    server = server_for(readings(), batt_getter=lambda: 4200)
    frames = []
    server._write_frame = frames.append
    server._push_self_telemetry()
    assert frames == [
        bytes([PUSH_CODE_TELEMETRY_RESPONSE, 0])
        + bytes(range(6))
        + VOLTAGE_4200
        + MCU
        + GPS
        + EXTERNAL_CH2
    ]


def test_die_temperature_alias_does_not_take_a_sensor_channel():
    """Firmware's MCU is the board, never an entry in sensors.querySensors()."""
    server = server_for([modem_reading()])
    assert server._get_mcu_temperature_c() == 25.0
    assert server._get_self_telemetry_lpp() == GPS


def test_distinct_ambient_probe_survives_the_alias_strip():
    """Only the mirrored value is dropped; a separate probe keeps its channel."""
    modem = modem_reading()
    modem["data"]["temperature_c"] = 20.0
    server = server_for([modem])
    assert server._get_mcu_temperature_c() == 25.0
    assert server._get_self_telemetry_lpp() == GPS + EXTERNAL_CH2


@pytest.mark.parametrize(
    "updates",
    [
        {"fix_valid": False},
        {"gps_enabled": False},
        {"latitude": None},
        {"latitude": float("nan")},
        {"longitude": float("inf")},
        {"altitude_m": 999999},
        {"latitude": 91},
    ],
)
def test_invalid_or_disabled_location_is_omitted(updates):
    reading = location_reading()
    reading["data"].update(updates)
    assert ProtocolRequestHelper.encode_sensor_telemetry([reading], 0xFF) == b""


@pytest.mark.parametrize("temp", [None, float("nan"), float("inf"), "bad"])
def test_unavailable_mcu_leaves_the_sensor_channels_untouched(temp):
    """With no MCU slot to duplicate, the modem keeps reporting on its own channel."""
    values = readings()
    values[0]["data"]["die_temperature_c"] = temp
    server = server_for(values)
    assert server._get_mcu_temperature_c() is None
    assert server._get_self_telemetry_lpp() == GPS + MODEM_TEMP_CH2 + EXTERNAL_CH3


def test_failed_sensor_manager_keeps_voltage_floor():
    bridge = Mock()
    bridge.get_public_key.return_value = bytes(32)
    server = CompanionFrameServer(
        bridge,
        "hash",
        port=0,
        sensor_manager=Mock(get_summary=Mock(side_effect=RuntimeError("offline"))),
    )
    frames = []
    server._write_frame = frames.append
    server._push_self_telemetry()
    assert frames == [bytes([PUSH_CODE_TELEMETRY_RESPONSE, 0]) + bytes(6) + VOLTAGE_FLOOR]
