"""Tests for hot-applying a companion's advertised node_name via update_identity.

Regression coverage for the #346 follow-up: changing a companion's node_name in
the openHop UI must reach the running bridge immediately (and be persisted to
SQLite so it survives a restart, since companion_base._load_prefs otherwise
overrides the config node_name on boot).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import cherrypy
import pytest

from repeater.companion.utils import companion_hash_str_from_identity_key
from repeater.web.api_endpoints import APIEndpoints

_KEY = "aa" * 32


def _make_api(config=None):
    api = APIEndpoints.__new__(APIEndpoints)
    api.config = config or {}
    api.daemon_instance = None
    api.send_advert_func = None
    api.event_loop = None
    api.stats_getter = None
    api._config_path = "/tmp/test-config.yaml"
    api.config_manager = MagicMock()
    api.config_manager.save_to_file.return_value = True
    return api


@pytest.fixture
def cherrypy_ctx(monkeypatch):
    request = SimpleNamespace(method="PUT", params={}, json={})
    response = SimpleNamespace(headers={}, status=200)
    monkeypatch.setattr(cherrypy, "request", request, raising=False)
    monkeypatch.setattr(cherrypy, "response", response, raising=False)
    return request, response


def _companion_config(node_name="old"):
    return {
        "identities": {
            "companions": [
                {"name": "c1", "identity_key": _KEY, "settings": {"node_name": node_name}}
            ]
        }
    }


def test_node_name_change_applied_to_live_bridge(cherrypy_ctx):
    request, _ = cherrypy_ctx
    api = _make_api(_companion_config("old"))

    bridge = MagicMock()
    hash_int = int(companion_hash_str_from_identity_key(_KEY), 16)
    api.daemon_instance = SimpleNamespace(
        companion_bridges={hash_int: bridge},
        repeater_handler=SimpleNamespace(storage=SimpleNamespace(sqlite_handler=None)),
    )

    request.json = {"name": "c1", "type": "companion", "settings": {"node_name": "new"}}
    result = api.update_identity()

    assert result["success"] is True
    bridge.set_advert_name.assert_called_once_with("new")
    assert "immediately" in result["message"].lower()
    # Config also reflects the new name.
    assert api.config["identities"]["companions"][0]["settings"]["node_name"] == "new"


def test_node_name_change_persists_to_sqlite_when_no_live_bridge(cherrypy_ctx):
    request, _ = cherrypy_ctx
    api = _make_api(_companion_config("old"))

    sqlite = MagicMock()
    sqlite.companion_load_prefs.return_value = {}
    sqlite.companion_count_contacts.return_value = 0  # pass capacity guard
    api.daemon_instance = SimpleNamespace(
        companion_bridges={},  # bridge not running
        repeater_handler=SimpleNamespace(storage=SimpleNamespace(sqlite_handler=sqlite)),
    )

    request.json = {"name": "c1", "type": "companion", "settings": {"node_name": "new"}}
    result = api.update_identity()

    assert result["success"] is True
    hash_str = companion_hash_str_from_identity_key(_KEY)
    sqlite.companion_save_prefs.assert_called_once()
    saved_hash, saved_prefs = sqlite.companion_save_prefs.call_args[0]
    assert saved_hash == hash_str
    assert saved_prefs["node_name"] == "new"
    # Not applied live, so the user is told a restart is required.
    assert "restart" in result["message"].lower()


def test_invalid_node_name_rejected(cherrypy_ctx):
    request, _ = cherrypy_ctx
    api = _make_api(_companion_config("old"))
    api.daemon_instance = SimpleNamespace(
        companion_bridges={},
        repeater_handler=SimpleNamespace(storage=SimpleNamespace(sqlite_handler=None)),
    )

    # 32+ byte name (over the 31-byte advert limit) must be rejected.
    request.json = {"name": "c1", "type": "companion", "settings": {"node_name": "x" * 40}}
    result = api.update_identity()

    assert result["success"] is False
    api.config_manager.save_to_file.assert_not_called()


def test_settings_update_without_node_name_change_no_live_apply(cherrypy_ctx):
    request, _ = cherrypy_ctx
    api = _make_api(_companion_config("keep"))

    bridge = MagicMock()
    hash_int = int(companion_hash_str_from_identity_key(_KEY), 16)
    api.daemon_instance = SimpleNamespace(
        companion_bridges={hash_int: bridge},
        repeater_handler=SimpleNamespace(storage=SimpleNamespace(sqlite_handler=None)),
    )

    request.json = {"name": "c1", "type": "companion", "settings": {"tcp_port": 5051}}
    result = api.update_identity()

    assert result["success"] is True
    bridge.set_advert_name.assert_not_called()
    assert "restart" in result["message"].lower()
