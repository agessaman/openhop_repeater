"""
Tests for default-region scoping of repeater-originated adverts.

Firmware reference (examples/simple_repeater/MyMesh.cpp + RegionMap.cpp):
flood adverts are scoped with the default region's transport key. Public
(implicit-hashtag) regions derive the key as SHA256("#" + name)[:16], while
private ('$') regions use provisioned key material from the TransportKeyStore
(loadKeysFor) — never a name-derived auto key. A private region with no
provisioned key leaves default_scope null, which firmware sends as a plain
unscoped flood.
"""

import base64

from openhop_core.protocol import LocalIdentity
from openhop_core.protocol.constants import ROUTE_TYPE_FLOOD, ROUTE_TYPE_TRANSPORT_FLOOD
from openhop_core.protocol.transport_keys import calc_transport_code, get_auto_key_for

from repeater.utils_packet import create_scoped_advert_packet


class MockStorage:
    def __init__(self, records):
        self._records = records

    def get_transport_keys(self):
        return self._records


def _make_advert(default_region, storage=None):
    return create_scoped_advert_packet(
        local_identity=LocalIdentity(),
        node_name="test-repeater",
        latitude=0.0,
        longitude=0.0,
        flags=0,
        default_region=default_region,
        scope_label="advert",
        storage=storage,
    )


def test_no_default_region_sends_plain_flood():
    packet, scoped_name = _make_advert(None)
    assert scoped_name is None
    assert packet.get_route_type() == ROUTE_TYPE_FLOOD
    assert packet.transport_codes == [0, 0]


def test_public_region_uses_auto_hash_key():
    packet, scoped_name = _make_advert("usa")
    assert scoped_name == "usa"
    assert packet.get_route_type() == ROUTE_TYPE_TRANSPORT_FLOOD
    assert packet.transport_codes[0] == calc_transport_code(get_auto_key_for("#usa"), packet)
    assert packet.transport_codes[1] == 0


def test_private_region_uses_provisioned_key():
    provisioned = bytes(range(16))
    storage = MockStorage(
        [
            {"id": 1, "name": "usa", "transport_key": None},
            {
                "id": 2,
                "name": "$secret",
                "transport_key": base64.b64encode(provisioned).decode("ascii"),
            },
        ]
    )

    packet, scoped_name = _make_advert("$secret", storage=storage)

    assert scoped_name == "$secret"
    assert packet.get_route_type() == ROUTE_TYPE_TRANSPORT_FLOOD
    assert packet.transport_codes[0] == calc_transport_code(provisioned, packet)
    # Must not fall back to hashing the name like a public region
    assert packet.transport_codes[0] != calc_transport_code(
        get_auto_key_for("$secret"), packet
    )


def test_private_region_name_match_is_case_insensitive():
    provisioned = b"\xaa" * 16
    storage = MockStorage(
        [
            {
                "id": 1,
                "name": "$Secret",
                "transport_key": base64.b64encode(provisioned).decode("ascii"),
            }
        ]
    )

    packet, scoped_name = _make_advert("$secret", storage=storage)

    assert scoped_name == "$secret"
    assert packet.transport_codes[0] == calc_transport_code(provisioned, packet)


def test_private_region_without_provisioned_key_sends_plain_flood():
    storage = MockStorage([{"id": 1, "name": "$secret", "transport_key": None}])

    packet, scoped_name = _make_advert("$secret", storage=storage)

    assert scoped_name is None
    assert packet.get_route_type() == ROUTE_TYPE_FLOOD
    assert packet.transport_codes == [0, 0]


def test_private_region_without_storage_sends_plain_flood():
    packet, scoped_name = _make_advert("$secret", storage=None)

    assert scoped_name is None
    assert packet.get_route_type() == ROUTE_TYPE_FLOOD
    assert packet.transport_codes == [0, 0]


def test_private_region_with_corrupt_key_sends_plain_flood():
    storage = MockStorage([{"id": 1, "name": "$secret", "transport_key": "!!!not-base64!!!"}])

    packet, scoped_name = _make_advert("$secret", storage=storage)

    assert scoped_name is None
    assert packet.get_route_type() == ROUTE_TYPE_FLOOD
    assert packet.transport_codes == [0, 0]
