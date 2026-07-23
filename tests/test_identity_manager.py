import pytest

from repeater.identity_manager import IdentityConfigurationError, IdentityManager, IdentitySpec


class _FakeIdentity:
    def __init__(self, pubkey: bytes, addr: bytes = b"\xaa\xbb"):
        self._pubkey = pubkey
        self._addr = addr

    def get_public_key(self):
        return self._pubkey

    def get_address_bytes(self):
        return self._addr


def test_identity_manager_register_lookup_and_collision_paths():
    mgr = IdentityManager(config={})
    id_a = _FakeIdentity(bytes([0x11]) + b"A" * 31, addr=b"\x01\x02")
    id_b_collision = _FakeIdentity(bytes([0x11]) + b"B" * 31, addr=b"\x03\x04")

    assert mgr.register_identity("alpha", id_a, {"k": 1}, "repeater") is True
    assert mgr.has_identity(0x11) is True
    assert mgr.get_identity_by_hash(0x11)[0] is id_a
    assert mgr.get_identity_by_name("alpha")[0] is id_a

    assert mgr.register_identity("beta", id_b_collision, {"k": 2}, "room_server") is False


def test_identity_manager_rejects_duplicate_names_without_mutating_state():
    mgr = IdentityManager(config={})
    id_a = _FakeIdentity(bytes([0x11]) + b"A" * 31)
    id_b = _FakeIdentity(bytes([0x22]) + b"B" * 31)

    assert mgr.register_identity("alpha", id_a, {}, "repeater") is True
    assert "already registered" in mgr.registration_error("alpha", id_b, "companion")
    assert mgr.validate_identity("alpha", id_b, "companion") is False
    assert mgr.get_identity_by_hash(0x22) is None


def test_identity_manager_list_and_type_filtering():
    mgr = IdentityManager(config={})
    id_a = _FakeIdentity(bytes([0x22]) + b"A" * 31)
    id_b = _FakeIdentity(bytes([0x33]) + b"B" * 31)

    mgr.register_identity("rep-main", id_a, {"x": 1}, "repeater")
    mgr.register_identity("room-a", id_b, {"y": 2}, "room_server")

    listed = mgr.list_identities()
    assert len(listed) == 2
    assert any(item["hash"] == "0x22" and item["name"] == "repeater:rep-main" for item in listed)
    assert any(item["hash"] == "0x33" and item["type"] == "room_server" for item in listed)

    assert mgr.has_identity_type("repeater") is True
    assert mgr.has_identity_type("room_server") is True
    assert mgr.has_identity_type("unknown") is False

    by_type = mgr.get_identities_by_type("room_server")
    assert len(by_type) == 1
    assert by_type[0][0] == "room-a"


def test_identity_manager_list_handles_none_identity_fields():
    mgr = IdentityManager(config={})
    mgr.named_identities["ghost"] = (None, {}, "repeater")

    listed = mgr.list_identities()
    assert listed[0]["address"] == "N/A"
    assert listed[0]["public_key"] is None


def test_validate_specs_rejects_intra_batch_same_namespace_hash_collision():
    """Two identities in the same routing namespace (here, two companions)
    cannot share a one-byte prefix."""
    mgr = IdentityManager(config={})
    id_a = _FakeIdentity(bytes([0x11]) + b"A" * 31)
    id_b = _FakeIdentity(bytes([0x11]) + b"B" * 31)

    with pytest.raises(IdentityConfigurationError, match="one-byte public-key prefix"):
        mgr.validate_specs(
            [
                IdentitySpec("alpha", id_a, {}, "companion"),
                IdentitySpec("beta", id_b, {}, "companion"),
            ]
        )


def test_validate_specs_allows_intra_batch_cross_namespace_hash_collision():
    """A companion and a server-side identity may share a one-byte prefix: the
    receive path offers a colliding packet to both and MAC verification decides
    the owner."""
    mgr = IdentityManager(config={})
    repeater_id = _FakeIdentity(bytes([0x11]) + b"A" * 31)
    companion_id = _FakeIdentity(bytes([0x11]) + b"B" * 31)

    mgr.validate_specs(
        [
            IdentitySpec("alpha", repeater_id, {}, "repeater"),
            IdentitySpec("beta", companion_id, {}, "companion"),
        ]
    )


def test_validate_specs_rejects_intra_batch_duplicate_name():
    mgr = IdentityManager(config={})
    id_a = _FakeIdentity(bytes([0x11]) + b"A" * 31)
    id_b = _FakeIdentity(bytes([0x22]) + b"B" * 31)

    with pytest.raises(IdentityConfigurationError, match="repeater:alpha"):
        mgr.validate_specs(
            [
                IdentitySpec("alpha", id_a, {}, "repeater"),
                IdentitySpec("alpha", id_b, {}, "companion"),
            ]
        )


def test_validate_specs_rejects_registered_collisions_without_mutation():
    mgr = IdentityManager(config={})
    id_a = _FakeIdentity(bytes([0x11]) + b"A" * 31)
    id_hash_collision = _FakeIdentity(bytes([0x11]) + b"B" * 31)
    id_name_collision = _FakeIdentity(bytes([0x22]) + b"C" * 31)

    assert mgr.register_identity("alpha", id_a, {}, "repeater") is True

    # Same-namespace (server/server) prefix collision against a registered id.
    with pytest.raises(IdentityConfigurationError, match="conflicts"):
        mgr.validate_specs([IdentitySpec("beta", id_hash_collision, {}, "room_server")])
    with pytest.raises(IdentityConfigurationError, match="already registered"):
        mgr.validate_specs([IdentitySpec("alpha", id_name_collision, {}, "companion")])

    # Validation never registers anything.
    assert mgr.get_identity_by_hash(0x22) is None
    assert mgr.get_identity_by_name("beta") is None


def test_validate_specs_accepts_distinct_batch():
    mgr = IdentityManager(config={})
    mgr.validate_specs(
        [
            IdentitySpec("alpha", _FakeIdentity(bytes([0x11]) + b"A" * 31), {}, "repeater"),
            IdentitySpec("beta", _FakeIdentity(bytes([0x22]) + b"B" * 31), {}, "room_server"),
            IdentitySpec("gamma", _FakeIdentity(bytes([0x33]) + b"C" * 31), {}, "companion"),
        ]
    )
